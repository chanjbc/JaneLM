import argparse
import kagglehub
import math
from pathlib import Path
import pickle
import sys
from tqdm import tqdm

from dataclasses import dataclass
from typing import Dict, Tuple
from pydantic import BaseModel, Field, model_validator

import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F

from model import ModelConfig, JaneLM



class TrainConfig(BaseModel):
    """
    Config class containing training parameters:

    B: batch size
    B_split: option for splitting batch size to reduce VRAM (must have B % B_split == 0)
    max_iters: maximum number of training epochs
    """
    B: int = Field(default=32, gt=0)
    B_split: int = Field(default=2, ge=1)
    max_iters: int = Field(default=15_000, gt=0)
    eval_interval: int = Field(default=500, gt=0)
    eval_iters: int = Field(default=250, gt=0)
    loss_tolerance: float = Field(default=0.075, gt=0)
    train_test_split: float = Field(default=0.9, ge=0, le=1)

    # LR scheduling params
    max_lr: float = Field(default=1e-3, gt=0)
    min_lr: float = Field(default=1e-6, gt=0)
    warmup_iters: int = Field(default=100, ge=0)
    lr_decay_iters: int = Field(default=14_900, gt=0)

    @model_validator(mode="after")
    def validate_B_split(self) -> "TrainConfig":
        if self.B < self.B_split:
            raise ValueError(f"Constraint failed: B({self.B}) >= B_split({self.B_split}) != True")
        if self.B % self.B_split != 0:
            raise ValueError(f"Constraint failed: B({self.B}) % B_split({self.B}) != 0")

        if self.max_lr < self.min_lr:
            raise ValueError(f"Constraint failed: max_lr({self.max_lr}) >= min_lr({self.min_lr}) != True")
        if self.warmup_iters > self.max_iters:
            raise ValueError(f"Constraint failed: warmup_iters({self.warmup_iters}) <= max_iters({self.max_iters}) != True")
        if self.lr_decay_iters > self.max_iters:
            raise ValueError(f"Constraint failed: lr_decay_iters({self.lr_decay_iters}) <= max_iters({self.max_iters}) != True")

        return self



class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_config: TrainConfig,
        model_config: ModelConfig,
        models_dir: Path,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        model_file: str = "model.pth",
    ):
        self.model = model.to(device)
        self.train_config = train_config
        self.model_config = model_config
        self.models_dir = models_dir
        self.device = device
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=train_config.max_lr
        )
        self.model_file = model_file

        self.ptdtype = ptdtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        self.ctx = torch.amp.autocast(device_type=device, dtype=ptdtype)

    def get_batch(
        self, 
        split: str,
        split_size: int, 
        train_data: torch.Tensor, 
        val_data: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get batch of data for training or validation, each containing context_size-length sequences sampled randomly.
        """
        data = train_data if split == "train" else val_data

        # Generate random starting indices
        ix = torch.randint(
            len(data) - self.model_config.T, (split_size,)
        )
        # Get sequences of context_size length (labels are the next tokens for each position)
        x = torch.stack([data[i : i + self.model_config.T] for i in ix])
        y = torch.stack(
            [data[i+1 : i+1+self.model_config.T] for i in ix]
        )
        return x.to(self.device), y.to(self.device)

    def get_lr(self, it: int) -> float:
        """Calculate learning rate using cosine decay with warmup."""

        # Before cosine decay - warmup
        if it < self.train_config.warmup_iters:
            return self.train_config.max_lr * (it + 1) / self.train_config.warmup_iters

        # After cosine decay - return min
        if it > self.train_config.lr_decay_iters:
            return self.train_config.min_lr
        
        # Cosine decay
        decay_ratio = (it - self.train_config.warmup_iters) / (self.train_config.lr_decay_iters - self.train_config.warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.train_config.min_lr + coeff * (self.train_config.max_lr - self.train_config.min_lr)
    
    @torch.no_grad()
    def estimate_loss(
        self, train_data: torch.Tensor, val_data: torch.Tensor
    ) -> Dict[str, float]:
        """Estimate loss by averaging over eval_iters iterations, reducing noise."""
        out = {}
        self.model.eval()

        for split in ["train", "val"]:
            losses = torch.zeros(self.train_config.eval_iters)

            for k in range(self.train_config.eval_iters):
                X, Y = self.get_batch(split, self.train_config.B, train_data, val_data)
                with self.ctx:
                    _, loss = self.model(X, Y)
                losses[k] = loss.item()

            out[split] = losses.mean()

        self.model.train()
        return out

    def train(self, data: torch.Tensor) -> None:
        """Train model using random sampling strategy."""

        # Train/val split
        n = int(self.train_config.train_test_split * len(data))
        train_data, val_data = data[:n], data[n:]

        best_val_loss = float("inf")
        progress_bar = tqdm(range(self.train_config.max_iters), desc="Training")

        for iter_num in progress_bar:

            # Determine and set learning rate
            lr = self.get_lr(iter_num)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            # Evaluate at set intervals
            if iter_num % self.train_config.eval_interval == 0:
                losses = self.estimate_loss(train_data, val_data)
                progress_bar.write(
                    f"Step {iter_num}: Train Loss: {losses['train']:.4f} (PPL: {math.exp(losses['train']):.2f}); "
                    f"Val Loss: {losses['val']:.4f} (PPL: {math.exp(losses['val']):.2f})"
                )

                # Save best model
                if losses["val"] < best_val_loss:
                    self.models_dir.mkdir(parents=True, exist_ok=True)
                    torch.save(self.model.state_dict(), self.models_dir / self.model_file)
                    best_val_loss = losses["val"]

                # Early stopping to prevent overfitting
                if losses["val"] > best_val_loss + self.train_config.loss_tolerance:
                    progress_bar.write("Early stopping triggered. Stopping training.")
                    break

            # Training step, with accumulation over batch splits
            self.optimizer.zero_grad(set_to_none=True)
            split_size = self.train_config.B // self.train_config.B_split
            for _ in range(self.train_config.B_split):
                xb, yb = self.get_batch("train", split_size, train_data, val_data)
                
                with self.ctx:
                    # Do not need logits here
                    _, loss = self.model(xb, yb)
                    loss /= self.train_config.B_split
                loss.backward()

            self.optimizer.step()

            # Update progress bar
            progress_bar.set_postfix({
                "train_loss": f"{loss.item():.4f}", 
                "train_ppl": f"{math.exp(loss.item()):.2f}",
                "lr": f"{lr:.4e}"
            })



def main():

    BASE_DIR = Path(__file__).resolve().parent

    data_path = BASE_DIR.parent / "data"

    austen_path = data_path / "austen.txt"
    dickens_path = data_path / "dickens.txt"
    austen_preprocess_path = data_path / "preprocess-austen.txt"
    dickens_preprocess_path = data_path / "preprocess-dickens.txt"

    if not austen_path.is_file() or not dickens_path.is_file():
        kagglehub.dataset_download("joshmcadams/jane-austin-and-charles-dickens", output_dir=data_path)

    if not austen_preprocess_path.is_file() or not dickens_preprocess_path.is_file():
        utils_path = BASE_DIR.parent / "utils"
        sys.path.append(str(utils_path))
        from preprocess import preprocess
        preprocess(austen_path, austen_preprocess_path)
        preprocess(dickens_path, dickens_preprocess_path)

    # Initialize configs
    train_config = TrainConfig()
    model_config = ModelConfig()

    # Load and process data
    austen_text = austen_preprocess_path.read_text(encoding="utf-8")
    dickens_text = dickens_preprocess_path.read_text(encoding="utf-8")
    text = austen_text + dickens_text

    # Parse tokenization argument
    parser = argparse.ArgumentParser(description="Train JaneLM language model")
    parser.add_argument(
        "--tokenization",
        choices=["tiktoken", "character"],
        default="character",
        help="Tokenization method to use",
    )
    parser.add_argument(
        "--model_file",
        default="model.pth",
        help="Model file name containing trained weights: will be saved to ./models/{model_file}",
    )
    parser.add_argument(
        "--config_file",
        default="model_config.pkl",
        help="File name containing saved model parameters: will be saved to ./models/{config_file}",
    )
    args = parser.parse_args()

    # Tokenization
    if args.tokenization == "tiktoken":
        enc = tiktoken.get_encoding("gpt2")
        encode, decode = enc.encode, enc.decode
        model_config.n_vocab = enc.n_vocab
    else:
        # Create encodings/decodings from characters
        chars = sorted(list(set(text)))
        model_config.n_vocab = len(chars)
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}
        # Encoders/decoders use character lookup tables
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: "".join([itos[c] for c in l])
    model_config.tokenization = args.tokenization
    
    # Save model config
    models_path = BASE_DIR.parent / "models"
    models_path.mkdir(parents=True, exist_ok=True)
    with (models_path / args.config_file).open("wb") as f:
        pickle.dump(model_config, f)

    # Encode data
    data = torch.tensor(encode(text), dtype=torch.long)

    # Initialize model
    model = JaneLM(model_config)
    print(f"Model Parameters: {model.config}")
    print(
        f"Number of Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f} M"
    )
    print(
        f"Number of Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f} M"
    )

    # Initialize trainer and start training
    trainer = Trainer(model, train_config, model_config, models_dir=models_path, model_file=args.model_file)
    trainer.train(data)


if __name__ == "__main__":
    main()

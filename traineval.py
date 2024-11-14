import argparse
import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
from tqdm import tqdm
from typing import Dict, Tuple, Optional
from model import ModelConfig, JaneLM


@dataclass
class TrainConfig:
    max_iters: int = 15_000  # number of training iterations
    eval_interval: int = 250  # how frequently the model is evaluated
    eval_iters: int = 250  # how many runs to average over when evaluating
    learning_rate: float = 1e-4
    loss_tolerance: float = (
        0.075  # if validation loss increases by this amount, stop training
    )
    train_test_split: float = 0.9  # fraction of data to use for training
    batch_size: int = 32  # how many independent sequences to process in parallel


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_config: TrainConfig,
        model_config: ModelConfig,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model.to(device)
        self.train_config = train_config
        self.model_config = model_config
        self.device = device
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=train_config.learning_rate
        )

    def get_batch(
        self, split: str, train_data: torch.Tensor, val_data: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a batch of data for training or validation, each containing context_size-length sequences sampled randomly. """
        data = train_data if split == "train" else val_data

        # generate random starting indices
        ix = torch.randint(
            len(data) - self.model_config.context_size, (self.train_config.batch_size,)
        )
        # get sequences of context_size length
        x = torch.stack([data[i : i + self.model_config.context_size] for i in ix])
        # labels are the next tokens for each position
        y = torch.stack(
            [data[i + 1 : i + 1 + self.model_config.context_size] for i in ix]
        )

        return x.to(self.device), y.to(self.device)

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
                X, Y = self.get_batch(split, train_data, val_data)
                _, loss = self.model(X, Y)
                losses[k] = loss.item()

            out[split] = losses.mean()

        self.model.train()
        return out

    def train(self, data: torch.Tensor) -> None:
        """Train the model using random sampling strategy."""

        # train/val split
        n = int(self.train_config.train_test_split * len(data))
        train_data, val_data = data[:n], data[n:]

        best_val_loss = float("inf")
        progress_bar = tqdm(range(self.train_config.max_iters), desc="Training")

        for iter_num in progress_bar:
            # evaluate at set intervals
            if iter_num % self.train_config.eval_interval == 0:
                losses = self.estimate_loss(train_data, val_data)
                progress_bar.write(
                    f"Step {iter_num}: Train Loss: {losses['train']:.4f}; Val Loss: {losses['val']:.4f}"
                )
                # save best model
                if losses["val"] < best_val_loss:
                    torch.save(self.model.state_dict(), "./models/model.pth")
                    best_val_loss = losses["val"]

                # early stopping to prevent overfitting
                if losses["val"] > best_val_loss + self.train_config.loss_tolerance:
                    progress_bar.write("Early stopping triggered!")
                    break

            # training step
            xb, yb = self.get_batch("train", train_data, val_data)
            self.optimizer.zero_grad(set_to_none=True)
            # do not need logits here
            _, loss = self.model(xb, yb)
            loss.backward()
            self.optimizer.step()

            # update progress bar
            progress_bar.set_postfix({"train_loss": f"{loss.item():.4f}"})


def main():
    # initialize configs
    train_config = TrainConfig(max_iters=1_000)
    model_config = ModelConfig(
        batch_size=1,
        context_size=1,
        n_embed=4,
        head_size=4,
        n_head=1,
        n_block=1,
        dropout=0,
    )

    # load and process data
    with open("./data/janeausten.txt", "r", encoding="utf-8") as f:
        text = f.read()

    # parse tokenization argument
    parser = argparse.ArgumentParser(description="Train JaneLM language model")
    parser.add_argument(
        "--tokenization",
        choices=["tiktoken", "character"],
        default="character",
        help="Tokenization method to use",
    )
    args = parser.parse_args()

    if args.tokenization == "tiktoken":
        # initialize tiktoken
        enc = tiktoken.get_encoding("gpt2")
        encode, decode = enc.encode, enc.decode
        model_config.vocab_size = enc.n_vocab
    else:
        # create encodings/decodings from characters
        chars = sorted(list(set(text)))
        model_config.vocab_size = len(chars)
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}
        # encoders/decoders use character lookup tables
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: "".join([itos[c] for c in l])
    data = torch.tensor(encode(text), dtype=torch.long)

    # initialize model
    model = JaneLM(model_config)
    print(f"Model Parameters: {model.config}")
    print(
        f"Number of Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M"
    )

    # initialize trainer and start training
    trainer = Trainer(model, train_config, model_config)
    trainer.train(data)


if __name__ == "__main__":
    main()

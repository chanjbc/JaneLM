import argparse
import pickle

import tiktoken
import torch
from pathlib import Path
from model import JaneLM



def main():
    """Generates text from a saved model."""

    device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_file", 
        type=str, 
        help="Model file name containing trained weights, assumed to be at ../models/{model_file}", 
        default="model.pth"
    )
    parser.add_argument(
        "--config_file",
        type=str,
        help="File name containing saved/pickled model parameters: assumed to be at ../models/{config_file}",
        default="model_config.pkl"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="Output file name: will be created at ../generated/{output_file}",
        default="output.txt"
    )
    parser.add_argument(
        "--text", 
        type=str, 
        help="Starting text for generation", 
        default=" "
    )
    parser.add_argument(
        "--num_tokens",
        type=int,
        help="Number of tokens to generate",
        default=10_000
    )

    args = parser.parse_args()

    BASE_DIR = Path(__file__).resolve().parent
    models_path = BASE_DIR.parent / "models"
    data_path = BASE_DIR.parent / "data"
    generated_path = BASE_DIR.parent / "generated"

    model_config_path = models_path / args.config_file
    with model_config_path.open("rb") as f:
        model_config = pickle.load(f)
    tokenization = model_config.tokenization

    if tokenization == "tiktoken":
        # Initialize tiktoken
        enc = tiktoken.get_encoding("gpt2")
        encode, decode = enc.encode, enc.decode
        model_config.n_vocab = enc.n_vocab
    elif tokenization == "custom-bpe":
        import sys
        utils_path = BASE_DIR.parent / "utils"
        if str(utils_path) not in sys.path:
            sys.path.append(str(utils_path))
        from bpe import BPE
        tokenizer = BPE()
        tokenizer.load(models_path / f"{args.config_file.split('.')[0]}_bpe.pkl")
        encode, decode = tokenizer.encode, tokenizer.decode
    else:  # Character
        # Load and process data
        austen_text = (data_path / "preprocess-austen.txt").read_text(encoding="utf-8")
        dickens_text = (data_path / "preprocess-dickens.txt").read_text(encoding="utf-8")
        text = austen_text + dickens_text
        # Create encodings/decodings from characters
        chars = sorted(list(set(text)))
        model_config.n_vocab = len(chars)
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}
        # Encoders/decoders use character lookup tables
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: "".join([itos[c] for c in l])

    starting_text = torch.tensor(encode(args.text), dtype=torch.long, device=device).unsqueeze(0)

    model_state = torch.load(models_path / args.model_file, map_location=device)
    model = JaneLM(model_config).to(device)
    model.load_state_dict(model_state)

    # TODO: add error handling
    output_path = generated_path / args.output_file
    generated_path.mkdir(parents=True, exist_ok=True)
    output_path.write_text(decode(model.generate(starting_text, max_new_tokens=args.num_tokens)[0].tolist()))
    print(f"Text successfully generated to {output_path}")



if __name__ == "__main__":
    main()
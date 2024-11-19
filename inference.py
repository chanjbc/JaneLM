import argparse
import pickle
import tiktoken
import torch
from model import JaneLM



def main():
    """Generates text from a saved model."""

    device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_file_name", 
        type=str, 
        help="Model file name containing trained weights: will be saved to ./models/{model_file_name}", 
        default="model.pth"
    )
    parser.add_argument(
        "--config_file_name",
        type=str,
        help="File name containing saved/pickled model parameters: assumed to be in ./models/{config_file_name}",
        default="model-config.pkl"
    )
    parser.add_argument(
        "--output_file_name",
        type=str,
        help="Output file name: will be created at ./generated/{output_file_name}",
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

    model_config = pickle.load(open(f"./models/{args.config_file_name}", "rb"))
    tokenization = model_config.tokenization

    if tokenization == "tiktoken":
        # initialize tiktoken
        enc = tiktoken.get_encoding("gpt2")
        encode, decode = enc.encode, enc.decode
        model_config.vocab_size = enc.n_vocab
    else:
        # load and process data
        with open("./data/janeausten.txt", "r", encoding="utf-8") as f:
            text = f.read()
        # create encodings/decodings from characters
        chars = sorted(list(set(text)))
        model_config.vocab_size = len(chars)
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}
        # encoders/decoders use character lookup tables
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: "".join([itos[c] for c in l])

    starting_text = torch.tensor(encode(args.text), dtype=torch.long, device=device).unsqueeze(0)

    model_state = torch.load(f"./models/{args.model_file_name}", map_location=device)
    model = JaneLM(model_config).to(device)
    model.load_state_dict(model_state)

    # TODO: add error handling
    open(f'./generated/{args.output_file_name}', 'w').write(decode(model.generate(starting_text, max_new_tokens=args.num_tokens)[0].tolist()))
    print(f"Text successfully generated to ./generated/{args.output_file_name}")

if __name__ == "__main__":
    main()
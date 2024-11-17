import argparse
import tiktoken
import torch

from model import JaneLM, ModelConfig

def main():
    # TODO: add arguments for these parameters
    model_config = ModelConfig(
        batch_size=1,
        context_size=1,
        n_embed=4,
        head_size=4,
        n_head=1,
        n_block=1,
        dropout=0,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--modelFileName", 
        type=str, 
        help="Model file name (place model in ./models)", 
        default="model.pth"
    )
    parser.add_argument(
        "--text", 
        type=str, 
        help="Starting text for generation", 
        default=" "
    )
    parser.add_argument(
        "--tokenization",
        choices=["tiktoken", "character"],
        default="character",
        help="Tokenization method to use",
    )
    parser.add_argument(
        "--outputFileName",
        type=str,
        help="Output .txt file name: will be created at ./generated",
        default="output"
    )
    parser.add_argument(
        "--num_tokens",
        type=int,
        help="Number of tokens to generate",
        default=1_000
    )
    args = parser.parse_args()


    # TODO: remove this
    # load and process data
    with open("./data/janeausten.txt", "r", encoding="utf-8") as f:
        text = f.read()

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

    starting_text = torch.tensor(encode(args.text), dtype=torch.long, device=device).unsqueeze(0)

    model_state = torch.load(f"./models/{args.modelFileName}", map_location=device)
    model = JaneLM(model_config).to(device)
    model.load_state_dict(model_state)

    # TODO: add error handling
    open(f'./models/{args.outputFileName}.txt', 'w').write(decode(model.generate(starting_text, max_new_tokens=args.num_tokens)[0].tolist()))
    print(f"Text successfully generated to ./models/{args.outputFileName}.txt")

if __name__ == "__main__":
    main()
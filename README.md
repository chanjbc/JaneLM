# JaneLM

This repository implements a language model trained on the works of Jane Austen. More specifically, the model is a decoder-only transformer model using multi-headed causal self-attention (as detailed in [Vaswani et al., 2017](https://doi.org/10.48550/arXiv.1706.03762)), and is implemented in PyTorch loosely "from scratch" (i.e., this repository does not use any of PyTorch's built-in transformer classes, but may use other classes such as `torch.nn.Linear`). The repository contains both code for training/evaluating/performing inference on smaller models, and a ~70 million parameter pretrained sample model.  

Here is a sample of some generated text from the sample 70 M model, taken from `./generated/sample.txt`:

```
Elizabeth's presence was as beautiful and entertainment of the meeting her linen was now such anmorrow, and in her turkeys and shades she had undone the music especially after the time, who was counting a horse in that pianof card-table down alone. Elizabeth lifted upcast eyes in half the midst of the crowne, petition suspicion on Mrs. Hurst's thoughts into which Bingley made her so high in the apartment, and they were going while; and Elizabeth remained in comfort, to part with so express out of her mother's heart. 
"It is very good Mr. Bingley," added Georgiana. "We never thought of him. That I am very glad they contrives we had been more birds are old in October." 
```

As you can see, the text is only half coherent--this is because modern language models are larger by orders of magnitude (e.g., GPT-2 has >1 billion parameters, and GPT-3 has >100 billion parameters), and therefore have better representational power and expressiveness.

## Authors
- [James Chan](https://github.com/chanjbc)

## Requirements
- Python 3.7+
- PyTorch
- tiktoken
- argparse
- tqdm

## Model Details
The model is a decoder-only transformer model using causal self-attention with prenorms (as opposed to the postnorms used in the original implementation). The model was trained using the cross-entropy loss with the AdamW optimizer, and trains on the dataset by first selecting random mini-batches of tokens and optimizing autoregressively. While training, the training/validation losses are averaged over multiple passes, reducing the noise of the error signals, and early stopping is additionally used to stop training before overfitting begins.

## Usage
You can either use this repository to train your own model or run the sample 70 M model (see ["Running the 70 M Model"](#running-the-70-m-model)).

### Training
To train a new model, first set the parameters of `ModelConfig`, located in `./model.py`, and `TrainConfig`, located in `traineval.py`, according to your computational resources. By default, these are set to the same values used in training the 70 M parameter model.

Then, run `traineval.py`, which uses the following command line arguments:
- `--tokenization`: set this to either "character" or "tiktoken" (default: "character"); controls the tokenization method used to encode/decode the dataset
- `--model_file`: set this to a .pth file (default: "model.pth"); sets the output file of the trained model, which will be placed in ./models
- `--config_file`: set this to a .pkl file (default: "model-config.pkl"); sets the config file of the final, trained model (this saves the model architecture), which will be placed in ./models

Here is an example:
```bash
py traineval.py --tokenization tiktoken --model_file my_model.pth --config_file my_model_config.pkl
```

### Running Inference

To generate text using a trained model, run `inference.py`, which uses the following command line arguments:
- `--model_file`: same as with `traineval.py`
- `--config_file`: same as with `traineval.py`
- `--output_file`: set this to a .txt file (default: "output.txt"); sets the output file of the generated text, which will be created at ./generated
- `--text`: set this to a string (default: " "); sets the starting text from which to generate
- `--num_tokens`: set this to a whole number (default: 10,000); sets the number of tokens to generate

Here is an example:
```bash
py inference.py --model_file model.py --config_file model_config.py --output_file output.txt --num_tokens 500
```

### Running the 70 M Model
Due to file size restrictions, the 70 M model weights could not be uploaded to GitHub, but are instead openly available [here](https://drive.google.com/file/d/1blo6THJ7BCKh_WTJQHEpAq33lGksipVv/view?usp=sharing) (file size: 276 MB). Once you download the file, place it in `./models` and run

```bash
py inference.py --model_file model.py --config_file model_config.py --output_file output.txt
```

Here's a screenshot of the training/validation loss plotted over time (note that early stopping was used to avoid overfitting):

<img src="https://github.com/chanjbc/JaneLM/blob/main/assets/loss.png" width="600" align="middle">

## References
- ["Attention is All You Need"](https://arxiv.org/abs/1706.03762)
- [Andrej Karpathy's GPT Series](https://www.youtube.com/watch?v=kCc8FmEb1nY&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=7)
- [Deep Learning: Foundation and Concepts](https://www.bishopbook.com/)

### TODO
- ~~Add `generate.py` and `model.pth`~~
- ~~Separate main file into `model.py`, `train.py`, ...~~
- ~~Add tqdm progress bars for train/eval~~
- ~~Add argparse support~~
- ~~Add dataclass support~~
- Continue to improve model
- Fine-tuning???
- Add more to this README :)

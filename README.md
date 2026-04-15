# JaneLM

This repository implements a foundation language model trained on the works of Jane Austen and Charles Dickens, implemented in PyTorch loosely "from scratch" (i.e., this repository does not directly use any of PyTorch's built-in transformer classes, but does use other modules like `torch.nn.Linear`).

The repository contains code for training from scratch, as well as a 70 million parameter pretrained model for testing. Here is a sample of some generated text from the sample 70 M model, taken from `./generated/sample.txt`:

```
Elizabeth's presence was as beautiful and entertainment of the meeting her linen was now such anmorrow, and in her turkeys and shades she had undone the music especially after the time, who was counting a horse in that pianof card-table down alone. Elizabeth lifted upcast eyes in half the midst of the crowne, petition suspicion on Mrs. Hurst's thoughts into which Bingley made her so high in the apartment, and they were going while; and Elizabeth remained in comfort, to part with so express out of her mother's heart. 
"It is very good Mr. Bingley," added Georgiana. "We never thought of him. That I am very glad they contrives we had been more birds are old in October." 
```

Note that the generated text is only half coherent; this is because modern language models are larger by orders of magnitude (e.g., GPT-2 has >1 billion parameters, and GPT-3 has >100 billion parameters), and therefore have better representational power and expressiveness.



## Requirements
- `Python 3.10+`
- `PyTorch`
- `tiktoken`
- `argparse`
- `tqdm`
- `kagglehub`



## Architecture Details

The dataset consists of the complete works of Jane Austen and Charles Dickens, available at Kaggle [here](https://www.kaggle.com/datasets/joshmcadams/jane-austin-and-charles-dickens). Note that running the training script will download the dataset automatically.

The model is a decoder-only transformer foundation model using grouped-query attention (GQA) ([Ainslie et al., 2023](https://arxiv.org/abs/2305.13245)) with prenorms. For embeddings, a custom BPE implementation is implemented, and RoPE ([Su et al., 2021](https://arxiv.org/abs/2104.09864)) is further applied before attention blocks. 

The cross-entropy loss with AdamW is used, and the model trains on the dataset by first selecting random mini-batches of tokens of length `context_size` and optimizing autoregressively. 

The learning rate is updated using cosine learning rate scheduling with linear warmup. To improve training efficiency and memory usage, the model utilizes BF16 (`bfloat16`) training. While training, the training/validation losses are averaged over multiple passes, reducing the noise of the error signals, and early stopping is additionally used to stop training before overfitting begins. For tokenization, this repo accepts character-level tokenization, `tiktoken` GPT-2 tokenization, or a custom-trained Byte-Pair Encoding (BPE) tokenizer. Note that BPE is preferred since it is more efficient than character-level and results in far fewer unnecessary parameters than `tiktoken`.




## Usage
This repository can be used to train from scratch, resume training, or run inference using a saved model (note also that a pretrained model is provided - see ["Running the 70 M Model"](#running-the-70-m-model)).

### Training
To train a new model, first set the parameters of `ModelConfig`, located in `./model.py`, and `TrainConfig`, located in `traineval.py`, according to your computational resources. By default, these are set to the same values used in training the 70 M parameter model.

Then, run `traineval.py`, which uses the following command line arguments:
- `--tokenization`: set this to either "character", "tiktoken", or "custom-bpe" (default: "custom-bpe"); controls the tokenization method used to encode/decode the dataset
- `--resume`: optional flag to resume training from the model and BPE tokenizer specified in `--model_file` and `--config_file`
- `--model_file`: set this to a .pth file (default: "model.pth"); sets the output file of the trained model, which will be placed in `./models`
- `--config_file`: set this to a .pkl file (default: "model-config.pkl"); sets the config file of the final, trained model (this saves the model architecture), which will be placed in `./models`

Here is an example:
```bash
py traineval.py \
  --tokenization=tiktoken \
  --model_file=my_model.pth \
  --config_file=my_config.pkl
```



### Running Inference

To generate text using a trained model, run `inference.py`, which uses the following command line arguments:
- `--model_file`: same as with `traineval.py`
- `--config_file`: same as with `traineval.py`
- `--output_file`: set this to a .txt file (default: `output.txt`); sets the output file of the generated text, which will be created at `./generated`
- `--text`: set this to a nonempty string (default: " "); sets the starting text from which to generate
- `--num_tokens`: set this to a whole number (default: 10,000); sets the number of tokens to generate

Here is an example:
```bash
py inference.py \
  --model_file=my_model.pth \
  --config_file=my_config.pkl \
  --output_file=output.txt \
  --text="Elizabeth screamed!" \
  --num_tokens=500
```

Note that inference may fail if the next token selected cannot be properly converted to UTF-8 (a fix for this is still being developed).

### Running the 70 M Model
Due to file size restrictions, the 70 M model weights could not be uploaded to GitHub, but are instead openly available [here](https://drive.google.com/file/d/1blo6THJ7BCKh_WTJQHEpAq33lGksipVv/view?usp=sharing) (file size: 276 MB). Once you download the file, place it in `./models` and run, for example, this:

```bash
py inference.py \
  --model_file=sample_model.pth \
  --config_file=sample_config.py \
  --output_file=output.txt
```

Here's a screenshot of the training/validation loss plotted over time (note that early stopping was used to avoid overfitting):

<img src="https://github.com/chanjbc/JaneLM/blob/main/assets/loss.png" width="600" align="middle">

### TODO

- ~~Add `generate.py` and `model.pth`~~
- ~~Separate main file into `model.py`, `train.py`, ...~~
- ~~Add tqdm progress bars for train/eval~~
- ~~Add argparse support~~
- ~~Add dataclass support~~
- ~~Add resume option to traineval.py~~
- ~~Add BF16 mixed precision training~~
- ~~Add cosine learning rate scheduling with linear warmup~~
- ~~Add custom BPE tokenizer~~
- ~~Fix inference errors due to UTF-8 and weird tokenization stuff~~
- ~~Add RoPE~~
- ~~Switch from MHA to GQA~~
- Rewrite GQA to avoid realizing the full QK^T matrix
- Add `torch.compile`
- Add kv-cache for faster inference
- Add Triton FlashAttention

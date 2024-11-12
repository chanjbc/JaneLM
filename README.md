# JaneLM

This repository implements a language model trained on the works of Jane Austen. The model is a decoder-only transformer model using multi-headed causal self-attention (as detailed in [Vaswani et al., 2017](https://doi.org/10.48550/arXiv.1706.03762)), and is implemented in PyTorch loosely "from scratch" (i.e., this repository does not use any of PyTorch's built-in transformer classes, but may use other classes such as `torch.nn.Linear`) and contains ~70 million parameters.  

## Authors
- [James Chan](https://github.com/chanjbc)

## Model Details

As mentioned in the description, this model is a decoder-only transformer model using multi-headed causal self-attention trained in an autoregressive manner. The exact parameters used are as follows (although these can be adjusted based on your computational resources):
- batch_size = 32
- context_size = 96
- n_embed = 512
- head_size = 128
- n_head = 4
- n_block = 6
- 
In total, the trained model contains over 70 million parameters. 

Here's a screenshot of the training/validation loss plotted over time (note that early stopping was used to avoid overfitting):

<img src="https://github.com/chanjbc/JaneLM/blob/main/assets/loss.png" width="600" align="middle">

Here is a sample of some generated text from the finalized model, taken from `sample-text.txt`:

```
Elizabeth's presence was as beautiful and entertainment of the meeting her linen was now such anmorrow, and in her turkeys and shades she had undone the music especially after the time, who was counting a horse in that pianof card-table down alone. Elizabeth lifted upcast eyes in half the midst of the crowne, petition suspicion on Mrs. Hurst's thoughts into which Bingley made her so high in the apartment, and they were going while; and Elizabeth remained in comfort, to part with so express out of her mother's heart. 
"It is very good Mr. Bingley," added Georgiana. "We never thought of him. That I am very glad they contrives we had been more birds are old in October." 
```

As you can see, the text is only half coherent--this is because modern language models are larger by orders of magnitude (e.g., GPT-2 has >1 billion parameters, and GPT-3 has >100 billion parameters), and therefore have better representational power and expressiveness.

## Requirements
- Python 3.7+
- PyTorch
- tiktoken (technically optional: if not supplied, the model will operate on the character level rather than on the token level) 

### TODO
- Add `generate.py` and `model.pth`
- Separate main file into `model.py`, `train.py`, ...
- Add tqdm progress bars for train/eval
- Add argparse support
- Add dataclass support
- Continue to improve model
- Attempt to replicate assistant behavior without fine-tuning (probably poor performance)
- Fine-tune model???
- Add more to this README :)

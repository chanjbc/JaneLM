# JaneLM

This repository implements a language model trained on the works of Jane Austen. The model is a decoder-only transformer model using multi-headed causal self-attention (as detailed in [Vaswani et al., 2017](https://doi.org/10.48550/arXiv.1706.03762)), and is implemented in PyTorch loosely "from scratch" (i.e., this repository does not use any of PyTorch's built-in transformer classes, but may use other classes such as `torch.nn.Linear`) and contains 70 million parameters.  

## Authors
- [James Chan](https://github.com/chanjbc)

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

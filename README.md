# mike-makemore

My notes and code from working through Andrej Karpathy's [makemore](https://github.com/karpathy/makemore) series ([videos](https://www.youtube.com/watch?v=PaCmpygFfXo&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=2)) - character-level language models that learn to generate new names by predicting the next character. It also includes a deeper study of PyTorch fundamentals (shapes, broadcasting, matmul, autograd).  The pytorch study culminated with training an MNIST MLP - which explores data set splitting fundamentals, and also how to spot **overfitting**.  

## Notebooks

| Notebook | Topic |
|---|---|
| [makemore-video-1.ipynb](makemore-video-1.ipynb) | Bigram model - counting, then a single-layer NN (one-hot, softmax, negative log likelihood) |
| [makemore-video-2.ipynb](makemore-video-2.ipynb) | MLP model following Bengio et al. (2003) - learned embeddings, `tanh` hidden layer |
| [makemore-video-3.ipynb](makemore-video-3.ipynb) | Part 3 (in progress) |
| [pytorch-video-series1.ipynb](pytorch-video-series1.ipynb) | PyTorch fundamentals - tensor rank/dims, broadcasting, reductions, matmul, autograd, loss functions; trains a FashionMNIST MLP ; data set splitting and epochs|
| [pytorch-worksheet-shapes-broadcasting-matmul.ipynb](pytorch-worksheet-shapes-broadcasting-matmul.ipynb) | Practice problems - predict shapes for dims, broadcasting, and `@` |

## Data

- `names.txt` - ~32k names used as the training set
- `fashion_mlp.pt` - saved weights from the FashionMNIST MLP

## Setup

Requires Python 3.14 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run jupyter lab
```

import torch
import torch.nn as nn
import numpy as np


class Embedding(nn.Module):
    def __init__(self, v_dim, e_dim):
        super(Embedding, self).__init__()
        self.vocab_layer = nn.Linear(v_dim, e_dim)
        self.embed_layer = nn.Linear(e_dim, v_dim)

    def forward(self, word):
        x = torch.sigmoid(self.vocab_layer(word))
        x = torch.softmax(self.embed_layer(x), dim=0)
        return x


if __name__ == '__main__':
    v_dim = 2**14
    e_dim = 2**10

    mdl = Embedding(v_dim, e_dim)
    vec = np.zeros(v_dim)
    hot_word_ix = np.random.randint(v_dim)
    vec[hot_word_ix] = 1.
    vec = torch.Tensor(vec)

    with torch.no_grad():
        out = mdl(vec)
    print(out)

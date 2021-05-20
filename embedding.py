import jax.ops as jops
import jax.numpy as jnp
import jax
from jax import random

import argparse
from flax import linen as nn
import flax.serialization as ser

import pickle
from tqdm import tqdm
from tqdm.contrib import tenumerate
from tqdm import trange


"""
    Simple word2vec embedding --- all positive examples
"""
class Embedding(nn.Module):
    """
        Embedding model
        call with (pkl_name=<str>) or (in_dim=<int>, embed_dim=<int>)

        pkl_name : pickled model filename
        in_dim, embed_dim : dimensions of the input and embedding
    """
    vocab_dim: int
    embed_dim: int

    def setup(self):
        self.vocab_layer = nn.Dense(self.embed_dim)
        self.embed_layer = nn.Dense(self.vocab_dim)

    def __call__(self, word):
        word_features = self.vocab_layer(word)
        word_features_act = nn.sigmoid(word_features)
        embed_features = self.embed_layer(word_features_act)
        embed_act = nn.softmax(embed_features)
        return embed_act

    def save_wordvecs(self, fname):
        # generate a word -> vec dict
        state = ser.to_state_dict(self)
        with open(fname, 'wb') as f:
            pickle.dump(state, f)


def nll_loss_fn(model, params, x_batch, y_batch):
    norm = 1./len(x_batch)
    def calc_nll(params):
        def nll(x, y):
            pred = model.apply(params, x)
            return jnp.sum(y * jnp.log(pred), axis=0)
        return - norm * jnp.sum(jax.vmap(nll)(x_batch, y_batch))
    return jax.jit(calc_nll)


def _nll_loss_fn(model, params, x_batch, y_batch):
    def get_total_loss(params):
        total_loss = 0.
        for sample_ix in range(len(x_batch)):
            x = x_batch[sample_ix]
            y = y_batch[sample_ix]
            pred = model.apply(params, x)
            loss = y * jnp.log(pred)
            sample_loss = jnp.sum(loss)
            total_loss += sample_loss

        norm = 1./len(x_batch)
        return - norm * total_loss
    return get_total_loss


def id_to_one_hot(data, one_hot_dim):
    num_samples = len(data)
    for ix, sample in enumerate(data):
        x_one_ixs.append((ix, sample[0]))
        y_one_ixs.append((ix, sample[1]))
    one_hots = jnp.zeros((num_samples, one_hot_dim), dtype=jnp.float32)
    x_one_hots = jops.index_update(one_hots, x_one_ixs, 1.,
                                   indices_are_sorted=True, unique_indices=True)
    y_one_hots = jops.index_update(one_hots, y_one_ixs, 1.,
                                   indices_are_sorted=True, unique_indices=True)
    return x_one_hots, y_one_hots


def _id_to_one_hot(data, one_hot_dim):
    x = []
    y = []
    for xy in data:
        _x = jnp.zeros(one_hot_dim, dtype=jnp.float32)
        _x = jops.index_update(_x, xy[0], 1.0)
        x.append(_x)
        _y = jnp.zeros(one_hot_dim, dtype=jnp.float32)
        _y = jops.index_update(_y, xy[1], 1.0)
        y.append(_y)
    x = jnp.array(x)
    y = jnp.array(y)
    return x, y


def __id_to_one_hot(data, one_hot_dim):
    x = []
    y = []
    vec = jnp.zeros(one_hot_dim, dtype=jnp.float32)
    for xy in data:
        _x = jops.index_update(vec, xy[0], 1.0)
        x.append(_x)
        _y = jops.index_update(vec, xy[1], 1.0)
        y.append(_y)
    x = jnp.array(x)
    y = jnp.array(y)
    return x, y


def main():
    with open('reuters_vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)

    v_dim = len(vocab['num_to_word'])
    e_dim = 1024

    prng_key = random.PRNGKey(0xdeadbeef)
    words = jnp.zeros(v_dim, dtype=jnp.float32)
    word_ix = random.uniform(prng_key, (1,)) * v_dim
    word_ix = int(jnp.floor(word_ix)[0])
    words = jops.index_update(words, word_ix, 1.0)

    # first create the architecture of the model
    mdl = Embedding(v_dim, e_dim)
    # then complete the model spec by giving an example input tensor
    params = mdl.init(prng_key, words)
    # now apply the params to the model with the input
    out = mdl.apply(params, words)
    print(f'out: {out}')
    print(f'shape: {out.shape}')

    # let's train the model on nltk's reuters dataset
    from nltk.corpus import reuters
    train_texts = []
    for fname in reuters.fileids():
        text = reuters.words(fname)
        train_texts.append(text)

    # now generate word-context elements
    window_size = 2
    word_pairs = []
    for words in tqdm(train_texts, desc='make train set'):
        for word_ix, word in enumerate(words):
            for offset in range(1, window_size+1):
                back_context = word_ix - offset
                if back_context >= 0:
                    word_pairs.append((word, words[back_context]))
                fwd_context = word_ix + offset
                if fwd_context < len(words):
                    word_pairs.append((word, words[fwd_context]))

    # convert words to vocab IDs
    w2n = vocab['word_to_num']

    id_pairs = []
    for word_pair in tqdm(word_pairs, desc='gen word pairs'):
        word = word_pair[0]
        context = word_pair[1]
        if word in w2n and context in w2n:
            w_id, c_id = w2n[word], w2n[context]
            id_pairs.append((w_id, c_id))
    id_pairs = jnp.array(id_pairs)
    print(f'train pairs: {len(id_pairs)}')

    # run grad desc
    id_pairs = id_pairs[0:len(id_pairs)//100]
    lr = 0.3
    batch_size = 2500

    # TEST: what if I run one at a time?
    '''
    loss_fn = lambda x, y : nll_loss_fn(mdl, params, x, y)
    grad_fn = jax.value_and_grad(loss_fn)
    grad_calc_fn = lambda params, x, y : grad_fn(params, x, y)
    param_update_fn = lambda old, grad: old -  lr * grad

    template_vec = jnp.zeros(v_dim, dtype=jnp.float32)
    for epoch in trange(5):
        for pair in tqdm(id_pairs):
            x = jops.index_update(template_vec, pair[0], 1.)
            y = jops.index_update(template_vec, pair[1], 1.)
            loss_val, grad = grad_calc_fn(params, x, y)
            params = jax.tree_multimap(param_update_fn, paramd, grad)
    import pdb; pdb.set_trace()
    pass
    '''
    # TEST END

    batches = jnp.split(id_pairs, jnp.arange(batch_size, len(id_pairs), batch_size))

    for epoch in trange(1):
        # TODO: shuffle & batch id_pairs
        pbar = trange(len(batches), desc=f'epoch:--- - loss:------')
        for batch in batches:
            x_vals, y_vals =__id_to_one_hot(batch, v_dim)
            loss_fn = nll_loss_fn(mdl, params, x_vals, y_vals)
            grad_fn = jax.value_and_grad(loss_fn)
            loss_val, grad = grad_fn(params)
            params = jax.tree_multimap(lambda old, grad: old - lr * grad, params, grad)
            pbar.set_description(f'epoch:{epoch:03d} - loss:{loss_val:0.4f}')
            pbar.update()

    import pdb; pdb.set_trace()
    print('done!')


if __name__ == '__main__':
    main()

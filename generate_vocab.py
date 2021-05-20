import defs
import nltk
import pickle
from tqdm.contrib import tenumerate

from nltk.corpus import reuters


def get_wordlist(corpus):
    word_list = []
    word_count = 0
    vocab = set()

    if corpus == 'reuters':
        corp = reuters
        file_ids = corp.fileids()
        for file_ix, f in tenumerate(file_ids, desc='articles'):
            if f.startswith('train'):
                #for word_ix, word in tenumerate(reuters.words(f), desc='words'):
                word_list = reuters.words(f)
                word_count += len(word_list)
                for word_ix, word in enumerate(word_list):
                    vocab.add(word)
        return vocab, word_count
    else:
        return None, None


def get_vocab_maps(word_set):
    # map word->num and num->word
    num_to_word = defs.DEFAULT_TOKEN_ORDER + list(word_set)
    word_to_num = {}
    for ix, word in enumerate(num_to_word):
        word_to_num[word] = ix

    vocab_dict = {
        'num_to_word' : num_to_word,
        'word_to_num' : word_to_num}

    return vocab_dict


def main():
    vocab_fname = './reuters_vocab.pkl'

    vocab = set()
    word_count = 0

    vocab, word_count = get_wordlist('reuters')
    vocab_dict = get_vocab_maps(vocab)

    with open(vocab_fname, 'wb+') as f:
        pickle.dump(vocab_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    # indicate completion
    print('scanned {} total words --- {} distinct words'.format(len(vocab), word_count))
    print('vocab saved as {}'.format(vocab_fname))
    print('done!')


if __name__ == '__main__':
    main()

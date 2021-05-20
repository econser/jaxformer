import defs
import jax
import pickle
import argparse

#CLASSIFIER_TOKEN = '[cls]'
#SEPERATION_TOKEN = '[sep]'
#PADDING_TOKEN = '[pad]'
#UNKNOWN_TOKEN = '[unk]'
#
#WHSP_AND_PUNCTS = "[\w']+|[.!?]"
#STOPS = '.!?'

"""
    Naive tokenization with no stemming
"""
def naive_tkz(sequence, pad_to=None):
    # split the sequence on the spaces
    import re
    tokens = re.findall(defs.WHSP_AND_PUNCTS, sequence)

    # convert each word to an int
    return tokens


"""
    pad a sequence to a desired length
"""
def pad(segment, length):
    # is it too long?
    if len(segment) >= length:
        segment[length-1] = defs.SEPERATION_TOKEN
        segment = segment[:length]
    # is it too short?
    elif len(segment) < length:
        for i in range(length - len(segment)):
            segment.append(defs.PADDING_TOKEN)
    # is it the exact length?
    else:
        segment[length-1] = defs.SEPERATION_TOKEN

    return segment


"""
    convert a sequence of tokens to integers
"""
def naive_cls_encode(tokens, vocab, pad_to=None):
    # insert sep tokens after stops
    updated_tokens = []
    import pdb;pdb.set_trace()
    for token in tokens:
        updated_tokens.append(token)
        if token in defs.STOPS:
            updated_tokens.append(defs.SEPERATION_TOKEN)
    tokens = updated_tokens

    # generate a list of lists, splitting on the [sep]
    segments = []
    current_segment = [defs.CLASSIFIER_TOKEN]

    for t in tokens:
        current_segment.append(t)
        if t == defs.SEPERATION_TOKEN:
            if pad_to is not None:
                current_segment = pad(current_segment, pad_to)
            segments.append(current_segment)
            current_segment = [defs.CLASSIFIER_TOKEN]

    if len(current_segment) > 0:
        if current_segment[-1] != defs.SEPERATION_TOKEN:
            current_segment.append(defs.SEPERATION_TOKEN)
        segments.append(current_segment)

    # TODO: convert each token in the sequences into an int
    word_to_num = vocab['word_to_num']
    unknown_num = word_to_num[defs.UNKNOWN_TOKEN]
    encodings = [word_to_num.get(t, unknown_num) for t in tokens]
    import pdb;pdb.set_trace()
    return encodings


"""
    naive encoding for the Next Sentence Prediction task
"""
def naive_nsp_encode(tokens):
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='the string to parse')
    parser.add_argument('--vocab', help='vocab file for encoding',  default='./reuters_vocab.pkl', )
    args = parser.parse_args()

    with open(args.vocab, 'rb') as f:
        vocab = pickle.load(f)

    tokens = naive_tkz(args.input)
    encodings = naive_cls_encode(tokens, vocab, pad_to=10)

    print(encodings)


if __name__ == '__main__':
    main()

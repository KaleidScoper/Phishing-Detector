import torch
from vocab import Vocab,save_vocab


# parameter
data0_path="./data/legtimate-58w.txt"
data1_path="./data/phish-58w.txt"
train_prop=0.8
val_prop=0.1
test_prop=0.1


def get_char_ngrams(word):
    chars = list(word)
    begin_idx = 0
    ngrams = []
    while (begin_idx + 1) <= len(chars):
        end_idx = begin_idx + 1
        ngrams.append("".join(chars[begin_idx:end_idx]))
        begin_idx += 1
    return ngrams

def load_data(filepath):
    data = []
    data_char = []
    with open(filepath, encoding="utf-8") as f:
        for line in f.readlines():
            data.append(line.strip('\n'))

    for line in data:
        data_char.append(get_char_ngrams(line))

    f.close()
    return data_char


def url_jieduan(url,len):
    jieduan=[]
    for line in url:
        jieduan.append(line[:len])
    return jieduan

def load_sentence_polarity(length):
    data0 = load_data(data0_path)
    data0 = url_cut(data0,length)
    data0_test = data0[:(int)(test_prop * len(data0))]
    data0_val = data0[(int)(test_prop * len(data0)):(int)((test_prop+val_prop) * len(data0))]
    data0_train = data0[(int)((test_prop+val_prop) * len(data0)):]

    data1 = load_data(data1_path)
    data1 = url_cut(data1,length)
    data1_test = data1[:(int)(test_prop * len(data1))]
    data1_val = data1[(int)(test_prop * len(data1)):(int)((test_prop+val_prop) * len(data1))]
    data1_train = data1[(int)((test_prop+val_prop) * len(data1)):]

    vocab = Vocab.build(data0+data1)
    save_vocab(vocab,'./vocab.txt')

    train_data = [(vocab.convert_tokens_to_ids(sentence), 0)
                  for sentence in data0_train] \
        + [(vocab.convert_tokens_to_ids(sentence), 1)
            for sentence in data1_train]

    val_data = [(vocab.convert_tokens_to_ids(sentence), 0)
                 for sentence in data0_val] \
        + [(vocab.convert_tokens_to_ids(sentence), 1)
            for sentence in data1_val]

    test_data = [(vocab.convert_tokens_to_ids(sentence), 0)
                 for sentence in data0_test] \
        + [(vocab.convert_tokens_to_ids(sentence), 1)
            for sentence in data1_test]

    return train_data,  val_data, test_data, vocab


def length_to_mask(lengths):
    max_len = torch.max(lengths)
    mask = torch.arange(max_len).expand(lengths.shape[0], max_len) < lengths.unsqueeze(1)
    return mask

def load_treebank():
    from nltk.corpus import treebank
    sents, postags = zip(*(zip(*sent) for sent in treebank.tagged_sents()))

    vocab = Vocab.build(sents, reserved_tokens=["<pad>"])

    tag_vocab = Vocab.build(postags)

    train_data = [(vocab.convert_tokens_to_ids(sentence), tag_vocab.convert_tokens_to_ids(tags)) for sentence, tags in zip(sents[:3000], postags[:3000])]
    test_data = [(vocab.convert_tokens_to_ids(sentence), tag_vocab.convert_tokens_to_ids(tags)) for sentence, tags in zip(sents[3000:], postags[3000:])]

    return train_data, test_data, vocab, tag_vocab
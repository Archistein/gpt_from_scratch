from itertools import chain
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer, Tokenizer, models, pre_tokenizers, trainers, processors, decoders

# TODO: arrange this file as a pipeline

access_token = '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' # TODO: Change to PATH_VAR

# Collect the data

wiki = load_dataset("pszemraj/simple_wikipedia")
books = load_dataset("suolyer/pile_books3")
peS2o = load_dataset('nampdn-ai/mini-peS2o', token=access_token)

# Tokenization

def train_iterator(batch_size: int = 10, *datasets):
    for ds in chain(*map(list, datasets)):
        for j in range(0, len(ds), batch_size):
            yield ds[j:batch_size+j]['text']

tokenizer = ByteLevelBPETokenizer()
tokenizer.train_from_iterator(train_iterator(5, books.values(), peS2o.values(), wiki.values()), vocab_size=8192, min_frequency=2, special_tokens=['[EOS]'], show_progress=True)
tokenizer.save('tokenizer.json')
tokenizer = Tokenizer.from_file('tokenizer.json')

def pipeline():
    pass

###
# if __name__ == '__main__':
# ...
###
from typing import Generator
from itertools import chain
from datasets import load_dataset, concatenate_datasets, Dataset
from tokenizers import ByteLevelBPETokenizer, Tokenizer, models, pre_tokenizers, trainers, processors, decoders
from operator import attrgetter
from tqdm import tqdm
import torch
import os


# TODO: arrange this file as a pipeline

# Tokenization

def tokenizer_train_iterator(batch_size: int = 50, *datasets) -> Generator[list[str], None, None]:
    for ds in chain(*map(list, datasets)):
        for j in range(0, len(ds), batch_size):
            yield ds[j:batch_size+j]['text']


def data_iterator(batch_size: int, dataset: Dataset) -> Generator[list[str], None, None]:
    for j in range(0, len(dataset), batch_size):
        yield dataset[j:batch_size+j]['text']


def tokenize_dataset(dataset: Dataset, 
                     tokenizer: Tokenizer, 
                     path: str, 
                     batch_size: int, 
                     shrad_size: int, 
                     eos_tok: str) -> int:

    os.makedirs(path, exist_ok=True)

    shrad = torch.empty(shrad_size, dtype=torch.int16)

    p = 0
    k = 0

    eos = tokenizer.encode(eos_tok).ids 

    bar = tqdm(total=len(dataset), desc=f'Shrad 0')
    for seq_batch in data_iterator(batch_size, dataset):
        enc = tokenizer.encode_batch(seq_batch)
        
        for ids in map(attrgetter('ids'), enc):
            
            if p + len(ids) + 1 >= shrad_size:
                shrad[p:] = torch.tensor(eos + ids, dtype=torch.int16)[:shrad_size-p]
                torch.save(shrad, f'{path}/shrad_{k}.pt')
                p = 0
                k += 1
                bar.set_description(f'Shrad {k}')
            else:
                shrad[p:p+len(ids)+1] = torch.tensor(eos + ids, dtype=torch.int16)
                p += len(ids) + 1

        bar.update(batch_size)

    torch.save(shrad[:p].clone(), f'{path}/shrad_{k}.pt')

    return shrad_size * k + p



def pipeline():
    pass


def main() -> None:

    access_token = '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' # TODO: Change to PATH_VAR

    # Collect the data

    wiki = load_dataset("pszemraj/simple_wikipedia")
    books = load_dataset("suolyer/pile_books3")
    peS2o = load_dataset('nampdn-ai/mini-peS2o', token=access_token)

    # Tokenizer

    # tokenizer = ByteLevelBPETokenizer()
    # tokenizer.train_from_iterator(tokenizer_train_iterator(5, books.values(), peS2o.values(), wiki.values()), vocab_size=8192, min_frequency=2, special_tokens=['[EOS]'], show_progress=True)
    # tokenizer.save('tokenizer.json')
    tokenizer = Tokenizer.from_file('tokenizer_8192.json')

    # Datasets splits and merges

    books_split = concatenate_datasets([books['test'], books['validation']]).train_test_split(0.1)
    peS2o_split = peS2o['train'].train_test_split(0.05)

    data_train = concatenate_datasets([wiki['train'], books_split['train'], peS2o_split['train']])
    data_val = concatenate_datasets([wiki['test'], wiki['validation'], books_split['test'], peS2o_split['test']])

    # Tokenization

    batch_size = 50
    shrad_size = 10**8 
    path = 'data/val'
    eos_tok = '[EOS]'

    total_val_tokens = tokenize_dataset(data_val, tokenizer, path, batch_size, shrad_size, eos_tok)
    print(f'Total validation tokens: {total_val_tokens}')


if __name__ == '__main__':
    main()
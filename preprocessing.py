from typing import Generator
from datasets import load_dataset, concatenate_datasets, Dataset
from tokenizers import ByteLevelBPETokenizer, Tokenizer, normalizers, Regex
from operator import attrgetter
from tqdm import tqdm
import torch
import os


def data_iterator(batch_size: int, dataset: Dataset) -> Generator[list[str], None, None]:
    for j in range(0, len(dataset), batch_size):
        yield dataset[j:batch_size+j]['text_markdown']


def tokenize_dataset(dataset: Dataset, 
                     tokenizer: Tokenizer, 
                     path: str, 
                     batch_size: int, 
                     shard_size: int, 
                     eos_tok: str) -> int:

    os.makedirs(path, exist_ok=True)

    shard = torch.empty(shard_size, dtype=torch.int16)

    p = 0
    k = 0

    eos = tokenizer.encode(eos_tok).ids 

    bar = tqdm(total=len(dataset), desc=f'Shard 0')
    for seq_batch in data_iterator(batch_size, dataset):
        enc = tokenizer.encode_batch(seq_batch)
        
        for ids in map(attrgetter('ids'), enc):
            
            if p + len(ids) + 1 >= shard_size:
                shard[p:] = torch.tensor(eos + ids, dtype=torch.int16)[:shard_size-p]
                torch.save(shard, f'{path}/shard_{k}.pt')
                p = 0
                k += 1
                bar.set_description(f'Shard {k}')
            else:
                shard[p:p+len(ids)+1] = torch.tensor(eos + ids, dtype=torch.int16)
                p += len(ids) + 1

        bar.update(batch_size)

    torch.save(shard[:p].clone(), f'{path}/shard_{k}.pt')

    return shard_size * k + p


def pipeline(vocab_size: int = 8192, data_root: str = 'data', tokenizer_root: str = 'tokenizer') -> Tokenizer:

    # Collect the data

    habr = load_dataset("IlyaGusev/habr")

    # Tokenizer
    
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.normalizer = normalizers.Replace(Regex('[[:space:]]+'), ' ')
    tokenizer.train_from_iterator(data_iterator(30, habr['train']), vocab_size=vocab_size, min_frequency=2, special_tokens=['[EOS]'], show_progress=True)
    tokenizer.save(f'{tokenizer_root}/tokenizer.json')

    # Dataset split

    habr_split = habr['train'].train_test_split(0.05)

    data_train = habr_split['train']
    data_val = habr_split['test']

    # Tokenization

    os.makedirs(f'{data_root}/train', exist_ok=True)
    os.makedirs(f'{data_root}/val', exist_ok=True)
    
    batch_size = 50
    shard_size = 2*10**7 
    path = f'{data_root}/val'
    eos_tok = '[EOS]'

    total_val_tokens = tokenize_dataset(data_val, tokenizer, path, batch_size, shard_size, eos_tok)
    print(f'Total validation tokens: {total_val_tokens}')

    path = f'{data_root}/train'
    shard_size = 10**8

    total_train_tokens = tokenize_dataset(data_train, tokenizer, path, batch_size, shard_size, eos_tok)
    print(f'Total train tokens: {total_train_tokens}')

    return tokenizer

if __name__ == '__main__':
    pipeline()
import os
import preprocessing
from tokenizers import Tokenizer
from model import GPT, sample
from train import trainer
import torch


# Operating mode
# 0 - Inference mode
# 1 - Training mode
MODE = 0

def main() -> None:
    
    data_root = 'data'
    tokenizer_root = 'tokenizer'
    params_root = 'params'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    n_ctx = 512
    vocab_size = 8192
    n_embed = 128
    n_layer = 8
    n_head = 8

    model = GPT(n_ctx, vocab_size, n_embed, n_layer, n_head)
    model.to(device)

    if MODE:
        print('[+] Training mode enabled')
        print(f'[+] Detected device: {device.type}')

        print('[+] Start data processing')
        tokenizer = preprocessing.pipeline(vocab_size, data_root, tokenizer_root)

        print('[+] Start training the model')
        trainer(model, device, data_root, params_root, n_ctx)
        # torch.save(model.state_dict(), f'{params_root}/params.pt')
    else:
        tokenizer = Tokenizer.from_file(f'{tokenizer_root}/tokenizer.json')
        model.load_state_dict(torch.load(f'{params_root}/params.pt'))

    print('Start generating (Ctrl+D to terminate)')

    temperature = 0.6

    while True:
        try:
            prompt = input('Prompt: ')
        except EOFError as e:
            break

        print(f'\033[FPrompt: {prompt}', end='')
        for tok in sample(model, tokenizer, device, prompt, temperature):
            print(tok, end='', flush=True)
        print()


if __name__ == '__main__':
    main()
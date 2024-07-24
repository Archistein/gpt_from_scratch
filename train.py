import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from datasets import Dataset
from random import randint
from math import exp
import os


# Sharded torch dataset

class ShardedDataset(torch.utils.data.Dataset):

    def __init__(self, n_ctx: int, path: str):
        self.n_ctx = n_ctx
        self.curr_shard_idx = 0
        self.shard_files = self._get_shards(path)
        self.length = self._count_length()
        self.bound = self.load_shard(self.curr_shard_idx)

    def _get_shards(self, path: str) -> list[str]:
        return sorted([os.path.join(path, f) for f in os.listdir(path) if f.endswith('.pt')])

    def _count_length(self) -> int:
        shrad_size = len(torch.load(self.shard_files[0]))
        last_size = len(torch.load(self.shard_files[-1]))
    
        return ((shrad_size - shrad_size % self.n_ctx)* (len(self.shard_files) - 1) 
              + last_size - last_size % self.n_ctx) // self.n_ctx

    def load_shard(self, shard_idx: int) -> int:
        self.data = torch.load(self.shard_files[shard_idx])
        self.data = self.data[:-(len(self.data)%self.n_ctx)]
        self.curr_shard_len = len(self.data) // self.n_ctx
        return self.curr_shard_len

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        shard_idx = idx // self.bound
        
        if shard_idx != self.curr_shard_idx:
            self.curr_shard_idx = shard_idx
            self.load_shard(self.curr_shard_idx)             

        local_idx = idx % self.curr_shard_len
        x = self.data[self.n_ctx*local_idx:self.n_ctx*(local_idx+1)].long()
        y = torch.cat((x[1:], torch.zeros(1, dtype=torch.long)))

        return x, y
    

# Evaluation

@torch.inference_mode
def evaluate(model: nn.Module, 
             criterion: callable,
             steps: int, 
             data_loader: DataLoader,
             device: torch.device
            ) -> float:
    
    model.eval()

    average_loss = 0
    
    random_skip = randint(0, len(data_loader) - steps)
    
    for i, (seqs, targets) in enumerate(data_loader):
        
        if (i < random_skip): continue # Sliding window
        
        seqs, targets = seqs.to(device), targets.to(device)
        
        #with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        logits = model(seqs)
        loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            
        average_loss += loss.item()

        if (i - random_skip >= steps): break

    model.train()

    return average_loss/steps


# Training

def trainer(gpt: nn.Module,
            device: torch.device,
            data_root: str,
            params_root: str,
            n_ctx: int = 512,
            batch_size: int = 128,
            epoch: int = 10,
            lr: float = 3e-4,
            grad_clip: int = 1
           ) -> None:   
    
    train_data_loader = DataLoader(ShardedDataset(n_ctx, f'{data_root}/train'), batch_size)
    val_data_loader = DataLoader(ShardedDataset(n_ctx, f'{data_root}/val'), batch_size)

    opt = optim.AdamW(gpt.parameters(), lr)
    lmbda = lambda epoch: 0.95
    scheduler = optim.lr_scheduler.MultiplicativeLR(opt, lr_lambda=lmbda)
    criterion = nn.CrossEntropyLoss()

    gpt.to(device)

    train_loss = 0

    train_loss_step = 10
    val_loss_step = 50
    checkpoint_step = 500

    for e in range(epoch):
        
        for i, (seqs, targets) in enumerate(train_data_loader):

            seqs, targets = seqs.to(device), targets.to(device)

            opt.zero_grad()

            #with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits = gpt(seqs)
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            
            train_loss += loss.item()
            
            loss.backward()

            nn.utils.clip_grad_norm_(gpt.parameters(), grad_clip)
            opt.step()
            
            if (i > 0 and i % train_loss_step == 0):
                print(f'Epoch [{e}/{epoch-1}] Batch [{i}/{len(train_data_loader)-1}] Loss: {train_loss/train_loss_step:.4f} Perplexity: {exp(train_loss/train_loss_step):.4f}')
                train_loss = 0

            if (i > 0 and i % val_loss_step == 0):
                val_loss = evaluate(gpt, criterion, 20, val_data_loader, device)
                print(f'Val loss: {val_loss:4f} Val Perplexity: {exp(val_loss):4f}')

            if (i > 0 and i % checkpoint_step == 0):
                torch.save({
                    'model_state_dict': gpt.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                }, f'checkpoint_{i + e*len(train_data_loader)}.pth')
                
        scheduler.step()

    torch.save(gpt.state_dict(), f'{params_root}/params.pt')
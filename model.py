import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
from typing import Generator

# GPT Model

class FeedForward(nn.Module):

    def __init__(self, n_embed: int):
        super().__init__()
        self.fc = nn.Linear(n_embed, 4 * n_embed)
        self.proj = nn.Linear(4 * n_embed, n_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.fc(x))
        x = self.proj(x)
        return x


class CasualSelfAttention(nn.Module):

    def __init__(self, n_embed: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.n_embed = n_embed

        self.c_attn = nn.Linear(n_embed, 3 * n_embed)
        self.c_proj = nn.Linear(n_embed, n_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()

        comb = self.c_attn(x)
        q, k, v = comb.split(self.n_embed, dim=-1)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.c_proj(out)
        return out
        

class DecoderBlock(nn.Module):

    def __init__(self, n_embed: int, n_head: int):
        super().__init__()
        self.fln = nn.LayerNorm(n_embed)
        self.atten = CasualSelfAttention(n_embed, n_head)
        self.sln = nn.LayerNorm(n_embed)
        self.fc = FeedForward(n_embed)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.atten(self.fln(x))
        x = x + self.fc(self.sln(x))
        return x


class GPT(nn.Module):

    def __init__(self, n_ctx: int, vocab_size: int, n_embed: int, n_layer: int, n_head: int):
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, n_embed)
        self.pos_embed = nn.Embedding(n_ctx, n_embed)
        self.decoders = nn.Sequential(*[DecoderBlock(n_embed, n_head) for _ in range(n_layer)])
        self.ln = nn.LayerNorm(n_embed)
        self.clf = nn.Linear(n_embed, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pos = torch.arange(0, x.size(-1), device=x.device)
        t_emb = self.tok_embed(x)
        p_emb = self.pos_embed(pos)
        emb = t_emb + p_emb
        out = self.decoders(emb)
        out = self.clf(self.ln(out))

        return out
    

@torch.inference_mode
def sample(model: nn.Module, 
           tokenizer: Tokenizer, 
           device: torch.device, 
           prompt: str = '',
           temperature: int = 0.5,
           max_length: int = 100, 
           eos_tok: int = 0,
          ) -> Generator[str, None, None]:

    seq = [eos_tok] + tokenizer.encode(prompt).ids

    for _ in range(max_length):
        t = torch.tensor(seq, device=device).unsqueeze(0)
        
        #with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        logits = model.forward(t)[0][-1]
        
        next_tok = torch.multinomial(F.softmax(logits / temperature, dim=0), 1).item()
        seq.append(next_tok)

        if next_tok == eos_tok:
            break

        yield tokenizer.decode([next_tok])
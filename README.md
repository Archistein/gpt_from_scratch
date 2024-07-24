# GPT from scratch

This is my attempt to create and pre-train a GPT in Russian language from scratch. For training was chosen the [dataset](https://huggingface.co/datasets/IlyaGusev/habr) of articles with habr.com. The training was not easy, but at the end of the day, the model (despite its relatively small size) generates coherent text in Russian.


## Model

Size: 3749120 params

<details>
  <summary>
    Architecture
  </summary>

  ```
GPT(
  (tok_embed): Embedding(8192, 128)
  (pos_embed): Embedding(512, 128)
  (decoders): Sequential(
    (0): DecoderBlock(
      (fln): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
      (atten): CasualSelfAttention(
        (c_attn): Linear(in_features=128, out_features=384, bias=True)
        (c_proj): Linear(in_features=128, out_features=128, bias=True)
      )
      (sln): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
      (fc): FeedForward(
        (fc): Linear(in_features=128, out_features=512, bias=True)
        (proj): Linear(in_features=512, out_features=128, bias=True)
      )
    )
    (1): DecoderBlock(
      (fln): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
      (atten): CasualSelfAttention(
        (c_attn): Linear(in_features=128, out_features=384, bias=True)
        (c_proj): Linear(in_features=128, out_features=128, bias=True)
      )
      (sln): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
      (fc): FeedForward(
        (fc): Linear(in_features=128, out_features=512, bias=True)
        (proj): Linear(in_features=512, out_features=128, bias=True)
      )
    )
    (2): DecoderBlock(
      (fln): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
      (atten): CasualSelfAttention(
        (c_attn): Linear(in_features=128, out_features=384, bias=True)
        (c_proj): Linear(in_features=128, out_features=128, bias=True)
      )
      (sln): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
      (fc): FeedForward(
        (fc): Linear(in_features=128, out_features=512, bias=True)
        (proj): Linear(in_features=512, out_features=128, bias=True)
      )
    )
    (3): DecoderBlock(
      (fln): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
      (atten): CasualSelfAttention(
        (c_attn): Linear(in_features=128, out_features=384, bias=True)
        (c_proj): Linear(in_features=128, out_features=128, bias=True)
      )
      (sln): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
      (fc): FeedForward(
        (fc): Linear(in_features=128, out_features=512, bias=True)
        (proj): Linear(in_features=512, out_features=128, bias=True)
      )
    )
    (4): DecoderBlock(
      (fln): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
      (atten): CasualSelfAttention(
        (c_attn): Linear(in_features=128, out_features=384, bias=True)
        (c_proj): Linear(in_features=128, out_features=128, bias=True)
      )
      (sln): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
      (fc): FeedForward(
        (fc): Linear(in_features=128, out_features=512, bias=True)
        (proj): Linear(in_features=512, out_features=128, bias=True)
      )
    )
    (5): DecoderBlock(
      (fln): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
      (atten): CasualSelfAttention(
        (c_attn): Linear(in_features=128, out_features=384, bias=True)
        (c_proj): Linear(in_features=128, out_features=128, bias=True)
      )
      (sln): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
      (fc): FeedForward(
        (fc): Linear(in_features=128, out_features=512, bias=True)
        (proj): Linear(in_features=512, out_features=128, bias=True)
      )
    )
    (6): DecoderBlock(
      (fln): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
      (atten): CasualSelfAttention(
        (c_attn): Linear(in_features=128, out_features=384, bias=True)
        (c_proj): Linear(in_features=128, out_features=128, bias=True)
      )
      (sln): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
      (fc): FeedForward(
        (fc): Linear(in_features=128, out_features=512, bias=True)
        (proj): Linear(in_features=512, out_features=128, bias=True)
      )
    )
    (7): DecoderBlock(
      (fln): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
      (atten): CasualSelfAttention(
        (c_attn): Linear(in_features=128, out_features=384, bias=True)
        (c_proj): Linear(in_features=128, out_features=128, bias=True)
      )
      (sln): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
      (fc): FeedForward(
        (fc): Linear(in_features=128, out_features=512, bias=True)
        (proj): Linear(in_features=512, out_features=128, bias=True)
      )
    )
  )
  (ln): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
  (clf): Linear(in_features=128, out_features=8192, bias=False)
)
  ```

</details>

## Usage

To retrain model just change mode in `main.py` to 1 (training mode).

To generate:

```bash
$ python main.py
Start generating (Ctrl+D to terminate)
Prompt: Привет, в этой статье
```

Type in your prompt (or don't) and press enter:

```bash
$ python main.py
Prompt: Привет, в этой статье мы поговорим о том, как мы можем построить самую важную часть процесса разработки. С одной стороны, для нас было создано множество различных функций, которые можно использовать для создания нового приложения. Они нам нужны, чтобы...
```

## Requirements

+ datasets==2.20.0
+ tokenizers==0.19.1
+ torch==2.3.1
+ tqdm==4.66.4

### Have a fun :)
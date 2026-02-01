# tiny-lm

A learning-focused repository for building small language models from scratch.

## Objective

Implement transformer architectures from first principles, train them on small datasets, and compare their behavior. The goal is to better understand how LLMs work by building each component manually.

## Planned Architectures

- **GPT-2**: Learned position embeddings, LayerNorm, GELU
- **LLaMA-3**: RoPE, RMSNorm, SwiGLU, Grouped Query Attention

## Features

- Config-driven design (adjustable vocab size, layers, hidden dimensions)
- BPE tokenizer training
- Support for multiple datasets
- Training and evaluation pipeline

## Inspiration

- [nanoGPT](https://github.com/karpathy/nanoGPT) - Simple, readable GPT training code
- [Mini-LLM](https://github.com/Ashx098/Mini-LLM) - Complete LLM pipeline from scratch with modern architecture
- [TinyStories](https://huggingface.co/papers/2305.07759) - Showing small models can generate coherent text on focused datasets
- [Baguettotron](https://huggingface.co/PleIAs/Baguettotron) - A recent 321M parameter model demonstrating what small models can achieve
- [SmolLM3](https://huggingface.co/blog/smollm3) - Full training recipe with architecture details and data mixtures

## Status

Work in progress.

# Small language models

## [TinyStories: How Small Can Language Models Be and Still Speak Coherent English?](https://arxiv.org/pdf/2305.07759)

This paper introduces *TinyStories*, a dataset of simple stories for training small language models (under 10M parameters). Despite their size, these models produce fluent, grammatically correct, multi-paragraph text. A new evaluation framework using GPT-4 grades outputs on grammar, creativity, and instruction-following. The work highlights the emergence of language capabilities in small LMs and supports their use in specialized domains.

---


## [SUPER TINY LANGUAGE MODELS](https://arxiv.org/pdf/2405.14159)

This paper explores Super Tiny Language Models (STLMs), which aim to achieve high performance with significantly fewer parameters (10M to 100M). It introduces techniques like byte-level tokenization, weight tying, and efficient training methods to reduce parameter counts while maintaining transformer model performance. The goal is to make high-performance models more accessible, addressing challenges of large LLMs, such as high computational costs and limited accessibility. Future work will explore tokenizer-free models, self-play training, and alternative objectives to improve parameter and sample efficiency.

---

## [Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling](https://arxiv.org/pdf/2304.01373)

This paper provides a lot of insights on training SLMs. Developed by Eleuther AI.

---

## [1.5-Pints Technical Report: Preatrining in Days, not months - Your Language model thrives on Quality data](https://arxiv.org/pdf/2408.03506)

This paper presents a compute-efficient approach to pre-training a Language  Model – the "1.5-Pints" – in only 9 days, while outperforming state-of-the-art  models as an instruction-following assistant. Based on MT-Bench (a benchmark  that emulates human judgments), 1.5-Pints outperforms Apple’s OpenELM and Microsoft’s Phi.

---

## [Honey, I Shrunk the Language: Language Model Behavior at Reduced Scale](https://arxiv.org/pdf/2305.17266)

Although focused on MLM tasks, it offers a similar perspective on what I want. The most interesting bit to me is using the [AO-CHILDES](https://www.sciencedirect.com/science/article/abs/pii/S0079742121000256) corpus of 21.000 words to reduce the vocabulary of the datasets

---

## [Getting the most out of your tokenizer for pre-training and domain adaptation](https://arxiv.org/pdf/2402.01035v2)

This paper highlights the importance of optimizing tokenization in large language models (LLMs), an often overlooked component. The authors show that the size, regular expression, and training data of a tokenizer can significantly affect model performance, including generation speed, memory usage, and context size. They conduct experiments with specialized Byte-Pair Encoding tokenizers, focusing on code generation tasks like HumanEval and MBPP. The study finds that fine-tuning a tokenizer on over 50 billion tokens can improve the generation speed and context size of pre-trained models, offering valuable recommendations for tokenizer design and fine-tuning strategies.


---

# Reasoning

## [Marco-o1: Towards Open Reasoning Models for Open-Ended Solutions](https://arxiv.org/pdf/2411.14405)

The *Marco-o1* model advances reasoning in LLMs using Chain-of-Thought fine-tuning, Monte Carlo Tree Search (MCTS), and self-reflective reasoning strategies. It achieves state-of-the-art results on reasoning tasks (+6% MGSM accuracy) and excels in nuanced translation, including colloquial expressions. Marco-o1 sets a new benchmark for c

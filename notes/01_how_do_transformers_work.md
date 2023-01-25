# How do Transformers work?

## Content

- [History](#history)
- [Transformers are language models](#transformers-are-language-models)
- [Transformers are big models](#transformers-are-big-models)
- [Transfer learning](#transfer-learning)
- [General architecture](#general-architecture)
    - [Introduction](#introduction)
    - [Attention layers](#attention-layers)
    - [Original architecture](#the-original-architecture)
- [Architectures vs. checkpoints](#architectures-vs-checkpoints)

## History

- **June 2017**: Transformer architecture
- **June 2018**: GPT, the first pretrained Transformer model, used for fine-tuning on various NLP tasks and obtained state-of-the-art results
- **October 2018**: BERT, another large pretrained model, this one designed to produce better summaries of sentences
- **February 2019**: GPT-2, an improved (and bigger) version of GPT that was not immediately publicly released due to ethical concerns
- **October 2019**: DistilBERT, a distilled version of BERT that is 60% faster, 40% lighter in memory, and still retains 97% of BERT’s performance
- **October 2019**: BART and T5, two large pretrained models using the same architecture as the original Transformer model (the first to do so)
- **May 2020**: GPT-3, an even bigger version of GPT-2 that is able to perform well on a variety of tasks without the need for fine-tuning (called zero-shot learning)

Three categories of Transformer models:
- GPT-like = *auto-regressive*
- BERT-like = *auto-encoding*
- BART/T5-like = *sequence-to-sequence*

## Transformers are language models

Models that have been trained on **large amounts of raw text** in a **self-supervised** fashion. They are not trained for specific tasks, that is why the model then goes through **transfer learning**.

## Transformers are big models

General strategy for a better performance: 
- increase models' sizes
- increase the amount of training data

-> need time and computational resources
=> environmental impact (ML CO2 Impact or Code Carbon to estimate the footprint of your training)

## Transfer Learning

- *Pretraining*: training a model from scratch
- *Fine-tuning*: training after the model has been pretrained, on a specific dataset for a specific task
    + Lower time, data, financial, and environmental costs
    + Training less contraining than a full pretraining
    + Better results than training from scratch

## General architecture

### Introduction

Two blocks:
- **Encoder**: input --> build representation (features)
- **Decoder**: features (encoder's outputs) + other inputs --> outputs

Types of architectures:
- **Encoder-only models**
    - Sentence classification
    - Named entity recognition
- **Decoder-only models**
    - Generative tasks e.g. text generation
- **Encoder-decoder models** or **sequence-to-sequence models**
    - Generative tasks that require an input, e.g. translation or summarization

### Attention layers

Tell the models to **pay specific attention to certain words** in the sentence you passed.

### The original architecture

Originally designed for translation. 

- Encoder: sentences in a certain language
    - Uses all the words to generate a representation
- Decoder: same sentences in the desired target language. 
    - Works sequentially, so only pays attention to words that have already been translated.
    - Predict the translation of the next word
    - It is fed the whole target but only allows to use future words

## Architectures vs. checkpoints

- **Architecture**: skeleton of the model = definition of each layer and each operation
- **Checkpoints**: weights loaded in a given architecture
- **Model**: can mean both architecture or checkpoints.
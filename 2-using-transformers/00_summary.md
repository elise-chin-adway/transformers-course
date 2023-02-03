# Using transformers - Summary

Pipeline: preprocessing (tokenizer), passing the inputs through the model, postprocessing

## Content 

- [Preprocessing](#preprocessing-with-a-tokenizer)
    - [Encoding](#encoding)
    - [Types of tokenization](#types-of-tokenization)
    - [Handling multiple sequences](#handling-multiple-sequences)
    - [Configure `tokenizer`](#configure-tokenizer)
- [Model](#going-through-the-model)
    - [Loading](#loading-models)
    - [Saving](#saving-models)
- [Postprocessing](#postprocessing-the-output)

## Preprocessing with a tokenizer

### Encoding
- Inputs --> Tokens (words, subwords, or symbols e.g. punctuation)
- Token --> Integer
- Add additional inputs that may be useful to the model

```python
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Encoding = tokenize + conversion to input IDs
tokenizer("Sequence") 
    # Output: dict('input_ids', 'token_type_ids', 'attention_mask')

# Tokenize
tokens = tokenizer.tokenize("Sequence")
    # Output: list of tokens

# Conversion to inputs IDs
ids = tokenizer(convert_tokens_to_ids(tokens))
    # Output: list of ids
```

### Types of tokenization

| Tokenization | Example | Description | Pros | Cons |
| ------------ | --------| ----------- | ---- | ---- |
| **Word-based** | | Text --> words | | <ul><li>Large vocabulary</li><li>Singular and plural nouns or verbal forms have different IDs</li><li>Lots of unknown tokens</li></ul> |
| **Character-based** | | Text --> characters | <ul><li>Smaller vocabulary size</li><li>Fewer unknown tokens</li></ul> | <ul><li>Spaces and punctuations?</li><li>Character less meaningful in Latin language</li><li>Large amount of tokens to be processed by our model</li></ul> | 
| **Subword-based** | **Byte-level BPE** (GPT-2), **WordPiece** (BERT), **SentencePiece or Unigram** | <ul><li>Frequently used words not split into smaller subwords</li><li>Rare words --> subwords</li></ul> | | |


### Handling multiple sequences

- **Padding** to make sure all our sentences have the same length by adding a a *padding token* = `tokenizer.pad_token_id`
- **Attention masks** to ignore padding tokens

```python

batched_ids = [
    [200, 200, 200],
    [200, 200, tokenizer.pad_token_id],
]

attention_mask = [
    [1, 1, 1],
    [1, 1, 0],
]

outputs = model(torch.tensor(batched_ids), attention_mask=torch.tensor(attention_mask))
```

### Configure `tokenizer`

- **Padding**
```python
# Will pad the sequences up to the maximum sequence length
model_inputs = tokenizer(sequences, padding="longest")

# Will pad the sequences up to the model max length
# (512 for BERT or DistilBERT)
model_inputs = tokenizer(sequences, padding="max_length")

# Will pad the sequences up to the specified max length
model_inputs = tokenizer(sequences, padding="max_length", max_length=8)
```

- **Truncation**
```python
sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]

# Will truncate the sequences that are longer than the model max length
# (512 for BERT or DistilBERT)
model_inputs = tokenizer(sequences, truncation=True)

# Will truncate the sequences that are longer than the specified max length
model_inputs = tokenizer(sequences, max_length=8, truncation=True)
```

- **Conversion**
```python
sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]

# Returns PyTorch tensors
model_inputs = tokenizer(sequences, padding=True, return_tensors="pt")

# Returns TensorFlow tensors
model_inputs = tokenizer(sequences, padding=True, return_tensors="tf")

# Returns NumPy arrays
model_inputs = tokenizer(sequences, padding=True, return_tensors="np")
```

## Going through the model

- Input --> Transformer network = [Embeddings + Layers] --> Hidden states --> [Head] --> Task specific output
- Forms of heads: language modeling heads, question answering heads, sequence classification heads, etc.
- Output shape: `[batch_size, sequence_length, hidden_size]`
- Output logits shape: `[n_samples, n_labels]`

### Loading models

```python
# Random initialization
config = BertConfig()
model = BertModel(config)

# Pre-trained models
model = AutoModel.from_pretrained(checkpoint) # with AutoModel
model = BertModel.from_pretrained("bert-base-cased") # with BertModel
```

### Saving models

```python
model.save_pretrained("path/to/directory")
```

## Postprocessing the output

- `torch.nn.functional.softmax(outputs.logits)`

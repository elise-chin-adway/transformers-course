# Encoder, decoder and encoder-decoder models


| Model | Examples | Description | Pretraining | Tasks | 
| ----- | -------- | ----------- | ----------- | ----- |
| Encoder, *auto-encoding models* | ALBERT, BERT, DistilBERT, ELECTRA, RoBERTa | Attention layers can access all the words in the initial sentence | Corrupting a given sentence (e.g. masking random words) and tasking the model with finding or reconstructing the initial sentence | Sentence classification, named entity recognition, extractive question answering |
| Decoder, *auto-regressive models* | CTRL, GPT, GPT-2, Transformer XL | For a given words, the attention layers can only access the words positioned before it in the sentence | Predicting the next word in a sentence | Text generation |
| Encoder-decoder, *sequence-to-sequence models* | BART, T5, Marian, mBART | At each stage, the attention layers of the encoder can access all the words in the initial sentence, whereas the attention layers of the decoder can only access the words positioned before a given word in the input | Can be done using the objectives of encoder or decoder models | Summarization, translation, generative question answering |
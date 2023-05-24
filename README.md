# Transformer-based-machine-translation
Transformer from scratch with BPE (byte pair encoding) tokenizer for machine translation

I took the transformer pytorch code from [here](https://github.com/bentrevett/pytorch-seq2seq) and I added the byte pair encoding tokenizer in it.

We trained model on Multi30k dataset (see data directory). There was two languages german (.de) and english (.en).

For getting the vocabulary, we first keep all the data (train, valid, text) into one file for english and similarly for german language. We then train the tokenizer to get the vocabulary.

### Run the code
  python translation.py en de        (English to German translation)
  python translation.py de en        (German to English translation)

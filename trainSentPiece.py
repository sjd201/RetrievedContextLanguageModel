import sentencepiece as spm
spm.SentencePieceTrainer.train(input="wiki103/corpus", model_prefix = "test", vocab_size=15000, split_by_whitespace = False, split_by_unicode_script=False)

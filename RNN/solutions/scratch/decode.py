word_to_idx = {w: i for i, w in enumerate(vocab)}
idx_to_word = {i: w for w, i in word_to_idx.items()}

print(word_to_idx['good'])
print(idx_to_word[0])
embedding_size = 32

rnn = Sequential(name="RNN")
rnn.add(Embedding(vocab_size, embedding_size, input_length=max_words))
rnn.add(LSTM(int(.5*embedding_size)))
rnn.add(Dropout(0.1))
rnn.add(Dense(1, activation='sigmoid'))

print(rnn.summary())
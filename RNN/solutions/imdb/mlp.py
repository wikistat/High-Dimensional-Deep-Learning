embedding_size = 32

mlp = Sequential(name="MLP")
mlp.add(Embedding(vocab_size, embedding_size, input_length=max_words))
mlp.add(Flatten())
mlp.add(Dropout(0.5))
mlp.add(Dense(5))
mlp.add(Dense(1, activation='sigmoid'))

mlp.summary()


# ----- #


batch_size = 100
num_epochs = 8

X_valid, y_valid = X_train_pad[:batch_size], y_train[:batch_size]
X_train_rnn, y_train_rnn = X_train_pad[batch_size:], y_train[batch_size:]


mlp.compile(loss='binary_crossentropy', 
             optimizer='adam', 
             metrics=['accuracy'])

history_mlp = mlp.fit(X_train_rnn, 
                    y_train_rnn, 
                    validation_data=(X_valid, y_valid), 
                    batch_size=batch_size, 
                    epochs=num_epochs)

plotTraining(history_mlp)
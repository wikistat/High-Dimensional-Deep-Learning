batch_size = 100
num_epochs = 8

X_valid, y_valid = X_train_pad[:batch_size], y_train[:batch_size]
X_train_rnn, y_train_rnn = X_train_pad[batch_size:], y_train[batch_size:]


rnn.compile(loss='binary_crossentropy', 
             optimizer='adam', 
             metrics=['accuracy'])

history_rnn = rnn.fit(X_train_rnn, 
                    y_train_rnn, 
                    validation_data=(X_valid, y_valid), 
                    batch_size=batch_size, 
                    epochs=num_epochs)
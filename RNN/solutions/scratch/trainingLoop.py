rnn = RNN(vocab_size, 2)

# Loop over each training example
for x, y in train_data.items():
    inputs = createInputs(x)
    target = int(y)

    # Forward
    out, _ = rnn.forward(inputs)
    probs = softmax(out)

    # Build dL/dy
    d_L_d_y = probs
    d_L_d_y[target] -= 1
    
    # Backward
    rnn.backprop(d_L_d_y)
    
print(probs)
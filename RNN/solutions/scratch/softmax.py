def softmax(x):
    # Applies the Softmax func<tion to the input array.
    return np.exp(x) / np.sum(np.exp(x))
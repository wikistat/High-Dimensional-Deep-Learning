def one_hot(y, n_classes):
    ohy = np.eye(n_classes)[y]
    return ohy
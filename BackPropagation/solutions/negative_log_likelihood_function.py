def NegLogLike(Y_true, Y_pred):
    Y_true = Y_true
    Y_pred = Y_pred
    Y_prod = np.multiply(Y_true,Y_pred)
    Y_sum = np.sum(Y_prod, axis=-1)
    nll = -np.log(Y_sum+EPSILON)
    nll_mean = np.mean(nll)
    return nll_mean

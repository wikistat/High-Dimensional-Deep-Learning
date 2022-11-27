def decodeReview(idx):
    '''
    Converts the encoded idx-th review to human readable form.
    Displays the review number, the review in words and the label
    '''
    print('---review number---')
    print(idx)

    print('\n---review in words---')
    print(" ".join(idx_to_word[i] for i in X_train[idx]))

    print('\n---label---')
    print(y_train[idx])
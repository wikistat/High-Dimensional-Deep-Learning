lengths = []
for x in X_train_pad+X_test_pad:
    lengths.append(len(x))
    
print('Maximum review length: {}'.format(max(lengths)))
print('Minimum review length: {}'.format(min(lengths)))
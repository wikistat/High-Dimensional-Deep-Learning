train_lengths = []
for x in X_train:
    train_lengths.append(len(x))
    
test_lengths = []
for x in X_test:
    test_lengths.append(len(x))
    
    
plt.figure(figsize = (10,6))
sns.histplot(x=train_lengths,color='orange',alpha=.8)
sns.histplot(x=test_lengths,alpha=.5)
plt.legend(['train lengths','test lengths'])    


print('Maximum review length: {}'.format(max(train_lengths+test_lengths)))
print('Minimum review length: {}'.format(min(train_lengths+test_lengths)))
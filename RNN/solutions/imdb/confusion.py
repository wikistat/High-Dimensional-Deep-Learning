def confusion(model, X=X_test_pad, y=y_test):
    """
    Displays the confusion matrix for the model "model"
    """
    y_pred = model.predict(X)
    y_pred = (y_pred > 0.5)
    cm = confusion_matrix(y,y_pred)

    df1 = pd.DataFrame(columns=["True","False"], index= ["True","False"], data= cm ) 

    f,ax = plt.subplots(figsize=(2,2))
    sns.heatmap(df1, annot=True,cmap="Blues", fmt= '.0f',ax=ax)
    plt.xlabel("Predicted Label")
    plt.xticks(size = 12)
    plt.yticks(size = 12, rotation = 0)
    plt.ylabel("True Label")
    plt.title("Confusion Matrix - "+model.name, size = 14)
    plt.show()

    print ("True Positive:" , (cm[0,0]))
    print ("True Negative:" , (cm[1,1]))
    print ("False Positive:" , (cm[0,1]))
    print ("False Negative:" , (cm[1,0]))
    print (" ")
    
    
confusion(rnn)
confusion(bi_rnn)
confusion(bi2_rnn)
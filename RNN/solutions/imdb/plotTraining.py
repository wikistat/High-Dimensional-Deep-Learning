def plotTraining(history):
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy') 
    plt.xlabel('epoch')
    plt.legend(['train','test'], loc='upper right')
    
    plt.subplot(1,2,2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss') 
    plt.xlabel('epoch')
    plt.legend(['train','test'], loc='upper right')
data_dir_test = data_dir+'test/'
N_test = len(os.listdir(data_dir_test+"/test"))

test_datagen = kpi.ImageDataGenerator(rescale=1. / 255)

test_generator = test_datagen.flow_from_directory(
    data_dir_test,
    #data_dir_sub+"/train/",
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode=None,
    shuffle=False)

test_prediction = model_VGG_LastConv_fcm.predict_generator(test_generator, N_test // batch_size)

images_test = [data_dir_test+"/test/"+k for k in os.listdir(data_dir_test+"/test")][:9]
x_test  = [kpi.img_to_array(kpi.load_img(image_test))/255 for image_test in images_test]  # this is a PIL image

fig = plt.figure(figsize=(10,10))
for k in range(9):
    ax = fig.add_subplot(3,3,k+1)
    ax.imshow(x_test[k], interpolation='nearest')
    pred = test_prediction[k]
    if pred >0.5:
        title = "Probabiliy for dog : %.1f" %(pred*100)
    else:
        title = "Probabiliy for cat : %.1f" %((1-pred)*100)
    ax.set_title(title)
plt.show()
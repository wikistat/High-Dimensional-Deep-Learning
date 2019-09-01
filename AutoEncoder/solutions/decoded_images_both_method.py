decoded_imgs_1 = decoder.predict(encoded_imgs)
decoded_imgs_2 = autoencoder.predict(x_test_flatten)
print("Are all pair of decoded images from both methods equal : %s " %np.all(decoded_imgs_1==decoded_imgs_2))

n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(4, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display encoded imgs
    ax = plt.subplot(4, n, i + 1 + n)
    plt.imshow(encoded_imgs[i].reshape(8, 4))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    # display encoded imgs
    ax = plt.subplot(4, n, i + 1 + 2*n)
    plt.imshow(decoded_imgs_1[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    # display encoded imgs
    ax = plt.subplot(4, n, i + 1 + 3*n)
    plt.imshow(decoded_imgs_2[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
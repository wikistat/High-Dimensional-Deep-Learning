# If only x and y are indicated, an image number is randomly drawn and the y label associated with the image is displayed
# If a 2nd y, named y_pred, is indicated, then the two labels are displayed side by side, so that they can be compared
# Finally, you can also indicate the id of the image you wish to view.

def print_data_localisation(x, y, y_pred=[], id=None, image_size=IMAGE_SIZE):
    if id==None:
        # Random drawing of an image in the database
        num_img = np.random.randint(x.shape[0]-1)
    else:
        num_img = id

    img = x[num_img]
    lab = y[num_img]

    colors = ["blue", "yellow", "red", "orange"] # Different colors for different classes
    classes = ["Buffalo", "Elephant", "Rhino", "Zebra"]

    if np.any(y_pred):
        plt.subplot(1, 2, 1)

    # Image display
    plt.imshow(img)
    # Determination of the class
    class_id = np.argmax(lab[5:])

    # Determining the coordinates of the bounding box in the image frame
    ax = (lab[1]*y_std[1] + y_mean[1]) * image_size
    ay = (lab[2]*y_std[2] + y_mean[2]) * image_size
    width = (lab[3]*y_std[3] + y_mean[3]) * image_size
    height = (lab[4]*y_std[4] + y_mean[4]) * image_size
    #print("x: {}, y: {}, w: {}, h:{}".format(ax,ay,width, height))
    # Determination of the extrema of the bounding box
    p_x = [ax-width/2, ax+width/2]
    p_y = [ay-height/2, ay+height/2]
    # Display the bounding box in the right color
    plt.plot([p_x[0], p_x[0]],p_y,color=colors[class_id])
    plt.plot([p_x[1], p_x[1]],p_y,color=colors[class_id])
    plt.plot(p_x,[p_y[0],p_y[0]],color=colors[class_id])
    plt.plot(p_x,[p_y[1],p_y[1]],color=colors[class_id])
    plt.title("Ground truth : Image {} - {}".format(num_img, classes[class_id]))

    if np.any(y_pred):
        plt.subplot(1, 2, 2)
        # Image display
        plt.imshow(img)
        lab = y_pred[num_img]
        # Determination of the class
        class_id = np.argmax(lab[5:])

        # Determining the coordinates of the bounding box in the image frame
        ax = (lab[1]*y_std[1] + y_mean[1]) * image_size
        ay = (lab[2]*y_std[2] + y_mean[2]) * image_size
        width = (lab[3]*y_std[3] + y_mean[3]) * image_size
        height = (lab[4]*y_std[4] + y_mean[4]) * image_size
        #print("x: {}, y: {}, w: {}, h:{}".format(ax,ay,width, height))
        # Determination of the extrema of the bounding box
        p_x = [ax-width/2, ax+width/2]
        p_y = [ay-height/2, ay+height/2]
        # Display the bounding box in the right color
        plt.plot([p_x[0], p_x[0]],p_y,color=colors[class_id])
        plt.plot([p_x[1], p_x[1]],p_y,color=colors[class_id])
        plt.plot(p_x,[p_y[0],p_y[0]],color=colors[class_id])
        plt.plot(p_x,[p_y[1],p_y[1]],color=colors[class_id])
        plt.title("Prediction: Image {} - {}".format(num_img, classes[class_id]))

    plt.show()
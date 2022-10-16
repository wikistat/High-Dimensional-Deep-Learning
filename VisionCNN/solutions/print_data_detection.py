def print_data_detection(x, y, id=None, image_size=IMAGE_SIZE, mode='gt'):
    if id==None:
        # Random drawing of an image in the database
        num_img = np.random.randint(x.shape[0]) 
        print(num_img)
    else:
        num_img = id

    img = x[num_img]
    lab = y[num_img]

    colors = ["blue", "yellow", "red", "orange"] # Different colors for the different classes
    classes = ["Buffalo", "Elephant", "Rhino", "Zebra"]

    boxes = lab[:, :, 1:5]
    for ind_x in range(CELL_PER_DIM):
        for ind_y in range(CELL_PER_DIM):
            box = boxes[ind_x, ind_y]
            box[0] = box[0] * PIX_PER_CELL + ind_x * PIX_PER_CELL
            box[1] = box[1] * PIX_PER_CELL + ind_y * PIX_PER_CELL
            box[2] = box[2]**2 * IMAGE_SIZE
            box[3] = box[3]**2 * IMAGE_SIZE
            boxes[ind_x, ind_y] = box

    # Retrieve all information from bounding boxes
    all_presences = np.reshape(lab[:, :, 0], (CELL_PER_DIM*CELL_PER_DIM))
    all_boxes = np.reshape(lab[:, :, 1:5], (-1, 4))
    all_classes = np.reshape(lab[:, :, 5:9], (-1, 4))

    if mode=='pred':
        all_presences = 1 / (1 + np.exp(-all_presences))
        all_classes = softmax(all_classes, axis=1)

    indices_sorted = np.argsort(-all_presences)

    # Eliminate all bounding boxes whose probability of presence is < 0.5 
    seuil = 0.35
    all_boxes = all_boxes[np.where(all_presences > threshold)]
    all_classes = all_classes[np.where(all_presences > threshold)]
    all_presences = all_presences[np.where(all_presences > threshold)]


    # Image display
    plt.imshow(img)
    for i in range(all_boxes.shape[0]):

        # Determination of the class
        class_id = np.argmax(all_classes[i])
        lab = all_boxes[i]
        #print("x: {}, y: {}, w: {}, h:{}".format(ax,ay,width, height))
        # Determination of the extrema of the bounding box
        p_x = [lab[0]-lab[2]/2, lab[0]+lab[2]/2]
        p_y = [lab[1]-lab[3]/2, lab[1]+lab[3]/2]
        # Display the bounding box in the right color
        plt.plot([p_x[0], p_x[0]],p_y,color=colors[class_id])
        plt.plot([p_x[1], p_x[1]],p_y,color=colors[class_id])
        plt.plot(p_x,[p_y[0],p_y[0]],color=colors[class_id])
        plt.plot(p_x,[p_y[1],p_y[1]],color=colors[class_id], label=classes[class_id] + " " +  str(all_presences[i]))
  
    plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
    plt.show()  
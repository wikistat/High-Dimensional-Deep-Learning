def load_data_detection():
    # Path to the database
    ds_path = "./wildlife/"
    # Paths to the data of the 4 different classes
    paths = [ds_path+"buffalo/", ds_path+"elephant/", ds_path+"rhino/", ds_path+"zebra/"]
    # Index for adding data to the x and y variables 
    i = 0
    # Preparation of data structures for x and y
    x = np.zeros((DATASET_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3))
    y = np.zeros((DATASET_SIZE, CELL_PER_DIM, CELL_PER_DIM, NB_CLASSES + 5*BOX_PER_CELL))

    # Save normalized bounding box width/height
    widths = []
    heights = []

    # Browse paths of each class
    for path in paths:

        # Browse the (sorted) files in the directory
        dirs = os.listdir(path)
        dirs.sort()

        for item in dirs:
            if os.path.isfile(path + item):
                # Extracting the file extension 
                extension = item.split(".")[1]

                if extension=="jpg" or extension=="JPG":
                    # Image : we will fill the variable x
                    # Reading the image
                    img = Image.open(path + item)
                    # Image scaling
                    img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.ANTIALIAS)
                    # Filling the variable x
                    x[i] = np.asarray(img, dtype=np.int32)

                elif extension=="txt":
                    # Text file: bounding box coordinates to fill y
                    labels = open(path + item, "r")
                    # Retrieving of lines from the text file
                    labels = labels.read().split('\n')
                    # If the last line is empty, delete it 
                    if labels[-1]=="":
                        del labels[-1]

                    err_flag = 0
                    boxes = []
                    for label in labels:
                        # Retrieving information from the bounding box
                        label = label.split()
                        # Save width/height of bounding boxes
                        widths.append(float(label[3]))
                        heights.append(float(label[4]))
                        # Bounding box center coordinates in the image frame
                        cx, cy = float(label[1]) * IMAGE_SIZE, float(label[2]) * IMAGE_SIZE
                        # Determination of the indices of the cell in which the center falls
                        ind_x, ind_y = int(cx // PIX_PER_CELL), int(cy // PIX_PER_CELL)
                        # YOLO : "The (x, y) coordinates represent the center of the box relative to the bounds of the grid cell."
                        # We will therefore compute the coordinates of the center relative to the cell in which it is located
                        cx_cell = (cx - ind_x * PIX_PER_CELL) / PIX_PER_CELL
                        cy_cell = (cy - ind_y * PIX_PER_CELL) / PIX_PER_CELL
                        # Confidence index of the bounding box
                        presence = np.array([1], dtype="i")
                        # "One-hot vector " to represent the class probabilities
                        classes = np_utils.to_categorical(label[0], num_classes=4)
                        # We arrange the class probabilities at the end of the vector ([ BOX 1 ; BOX 2 ; ... ; BOX N ; CLASSES])
                        y[i, ind_x, ind_y, 5 * BOX_PER_CELL:] = classes

                        boxes.append([cx, cy, label[3]*IMAGE_SIZE, label[4]*IMAGE_SIZE])
                        # Determining the cell bounding box index in which to store the information
                        ind_box = 0
                        while y[i, ind_x, ind_y, 5*ind_box] == 1 and ind_box < BOX_PER_CELL - 1:
                            # If the current index box is already in use (presence = 1)  
                            # and the maximum number of boxes has not been reached, we go to the next box
                            ind_box = ind_box + 1

                        if y[i, ind_x, ind_y, 5*ind_box] == 1:
                            print("ERROR: THE CELL ALREADY CONTAINS ALL THE AVAILABLE BOXES")
                            print(path + item)
                            err_flag = 1
                        else:
                            y[i, ind_x, ind_y, 5*ind_box] = 1
                            y[i, ind_x, ind_y, 5*ind_box + 1] = cx_cell
                            y[i, ind_x, ind_y, 5*ind_box + 2] = cy_cell
                            # Square root of the width and height of the box
                            y[i, ind_x, ind_y, 5*ind_box + 3] = math.sqrt(float(label[3]))
                            y[i, ind_x, ind_y, 5*ind_box + 4] = math.sqrt(float(label[4]))

                    i = i + 1
                    if err_flag == 1:
                        img_name = item.split(".")[0]
                        img = Image.open(path + img_name + '.jpg')
                        # Image scaling
                        img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.ANTIALIAS)

                        plt.imshow(img)
                        for ind_cell in range(CELL_PER_DIM):                
                            plt.plot([ind_cell*PIX_PER_CELL, ind_cell*PIX_PER_CELL], [0, IMAGE_SIZE-1], 'k-')
                            plt.plot([0, IMAGE_SIZE-1], [ind_cell*PIX_PER_CELL, ind_cell*PIX_PER_CELL], 'k-')

                        for ind_box_plot in range(len(boxes)):
                            box = boxes[ind_box_plot]
                            plt.plot(box[0], box[1], 'b.')
                        plt.show()

                else:
                    print("extension found: ", extension)

    return x, y, widths, heights
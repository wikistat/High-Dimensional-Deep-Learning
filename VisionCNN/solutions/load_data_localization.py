def load_data_localization(image_size):
    # Path to the database
    ds_path = "./wildlife/"
    # Paths to the data of the 4 different classes
    paths = [ds_path + "buffalo/", ds_path + "elephant/", ds_path + "rhino/", ds_path + "zebra/"]
    # Index for adding data to the x and y variables 
    i = 0
    # Preparation of data structures for x and y
    x = np.zeros((DATASET_SIZE, image_size, image_size, 3))
    y = np.empty((DATASET_SIZE, 9)) # 9 = 1 + 4 + 4 : presence + bounding box + classes

    # Browse paths of each class
    for path in paths:

        # Browse the (sorted) files in the directory
        dirs = os.listdir(path)
        dirs.sort()

        for item in dirs:
            #print(path+item)
            if os.path.isfile(path + item):
                # Extracting the file extension 
                extension =item.split(".")[1]

            if extension=="jpg" or extension=="JPG":
                # Image : we will fill the variable x
                # Reading the image
                img = Image.open(path + item)
                # Image scaling
                img = img.resize((image_size,image_size), Image.ANTIALIAS)
                # Filling the variable x
                x[i] = np.asarray(img)

            elif extension=="txt":
                # Text file: bounding box coordinates to fill y
                labels = open(path + item, "r")
                # Retrieving of lines from the text file
                labels= labels.read().split('\n')
                # If the last line is empty, delete it 
                if labels[-1]=="":
                    del labels[-1]

                # Maximum area bounding box index
                j_max = 0
                if len(labels) > 1:
                    area_max = 0 # Bounding box area of maximum area
                    # Browse bounding boxes for objects in the image
                    for j in range(len(labels)):
                        # Compute the area of the current bounding box
                        area = float(labels[j].split()[3]) * float(labels[j].split()[4])
                        # Update the maximum area bounding box, if necessary
                        if area > area_max:
                            area_max = aire
                            j_max = j    

                # An object is present on the image (presence = 1)
                presence = np.array([1], dtype="i")
                # "One-hot vector " to represent the class probabilities
                classes = np_utils.to_categorical(labels[j_max].split()[0], num_classes=4)
                # Coordinates of the maximum area bounding box
                coordinates = np.array(labels[j_max].split()[1:], dtype="f")
                # Filling the variable y
                y[i, 0] = presence
                y[i, 1:5] = coordinates
                y[i, 5:] = classes

                #plt.imshow()
                i = i + 1
            else:
                print("extension found: ", extension)

    return x, y
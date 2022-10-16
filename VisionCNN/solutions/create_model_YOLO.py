def create_model_YOLO(input_shape=(64, 64, 3)):
    weight_decay = 0

    input_layer = Input(shape=input_shape)

    conv1 = Conv2D(32, 3, activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(input_layer)
    conv1 = Conv2D(32, 3, activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    conv1 = Conv2D(32, 3, activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, 3, activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(64, 3, activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    conv2 = Conv2D(64, 3, activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, 3, activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(128, 3, activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    conv3 = Conv2D(128, 3, activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    dense4 = Flatten()(pool3)
    dense4 = Dense(512, activation='elu',kernel_regularizer=regularizers.l2(weight_decay))(dense4)
    dense5 = Dense(512, activation='elu',kernel_regularizer=regularizers.l2(weight_decay))(dense4)
    output = Dense(CELL_PER_DIM*CELL_PER_DIM*(NB_CLASSES + 5*BOX_PER_CELL), activation='linear',kernel_regularizer=regularizers.l2(weight_decay))(dense5)
    output = Reshape((CELL_PER_DIM, CELL_PER_DIM, NB_CLASSES + 5*BOX_PER_CELL))(output)

    model = Model(input_layer, output)

    return model
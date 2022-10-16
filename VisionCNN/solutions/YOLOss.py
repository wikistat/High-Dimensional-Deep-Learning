# Definition of the YOLO loss function
def YOLOss(lambda_coord, lambda_noobj, batch_size):

    # "Green" part: subpart concerning the confidence index 
    # and the class probabilities probabilities in the case where a box is present in the cell
    def box_loss(y_true, y_pred):
        return K.sum(K.square(y_true[:,0] - K.sigmoid(y_pred[:,0]))) + K.sum(K.square(y_true[:,5:9] - K.softmax(y_pred[:,5:9])))

    # "Blue" part: subpart concerning the coordinates of the bounding box in the case where a box is present in the cell
    def coord_loss(y_true, y_pred):
        return K.sum(K.square(y_true[:,1:5] - y_pred[:,1:5]))


    # "Red" part: subpart concerning the confidence index in case no box is present in the cell
    def nobox_loss(y_true, y_pred):
        return K.sum(K.square(y_true[:,0] - K.sigmoid(y_pred[:,0])))


    def YOLO_loss(y_true, y_pred):

        # Reshape the tensors from bs x S x S x (5B+C) to (bsxSxS) x (5B+C)
        y_true = K.reshape(y_true, shape=(-1, 9))
        y_pred = K.reshape(y_pred, shape=(-1, 9))

        # Search (in y_true labels) for indices of cells for which at least the first bounding box is present
        not_empty = K.greater_equal(y_true[:, 0], 1)      
        indices = K.arange(0, K.shape(y_true)[0])
        indices_notempty_cells = indices[not_empty]

        empty = K.less_equal(y_true[:, 0], 0)
        indices_empty_cells = indices[empty]

        # Separate the cells of y_true and y_pred with or without bounding box
        y_true_notempty = K.gather(y_true, indices_notempty_cells)
        y_pred_notempty = K.gather(y_pred, indices_notempty_cells)

        y_true_empty = K.gather(y_true, indices_empty_cells)
        y_pred_empty = K.gather(y_pred, indices_empty_cells)

        return (box_loss(y_true_notempty, y_pred_notempty) + lambda_coord*coord_loss(y_true_notempty, y_pred_notempty) + lambda_noobj*nobox_loss(y_true_empty, y_pred_empty))/batch_size

   
    # Return a function
    return YOLO_loss
def compute_iou(y_true, y_pred):
    ### "Denormalization" of bounding box coordinates
    pred_box_xy = y_pred[:, 0:2]* y_std[0:2] + y_mean[0:2]
    true_box_xy = y_true[:, 0:2]* y_std[0:2] + y_mean[0:2]

    ### "Denormalization of the width and height of bounding boxes
    pred_box_wh = y_pred[:, 2:4] * y_std[2:4] + y_mean[2:4]
    true_box_wh = y_true[:, 2:4] * y_std[2:4] + y_mean[2:4]

    # Computation of the minimum and maximum coordinates of the real bounding box
    true_wh_half = true_box_wh / 2.
    true_mins    = true_box_xy - true_wh_half
    true_maxs    = true_box_xy + true_wh_half

    # Computation of the minimum and maximum coordinates of the predicted bounding box
    pred_wh_half = pred_box_wh / 2.
    pred_mins    = pred_box_xy - pred_wh_half
    pred_maxs    = pred_box_xy + pred_wh_half       

    # Determining the intersection of bounding boxes
    intersect_mins  = tf.maximum(pred_mins, true_mins)
    intersect_maxs  = tf.minimum(pred_maxs, true_maxs)
    intersect_wh    = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[:, 0] * intersect_wh[:, 1]

    # Area of predicted and actual bounding boxes
    true_areas = true_box_wh[:, 0] * true_box_wh[:, 1]
    pred_areas = pred_box_wh[:, 0] * pred_box_wh[:, 1]

    # Area of the union of predicted and real boxes
    union_areas = pred_areas + true_areas - intersect_areas

    iou_scores  = tf.truediv(intersect_areas, union_areas)
    return iou_scores
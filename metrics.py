import numpy as np
import tensorflow as tf
from keras import backend as K

def IoU(xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2):
    intersection_width  = np.maximum(0.0, np.minimum(xmax1, xmax2) - np.maximum(xmin1, xmin2))
    intersection_height = np.maximum(0.0, np.minimum(ymax1, ymax2) - np.maximum(ymin1, ymin2))

    intersection = intersection_width * intersection_height
    union = (xmax1 - xmin1)*(ymax1 - ymin1) + (xmax2 - xmin2)*(ymax2 - ymin2) - intersection

    return intersection / union

def get_smooth_l1():
    def smooth_l1(y_true, y_pred):
        x = y_true - y_pred
        #x = K.variable(y_true - y_pred, dtype = 'float32')
        return tf.where(K.abs(x) < 1.0, 0.5*x*x, K.abs(x) - 0.5)
    return smooth_l1

def multi_class_crossentropy(y_true, y_pred):
    return K.sum(- y_true * K.log(y_pred + K.epsilon()), axis = -1)

def get_focal_loss(gamma = 2.0):
    def focal_loss(y_true, y_pred):
        #return K.sum(- y_true * K.pow(1.0 - y_pred, gamma) * K.log(y_pred + K.epsilon()), axis = -1)
        return K.sum(- y_true * K.pow(1.0 - y_pred, gamma) * K.log(y_pred + 1e-12), axis = -1)
    return focal_loss


def get_ppn_loss(gamma = 2.0, alpha = 1.0, background_id = 0):
    def ppn_loss(y_true, y_pred):
        neg_mask = y_true[..., background_id]
        pos_mask = K.sum(y_true[..., :-4], axis = -1) - y_true[..., background_id]

        focal_loss = get_focal_loss(gamma)
        smooth_l1 = get_smooth_l1()

        class_loss = (pos_mask + alpha * neg_mask) * focal_loss(y_true[..., :-4], y_pred[..., :-4])
        loc_loss = pos_mask * K.sum(smooth_l1(y_true[..., -4:], y_pred[..., -4:]), axis = -1)

        return (K.sum(class_loss) + K.sum(loc_loss)) / K.sum(pos_mask + neg_mask)
    return ppn_loss

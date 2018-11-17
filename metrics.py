import numpy as np
import tensorflow as tf
from keras import backend as K
import imgaug as ia

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
        return K.sum(- y_true * K.pow(1.0 - y_pred, gamma) * K.log(K.clip(y_pred, min_value=1e-12, max_value=None)), axis = -1)
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

def AP(batch_ground_truth, batch_predictions, iou_threshold = 0.5):
    all_predictions = []
    total_positives = 0

    for ground_truth, predictions in zip(batch_ground_truth, batch_predictions):
        ground_truth = ground_truth.bounding_boxes
        total_positives += len(ground_truth)

        predictions = predictions.bounding_boxes
        predictions = sorted(predictions, key=lambda x: x.confidence, reverse=True)

        matched = np.zeros(len(ground_truth))

        for pred in predictions:
            iou = [pred.iou(gt) for gt in ground_truth]
            i = np.argmax(iou)

            if iou[i] >= iou_threshold and not matched[i]:
                all_predictions.append((pred.confidence, True))
                matched[i] = True
            else:
                all_predictions.append((pred.confidence, False))

    all_predictions = sorted(all_predictions, reverse=True)

    recalls, precisions = [0], [1]
    TP, FP = 0, 0


    for conf, result in all_predictions:
        if result: TP += 1
        else: FP += 1

        precisions.append(TP / (TP+FP))
        recalls.append(TP / total_positives)

    for i in range(len(precisions)-2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i+1])

    recalls = np.array(recalls)
    precisions = np.array(precisions)

    return np.sum((recalls[1:]-recalls[:-1]) * precisions[1:])

def mAP(ground_truth, predictions, classes, iou_threshold = 0.5):
    APs = []

    for label in classes:
        class_ground_truth = [
            ia.BoundingBoxesOnImage(
                [box for box in boxes.bounding_boxes if box.label == label],
                shape = boxes.shape
            ) for boxes in ground_truth
        ]
        class_predictions = [
            ia.BoundingBoxesOnImage(
                [box for box in boxes.bounding_boxes if box.label == label],
                shape = boxes.shape
            ) for boxes in predictions
        ]

        APs.append(AP(class_ground_truth, class_predictions, iou_threshold))

    return np.mean(APs)

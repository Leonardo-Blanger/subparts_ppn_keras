import numpy as np
import imgaug as ia

from . import utils
from .metrics import IoU

def subparts_output_encoder(ground_truth, network, neg_iou_threshold = 0.3, pos_iou_threshold = 0.5):
    anchor_xmin = network.subparts_anchor_xmin
    anchor_ymin = network.subparts_anchor_ymin
    anchor_xmax = network.subparts_anchor_xmax
    anchor_ymax = network.subparts_anchor_ymax
    num_anchors = anchor_xmin.shape[0]
    batch_output = []

    # For each item in the batch
    for boxes in ground_truth:
        num_gt = len(boxes.bounding_boxes)
        output = np.zeros((num_anchors, network.num_subpart_classes + 4))

        if num_gt == 0:
            output[:, network.subparts_background_id] = 1.0
            batch_output.append(output)
            continue

        ious = []

        for box in boxes.bounding_boxes:
            ious.append(IoU(box.x1, box.y1, box.x2, box.y2,
                            anchor_xmin, anchor_ymin, anchor_xmax, anchor_ymax))
        ious = np.array(ious)

        matches = [[] for _ in range(num_gt)]

        def find_best_match_for_gts(ious):
            ious = ious.copy()

            for _ in range(num_gt):
                best_gt = np.argmax(np.max(ious, axis = 1))
                best_anchor = np.argmax(ious[best_gt, :])

                matches[best_gt].append(best_anchor)
                ious[best_gt, :] = -1.0
                ious[:, best_anchor] = -1.0

        def find_best_match_for_anchors(ious):
            for anchor_index in range(num_anchors):
                if ious[0, anchor_index] < 0.0: continue
                best_gt = np.argmax(ious[:, anchor_index])

                if ious[best_gt, anchor_index] >= pos_iou_threshold:
                    matches[best_gt].append(anchor_index)
                elif ious[best_gt, anchor_index] < neg_iou_threshold:
                    output[anchor_index, network.subparts_background_id] = 1.0

        find_best_match_for_gts(ious.copy())
        ious[:, [match[0] for match in matches]] = -1.0
        find_best_match_for_anchors(ious)

        for i, box in enumerate(boxes.bounding_boxes):
            if box.area < 1.0: continue

            box_cx = (box.x1 + box.x2) * 0.5
            box_cy = (box.y1 + box.y2) * 0.5
            box_w = box.x2 - box.x1
            box_h = box.y2 - box.y1

            anchor_cx = network.subparts_anchor_cx[matches[i]]
            anchor_cy = network.subparts_anchor_cy[matches[i]]
            anchor_w  = network.subparts_anchor_width[matches[i]]
            anchor_h  = network.subparts_anchor_height[matches[i]]

            output[matches[i], network.subparts_class_labels.index(box.label)] = 1.0
            output[matches[i], -4] = (box_cx - anchor_cx) / anchor_w
            output[matches[i], -3] = (box_cy - anchor_cy) / anchor_h
            output[matches[i], -2] = np.log(box_w / anchor_w)
            output[matches[i], -1] = np.log(box_h / anchor_h)

        batch_output.append(output)

    return np.array(batch_output)


def subparts_output_decoder(batch_output, network, conf_threshold = 0.5, nms_threshold = 0.5):
    predicted_boxes = []

    for output in batch_output:
        predictions = np.where(np.logical_and(
            np.argmax(output[:,:-4], axis=1) != network.subparts_background_id, np.max(output[:,:-4], axis=1) >= conf_threshold
        ))[0]

        class_id = np.argmax(output[predictions, :-4], axis=1)
        conf = np.max(output[predictions, :-4], axis=1)

        anchor_cx = network.subparts_anchor_cx[predictions]
        anchor_cy = network.subparts_anchor_cy[predictions]
        anchor_w  = network.subparts_anchor_width[predictions]
        anchor_h  = network.subparts_anchor_height[predictions]

        box_cx = output[predictions, -4] * anchor_w + anchor_cx
        box_cy = output[predictions, -3] * anchor_h + anchor_cy
        box_w  = np.exp(output[predictions, -2]) * anchor_w
        box_h  = np.exp(output[predictions, -1]) * anchor_h

        xmin = box_cx - box_w * 0.5
        ymin = box_cy - box_h * 0.5
        xmax = box_cx + box_w * 0.5
        ymax = box_cy + box_h * 0.5

        class_id = np.expand_dims(class_id, axis = -1)
        conf = np.expand_dims(conf, axis = -1)
        xmin = np.expand_dims(xmin, axis = -1)
        ymin = np.expand_dims(ymin, axis = -1)
        xmax = np.expand_dims(xmax, axis = -1)
        ymax = np.expand_dims(ymax, axis = -1)

        boxes = np.concatenate([class_id, conf, xmin, ymin, xmax, ymax], axis = -1)

        if boxes.shape[0] == 0:
            predicted_boxes.append(ia.BoundingBoxesOnImage([], shape = network.input_shape[:2]))
            continue

        # NMS
        nms_boxes = []

        for class_id in range(network.num_subpart_classes):
            if class_id == network.subparts_background_id: continue

            class_predictions = np.array([box for box in boxes if box[0] == class_id])
            if class_predictions.shape[0] == 0: continue

            class_predictions = class_predictions[np.flip(np.argsort(class_predictions[:,1], axis=0), axis=0)]
            nms_class_boxes = np.array([class_predictions[0]])

            for box in class_predictions[1:]:
                xmin1, ymin1, xmax1, ymax1 = box[2:]
                xmin2, ymin2, xmax2, ymax2 = [nms_class_boxes[:,i] for i in range(2,6)]

                if np.all(IoU(xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2) < nms_threshold):
                    nms_class_boxes = np.concatenate([nms_class_boxes, [box]], axis=0)

            [nms_boxes.append(utils.BoundingBox(
                x1=box[2], y1=box[3], x2=box[4], y2=box[5], label=network.subparts_class_labels[int(box[0])], confidence=box[1]
            )) for box in nms_class_boxes]

        predicted_boxes.append(ia.BoundingBoxesOnImage(nms_boxes, shape = network.input_shape[:2]))

    return predicted_boxes

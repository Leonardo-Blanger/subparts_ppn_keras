import numpy as np

from metrics import IoU

def output_encoder(ground_truth, network, neg_iou_threshold = 0.3, pos_iou_threshold = 0.5):
    anchor_xmin = network.anchor_xmin
    anchor_ymin = network.anchor_ymin
    anchor_xmax = network.anchor_xmax
    anchor_ymax = network.anchor_ymax
    num_anchors = anchor_xmin.shape[0]
    batch_output = []

    # For each item in the batch
    for boxes in ground_truth:
        num_gt = boxes.shape[0]
        #if num_gt == 0: continue

        output = np.zeros((num_anchors, network.num_classes + 4))
        ious = []

        for box in boxes:
            _, box_xmin, box_ymin, box_xmax, box_ymax = box
            ious.append(IoU(box_xmin, box_ymin, box_xmax, box_ymax,
                            anchor_xmin, anchor_ymin, anchor_xmax, anchor_ymax))
        ious = np.array(ious)

        matches = [[] for _ in range(num_gt)]

        def find_best_match_for_gts(ious):
            ious = ious.copy()

            for _ in range(num_gt):
                best_gt = np.argmax(np.max(ious, axis = 1))
                best_anchor = np.argmax(ious[best_gt, :])

                matches[best_gt].append(best_anchor)
                ious[best_gt, :] = 0.0
                ious[:, best_anchor] = 0.0

        def find_best_match_for_anchors(ious):
            for anchor_index in range(num_anchors):
                if ious[0, anchor_index] < 0.0: continue
                best_gt = np.argmax(ious[:, anchor_index])

                if ious[best_gt, anchor_index] >= pos_iou_threshold:
                    matches[best_gt].append(anchor_index)
                elif ious[best_gt, anchor_index] < neg_iou_threshold:
                    output[anchor_index, network.background_id] = 1.0

        find_best_match_for_gts(ious.copy())
        ious[:, [match[0] for match in matches]] = -1.0
        find_best_match_for_anchors(ious)

        for i, box in enumerate(boxes):
            class_id, box_xmin, box_ymin, box_xmax, box_ymax = box

            box_cx = (box_xmin + box_xmax) * 0.5
            box_cy = (box_ymin + box_ymax) * 0.5
            box_w = box_xmax - box_xmin
            box_h = box_ymax - box_ymin

            anchor_cx = network.anchor_cx[matches[i]]
            anchor_cy = network.anchor_cy[matches[i]]
            anchor_w  = network.anchor_width[matches[i]]
            anchor_h  = network.anchor_height[matches[i]]

            output[matches[i], int(class_id)] = 1.0
            output[matches[i], -4] = (box_cx - anchor_cx) / anchor_w
            output[matches[i], -3] = (box_cy - anchor_cy) / anchor_h
            output[matches[i], -2] = np.log(box_w / anchor_w)
            output[matches[i], -1] = np.log(box_h / anchor_h)

        batch_output.append(output)

    return np.array(batch_output)


def output_decoder(batch_output, network, conf_threshold = 0.5):
    predicted_boxes = []

    for output in batch_output:
        predictions = np.where(np.logical_and(
                np.argmax(output[:,:-4], axis=1) != network.background_id, np.max(output[:,:-4], axis=1) >= conf_threshold
        ))[0]

        class_id = np.argmax(output[predictions, :-4], axis=1)
        conf = np.max(output[predictions, :-4], axis=1)

        anchor_cx = network.anchor_cx[predictions]
        anchor_cy = network.anchor_cy[predictions]
        anchor_w  = network.anchor_width[predictions]
        anchor_h  = network.anchor_height[predictions]

        box_cx = output[predictions, -4] * anchor_w + anchor_cx
        box_cy = output[predictions, -3] * anchor_h + anchor_cy
        box_w  = np.exp(output[predictions, -2]) * anchor_w
        box_h  = np.exp(output[predictions, -1]) * anchor_h

        xmin = box_cx - box_w * 0.5
        ymin = box_cy - box_h * 0.5
        xmax = box_cx + box_w * 0.5
        ymax = box_cy + box_h * 0.5

        class_id = np.expand_dims(class_id, axis = -1)
        xmin = np.expand_dims(xmin, axis = -1)
        ymin = np.expand_dims(ymin, axis = -1)
        xmax = np.expand_dims(xmax, axis = -1)
        ymax = np.expand_dims(ymax, axis = -1)

        box = np.concatenate([class_id, conf, xmin, ymin, xmax, ymax], axis = -1)
        predicted_boxes.append(box)

    return predicted_boxes

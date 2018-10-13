import pandas as pd
import numpy as np
import cv2 as cv
import os
import sys
from tqdm import tqdm
import pickle
import imgaug as ia
from keras.preprocessing.image import load_img, img_to_array

from encode_decode_output import output_encoder
from augmentation import augment

def batch_generator(network,
                    dataset = None,
                    pickled_dataset = None,
                    batch_size = 32,
                    shuffle = False,
                    encode_output = False,
                    augmentation = None,
                    channels = 'RGB'):

    if not pickled_dataset is None and os.path.exists(pickled_dataset):
        with open(pickled_dataset, 'rb') as f:
            images, objects = pickle.load(f)

        if images.shape[1:] != network.input_shape:
            raise Exception('The shape of the images in %s is not compatible with the network')
    else:
        if dataset is None:
            raise Exception('At least one of dataset or pickled_dataset must be provided')

        if isinstance(dataset, str):
            dataset = pd.read_csv(dataset)

        input_height, input_width = network.input_shape[:2]
        images = {}
        objects = {}

        for i in tqdm(range(dataset.shape[0]), desc='Preprocessing Dataset', file=sys.stdout):
            entry = dataset.loc[i]
            filepath = entry['filepath']
            image_height = entry['image_height']
            image_width = entry['image_width']

            img_id = ''.join(filepath.split('/')[-1].split('.')[:-1])

            if not img_id in images:
                #img = img_to_array(load_img(filepath, target_size=(input_height, input_width)))
                img = cv.resize(cv.imread(filepath), (input_width, input_height))
                images[img_id] = img
                objects[img_id] = []
                del img

            obj_class = network.class_labels.index(entry['class'])
            xmin = entry['xmin'] * float(input_width) / image_width
            ymin = entry['ymin'] * float(input_height) / image_height
            xmax = entry['xmax'] * float(input_width) / image_width
            ymax = entry['ymax'] * float(input_height) / image_height

            objects[img_id].append([obj_class, xmin, ymin, xmax, ymax])

        images = np.array(list(images.values()))
        for img_id in objects:
            objects[img_id] = np.array(objects[img_id])
        objects = list(objects.values())

        if not pickled_dataset is None:
            with open(pickled_dataset, 'wb') as f:
                pickle.dump((images, objects), f)

    channels = channels.lower()
    if channels == 'bgr':
        pass
    elif channels == 'rgb':
        images = images[..., [2,1,0]]
    else:
        raise Exception('Channel format not supported: %s' % channels)

    def generator(images, objects):
        batch_start = 0

        while True:
            if batch_start >= len(images):
                if shuffle:
                    perm = np.random.permutation(len(images))
                    images = images[perm]
                    objects = [objects[i] for i in perm]
                batch_start = 0

            batch_X = images[batch_start : batch_start+batch_size]
            batch_y = objects[batch_start : batch_start+batch_size]
            batch_start += batch_size

            if augmentation:
                batch_X, batch_y = augment(batch_X, batch_y, network, augmentation)

            if encode_output:
                batch_y = output_encoder(batch_y, network)

            yield batch_X, batch_y

    return generator(images, objects), len(images)


def augment(images, boxes, network, augmentation_seq):  
    BBs = [
        ia.BoundingBoxesOnImage([
            ia.BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, label=label)
            for (label, x1, y1, x2, y2) in img_boxes
        ], shape = network.input_shape)
        for img_boxes in boxes
    ]

    for _ in range(5):
        try:
            seq_det = augmentation_seq.to_deterministic()
            BBs = seq_det.augment_bounding_boxes(BBs)
            images = seq_det.augment_images(np.copy(images))
            break
        except:
            continue
        
    BBs = [imgBBs.remove_out_of_image().cut_out_of_image() for imgBBs in BBs]

    boxes = [
        np.array([
            [bb.label, bb.x1, bb.y1, bb.x2, bb.y2]
            for bb in imgBBs.bounding_boxes
        ]) for imgBBs in BBs
    ]
    
    return images, boxes

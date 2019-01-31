import pandas as pd
import numpy as np
import cv2 as cv
import os
import sys
from tqdm import tqdm
import pickle
import imgaug as ia
from imgaug import BoundingBox, BoundingBoxesOnImage
from glob import glob
from keras.preprocessing.image import load_img, img_to_array

from . import utils
from .encode_decode_output import output_encoder
from .encode_decode_subparts_output import subparts_output_encoder

class SubPartsBatchGenerator:
    def __init__(self,
                 network,
                 dataset = None,
                 subparts_dataset = None,
                 images_dir = None,
                 pickled_dataset = None,
                 channels = 'RGB'):

        self.network = network
        self.images = []
        self.objects = []
        self.subparts = []

        if not dataset is None or not pickled_dataset is None:
            self.add_data(dataset, subparts_dataset, images_dir, pickled_dataset)

    def add_data(self,
                 dataset = None,
                 subparts_dataset = None,
                 images_dir = None,
                 pickled_dataset = None,
                 channels = 'RGB'):

        if not pickled_dataset is None and os.path.exists(pickled_dataset):
            with open(pickled_dataset, 'rb') as f:
                images, objects, subparts = pickle.load(f)

            if images.shape[1:] != self.network.input_shape:
                raise Exception('The shape of the images in %s is '+
                                'not compatible with the network')
        else:
            if dataset is None:
                raise Exception('At least one of dataset or pickled_dataset must be provided')

            if isinstance(dataset, str):
                dataset = pd.read_csv(dataset, dtype={
                                                'image_id': str,
                                                'object_id': str})

            if isinstance(subparts_dataset, str):
                subparts_dataset = pd.read_csv(subparts_dataset, dtype={
                                                                    'image_id': str,
                                                                    'object_id': str})

            input_height, input_width = self.network.input_shape[:2]
            images = {}
            objects = {}
            subparts = {}

            for i in tqdm(range(dataset.shape[0]), desc='Preprocessing Dataset'):
                entry = dataset.loc[i]
                img_id = str(entry['image_id'])
                obj_id = str(entry['object_id'])

                filepath = glob(os.path.join(images_dir, img_id + '*'))[0]
                image_height = entry['image_height']
                image_width = entry['image_width']

                if not img_id in images:
                    #img = img_to_array(load_img(filepath, target_size=(input_height, input_width)))
                    img = cv.resize(cv.imread(filepath), (input_width, input_height))
                    images[img_id] = img
                    objects[img_id] = []
                    subparts[img_id] = []
                    del img

                obj_class = self.network.class_labels.index(entry['class'])
                xmin = entry['xmin'] * float(input_width) / image_width
                ymin = entry['ymin'] * float(input_height) / image_height
                xmax = entry['xmax'] * float(input_width) / image_width
                ymax = entry['ymax'] * float(input_height) / image_height
                objects[img_id].append([obj_class, xmin, ymin, xmax, ymax])

                obj_subparts = subparts_dataset.loc[subparts_dataset['object_id'] == obj_id].reset_index()

                for j in range(len(obj_subparts)):
                    subpart = obj_subparts.loc[j]

                    subpart_class = self.network.subparts_class_labels.index(subpart['class'])
                    xmin = subpart['xmin'] * float(input_width) / image_width
                    ymin = subpart['ymin'] * float(input_height) / image_height
                    xmax = subpart['xmax'] * float(input_width) / image_width
                    ymax = subpart['ymax'] * float(input_height) / image_height
                    subparts[img_id].append([subpart_class, xmin, ymin, xmax, ymax])

            images = np.array(list(images.values()))

            for img_id in objects:
                objects[img_id] = np.array(objects[img_id])
            objects = list(objects.values())

            for img_id in subparts:
                subparts[img_id] = np.array(subparts[img_id])
            subparts = list(subparts.values())

            channels = channels.lower()
            if channels == 'bgr':
                pass
            elif channels == 'rgb':
                images = images[..., [2,1,0]]
            else:
                raise Exception('Channel format not supported: %s' % channels)

            if not pickled_dataset is None:
                with open(pickled_dataset, 'wb') as f:
                    pickle.dump((images, objects, subparts), f)

        if len(self.images) == 0:
            self.images = images
        else:
            self.images = np.concatenate([self.images, images], axis = 0)
        self.objects += objects
        self.subparts += subparts

    def get_generator(self, batch_size = 32,
                      shuffle = False,
                      encode_output = False,
                      augmentation = None):

        def generator(images, objects, subparts):
            batch_start = 0

            if shuffle:
                perm = np.random.permutation(len(images))
                images = images[perm]
                objects = [objects[i] for i in perm]
                subparts = [subparts[i] for i in perm]

            while True:
                if batch_start + batch_size > len(images):
                    if shuffle:
                        perm = np.random.permutation(len(images))
                        images = images[perm]
                        objects = [objects[i] for i in perm]
                        subparts = [subparts[i] for i in perm]
                    batch_start = 0

                batch_X = images[batch_start : batch_start+batch_size]
                batch_y_objects = [
                    ia.BoundingBoxesOnImage([
                        utils.BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, label=self.network.class_labels[int(label)])
                        for (label, x1, y1, x2, y2) in img_boxes
                    ], shape = self.network.input_shape)
                    for img_boxes in objects[batch_start : batch_start+batch_size]
                ]
                batch_y_subparts = [
                    ia.BoundingBoxesOnImage([
                        utils.BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, label=self.network.subparts_class_labels[int(label)])
                        for (label, x1, y1, x2, y2) in img_boxes
                    ], shape = self.network.input_shape)
                    for img_boxes in subparts[batch_start : batch_start+batch_size]
                ]
                batch_start += batch_size

                if augmentation:
                    batch_X, batch_y_objects, batch_y_subparts = self.augment(
                        batch_X, batch_y_objects, batch_y_subparts, augmentation)

                if encode_output:
                    batch_y_objects = output_encoder(batch_y_objects, self.network)
                    batch_y_subparts = subparts_output_encoder(batch_y_subparts, self.network)

                batch_y = [batch_y_subparts, batch_y_objects]

                yield batch_X, batch_y

        return generator(self.images, self.objects, self.subparts), len(self.images)


    def augment(self, images, object_boxes, subpart_boxes, augmentation_seq, max_tries = 1):
        for _ in range(max_tries):
            try:
                seq_det = augmentation_seq.to_deterministic()
                _object_boxes = seq_det.augment_bounding_boxes(object_boxes)
                _subpart_boxes = seq_det.augment_bounding_boxes(subpart_boxes)
                _images = seq_det.augment_images(np.copy(images))
                object_boxes = _object_boxes
                subpart_boxes = _subpart_boxes
                images = _images
                break
            except:
                continue

        object_boxes = [img_boxes.remove_out_of_image().cut_out_of_image() for img_boxes in object_boxes]
        subpart_boxes = [img_boxes.remove_out_of_image().cut_out_of_image() for img_boxes in subpart_boxes]

        return images, object_boxes, subpart_boxes

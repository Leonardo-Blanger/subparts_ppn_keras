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

class BatchGenerator:
    def __init__(self,
                 network,
                 dataset = None,
                 images_dir = None,
                 pickled_dataset = None,
                 channels = 'RGB'):

         self.network = network
         self.images = []
         self.objects = []

         if not dataset is None or not pickled_dataset is None:
             self.add_data(dataset, images_dir, pickled_dataset)


    def add_data(self,
                 dataset = None,
                 images_dir = None,
                 pickled_dataset = None,
                 channels = 'RGB'):

        if not pickled_dataset is None and os.path.exists(pickled_dataset):
            with open(pickled_dataset, 'rb') as f:
                images, objects = pickle.load(f)

            if images.shape[1:] != self.network.input_shape:
                raise Exception('The shape of the images in %s is '+
                                'not compatible with the network')
        else:
            if dataset is None:
                raise Exception('At least one of dataset or pickled_dataset must be provided')

            if isinstance(dataset, str):
                dataset = pd.read_csv(dataset, dtype={'image_id': str})

            input_height, input_width = self.network.input_shape[:2]
            images = {}
            objects = {}

            for i in tqdm(range(dataset.shape[0]), desc='Preprocessing Dataset'):
                entry = dataset.loc[i]
                img_id = str(entry['image_id'])
                filepath = glob(os.path.join(images_dir, img_id + '*'))[0]
                image_height = entry['image_height']
                image_width = entry['image_width']

                if not img_id in images:
                    #img = img_to_array(load_img(filepath, target_size=(input_height, input_width)))
                    img = cv.resize(cv.imread(filepath), (input_width, input_height))
                    images[img_id] = img
                    objects[img_id] = []
                    del img

                obj_class = self.network.class_labels.index(entry['class'])
                xmin = entry['xmin'] * float(input_width) / image_width
                ymin = entry['ymin'] * float(input_height) / image_height
                xmax = entry['xmax'] * float(input_width) / image_width
                ymax = entry['ymax'] * float(input_height) / image_height

                objects[img_id].append([obj_class, xmin, ymin, xmax, ymax])

            images = np.array(list(images.values()))
            for img_id in objects:
                objects[img_id] = np.array(objects[img_id])
            objects = list(objects.values())

            channels = channels.lower()
            if channels == 'bgr':
                pass
            elif channels == 'rgb':
                images = images[..., [2,1,0]]
            else:
                raise Exception('Channel format not supported: %s' % channels)

            if not pickled_dataset is None:
                with open(pickled_dataset, 'wb') as f:
                    pickle.dump((images, objects), f)

        if len(self.images) == 0:
            self.images = images
        else:
            self.images = np.concatenate([self.images, images], axis = 0)
        self.objects += objects

    def get_generator(self, batch_size = 32,
                      shuffle = False,
                      encode_output = False,
                      augmentation = None):

        def generator(images, objects):
            batch_start = 0

            if shuffle:
                perm = np.random.permutation(len(images))
                images = images[perm]
                objects = [objects[i] for i in perm]

            while True:
                if batch_start + batch_size > len(images):
                    if shuffle:
                        perm = np.random.permutation(len(images))
                        images = images[perm]
                        objects = [objects[i] for i in perm]
                    batch_start = 0

                batch_X = images[batch_start : batch_start + batch_size]
                batch_y = [
                    ia.BoundingBoxesOnImage([
                        utils.BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, label=self.network.class_labels[int(label)])
                        for (label, x1, y1, x2, y2) in img_boxes
                    ], shape = self.network.input_shape)
                    for img_boxes in objects[batch_start : batch_start + batch_size]
                ]
                batch_start += batch_size

                if augmentation:
                    batch_X, batch_y = self.augment(batch_X, batch_y, augmentation)

                if encode_output:
                    batch_y = output_encoder(batch_y, self.network)

                yield batch_X, batch_y

        return generator(self.images, self.objects), len(self.images)


    def augment(self, images, boxes, augmentation_seq, max_tries = 1):
        for _ in range(max_tries):
            try:
                seq_det = augmentation_seq.to_deterministic()
                _boxes = seq_det.augment_bounding_boxes(boxes)
                _images = seq_det.augment_images(np.copy(images))
                boxes = _boxes
                images = _images
                break
            except:
                continue

        return images, boxes

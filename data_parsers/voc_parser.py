import os
import sys
from tqdm import tqdm
import pandas as pd
from bs4 import BeautifulSoup

def voc_parser(ids, images_dir, annotations_dir, output_file = None):
    if isinstance(ids, str):
        with open(ids, 'r') as f:
            ids = [line.strip() for line in f]

    data = {
        'filepath': [],
        'image_height': [],
        'image_width': [],
        'class': [],
        'xmin': [],
        'ymin': [],
        'xmax': [],
        'ymax': []
    }

    for id in tqdm(ids, desc='Parsing dataset', file=sys.stdout):
        annotation_file = os.path.join(annotations_dir, id + '.xml')

        if not os.path.exists(annotation_file):
            raise Exception('Annotation file for id %s not found' % id)

        with open(annotation_file, 'r') as f:
            annotation = BeautifulSoup(f, 'xml')

        filepath = os.path.join(images_dir, annotation.filename.text)

        if not os.path.exists(filepath):
            raise Exception('Image file for id %s not found' % id)

        image_height = int(annotation.size.height.text)
        image_width = int(annotation.size.width.text)

        for obj in annotation.find_all('object'):
            class_label = obj.find('name').text
            xmin = float(obj.bndbox.xmin.text)
            ymin = float(obj.bndbox.ymin.text)
            xmax = float(obj.bndbox.xmax.text)
            ymax = float(obj.bndbox.ymax.text)

            data['filepath'].append(filepath)
            data['image_height'].append(image_height)
            data['image_width'].append(image_width)
            data['class'].append(class_label)
            data['xmin'].append(xmin)
            data['ymin'].append(ymin)
            data['xmax'].append(xmax)
            data['ymax'].append(ymax)

    data = pd.DataFrame(data)

    if not output_file is None:
        data.to_csv(output_file)

    return data

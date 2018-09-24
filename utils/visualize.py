import cv2 as cv
import copy
import os

def maybe_load(image):
    if isinstance(image, str):
        image_file = image
        if not os.path.exists(image_file):
            raise Exception('Image not found: %s' % image_file)
        else:
            image = cv.imread(image_file)
            if image is None:
                raise Exception('Error loading image: %s' % image_file)
    return image


def draw_boxes(image, boxes):
    image = maybe_load(image)
    image = image.copy()

    for box in boxes:
        xmin, ymin, xmax, ymax = [int(x) for x in box]
        cv.rectangle(image, (xmin,ymin), (xmax,ymax), (255,0,0), 2)

    return image

def display_image(image, boxes = None, channels = 'rgb'):
    image = maybe_load(image)
    image = image.copy()

    channels = channels.lower()
    if channels == 'bgr':
        pass
    elif channels == 'rgb':
        image = image[..., [2,1,0]]
    else:
        raise Exception('Channels format not supported: %s' % channels)

    if not boxes is None:
        image = draw_boxes(image, boxes)

    cv.imshow('Image', image)
    cv.waitKey(0)
    cv.destroyAllWindows()

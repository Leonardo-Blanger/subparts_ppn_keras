import pandas as pd
import numpy as np
import cv2
import imgaug as ia
import sys

output = sys.argv[1]

predictions = {}

with open(output, 'r') as f:
    output = [line.strip() for line in f.readlines()]
output = output[1:]

qrs = pd.read_csv('qr_codes_test.csv', dtype = {'image_id': str, 'object_id': str})

matched = set()

TP, FP = 0, 0

i = 0
while i < len(output):
    img_id = output[i].split(' ')[1].split('.')[0]
    print(img_id)

    if i+1 == len(output): break
    line = output[i+1]

    img_codes = qrs.loc[qrs['image_id'] == img_id].reset_index()
    #img_shape = (img_codes.loc[0,'image_height'], img_codes.loc[0,'image_width'])

    for code in line.split(' '):
        if len(code) == 0: continue

        coord = np.array([int(c) for c in code.split('_')])
        fip_centers = coord.reshape((3,2))
        print(fip_centers)

        '''
        bnd_boxes = ia.BoundingBoxesOnImage([
            ia.BoundingBox(
                x1=np.min(fip_centers[:,0]),
                y1=np.min(fip_centers[:,1]),
                x2=np.max(fip_centers[:,0]),
                y2=np.max(fip_centers[:,1]))
            ], shape = img_shape)

        img = cv2.imread('../qr_codes/images/%s.jpg' % img_id)

        cv2.imshow('Image', cv2.resize(img, (500,500)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        img = bnd_boxes.draw_on_image(img, thickness = 8)

        cv2.imshow('Image', cv2.resize(img, (500,500)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        '''

        correct = False

        for j in range(len(img_codes)):
            img_code = img_codes.loc[j]

            if img_code['object_id'] in matched:
                continue

            xmin, ymin, xmax, ymax = img_code[['xmin','ymin','xmax','ymax']]

            correct = True
            for c in fip_centers:
                if xmin <= c[0] <= xmax and ymin <= c[1] <= ymax:
                    pass
                else:
                    correct = False

            if correct:
                matched.add(img_code['object_id'])
                break

        print(correct)
        TP += correct
        FP += 1 - correct

    i += 2
    #break

print('True Positives: %d' % TP)
print('False Positives: %d' % FP)
print('Total Number of Positives: %d' % len(qrs))

import numpy as np
import math

class SSD:
    def __init__(self):
        pass

    def build_anchors(self):
        img_height, img_width = self.input_shape[:2]

        self.anchor_cx = []
        self.anchor_cy = []
        self.anchor_width = []
        self.anchor_height = []

        for i in range(self.num_scales):
            featmap_height, featmap_width = self.feature_map_sizes[i]
            horizontal_stride = float(img_width) / featmap_width
            vertical_stride = float(img_height) / featmap_height

            cols, rows = np.meshgrid(range(featmap_width), range(featmap_width))
            cx = (cols + 0.5) * horizontal_stride
            cy = (rows + 0.5) * vertical_stride
            cx = np.expand_dims(cx, axis = -1)
            cy = np.expand_dims(cy, axis = -1)
            cx = np.repeat(cx, self.boxes_per_cell, axis = -1)
            cy = np.repeat(cy, self.boxes_per_cell, axis = -1)

            width = np.zeros_like(cx)
            height = np.zeros_like(cx)
            for j, ar in enumerate(self.aspect_ratios):
                width[...,j] = img_width * self.scales[i] * math.sqrt(ar)
                height[...,j] = img_height * self.scales[i] / math.sqrt(ar)
            width[...,-1] = img_width * math.sqrt(self.scales[i] * self.scales[i+1])
            height[...,-1] = img_height * math.sqrt(self.scales[i] * self.scales[i+1])

            self.anchor_cx.append(cx.reshape((-1,)))
            self.anchor_cy.append(cy.reshape((-1,)))
            self.anchor_width.append(width.reshape((-1,)))
            self.anchor_height.append(height.reshape((-1,)))

        self.anchor_cx = np.concatenate(self.anchor_cx)
        self.anchor_cy = np.concatenate(self.anchor_cy)
        self.anchor_width = np.concatenate(self.anchor_width)
        self.anchor_height = np.concatenate(self.anchor_height)

        self.anchor_xmin = self.anchor_cx - self.anchor_width * 0.5
        self.anchor_ymin = self.anchor_cy - self.anchor_height * 0.5
        self.anchor_xmax = self.anchor_cx + self.anchor_width * 0.5
        self.anchor_ymax = self.anchor_cy + self.anchor_height * 0.5

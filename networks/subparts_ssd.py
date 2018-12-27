import numpy as np
import math

class SubParts_SSD:
    def __init__(self):
        pass

    def build_anchors(self):
        img_height, img_width = self.input_shape[:2]

        self.anchor_cx = []
        self.anchor_cy = []
        self.anchor_width = []
        self.anchor_height = []

        for i in range(self.num_scales):
            tensor_height, tensor_width = self.tensor_sizes[i]
            horizontal_stride = float(img_width) / tensor_width
            vertical_stride = float(img_height) / tensor_height

            cols, rows = np.meshgrid(range(tensor_width), range(tensor_width))
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


    def build_subpart_anchors(self):
        img_height, img_width = self.input_shape[:2]

        self.subparts_anchor_cx = []
        self.subparts_anchor_cy = []
        self.subparts_anchor_width = []
        self.subparts_anchor_height = []

        for i in range(self.num_subparts_scales):
            tensor_height, tensor_width = self.subparts_tensor_sizes[i]
            horizontal_stride = float(img_width) / tensor_width
            vertical_stride = float(img_height) / tensor_height

            cols, rows = np.meshgrid(range(tensor_width), range(tensor_width))
            cx = (cols + 0.5) * horizontal_stride
            cy = (rows + 0.5) * vertical_stride
            cx = np.expand_dims(cx, axis = -1)
            cy = np.expand_dims(cy, axis = -1)
            cx = np.repeat(cx, self.boxes_per_cell, axis = -1)
            cy = np.repeat(cy, self.boxes_per_cell, axis = -1)

            width = np.zeros_like(cx)
            height = np.zeros_like(cx)
            for j, ar in enumerate(self.subparts_aspect_ratios):
                width[...,j] = img_width * self.subparts_scales[i] * math.sqrt(ar)
                height[...,j] = img_height * self.subparts_scales[i] / math.sqrt(ar)
            width[...,-1] = img_width * math.sqrt(self.subparts_scales[i] * self.scales[i+1])
            height[...,-1] = img_height * math.sqrt(self.subparts_scales[i] * self.scales[i+1])

            self.subparts_anchor_cx.append(cx.reshape((-1,)))
            self.subparts_anchor_cy.append(cy.reshape((-1,)))
            self.subparts_anchor_width.append(width.reshape((-1,)))
            self.subparts_anchor_height.append(height.reshape((-1,)))

        self.subparts_anchor_cx = np.concatenate(self.subparts_anchor_cx)
        self.subparts_anchor_cy = np.concatenate(self.subparts_anchor_cy)
        self.subparts_anchor_width = np.concatenate(self.subparts_anchor_width)
        self.subparts_anchor_height = np.concatenate(self.subparts_anchor_height)

        self.subparts_anchor_xmin = self.subparts_anchor_cx - self.subparts_anchor_width * 0.5
        self.subparts_anchor_ymin = self.subparts_anchor_cy - self.subparts_anchor_height * 0.5
        self.subparts_anchor_xmax = self.subparts_anchor_cx + self.subparts_anchor_width * 0.5
        self.subparts_anchor_ymax = self.subparts_anchor_cy + self.subparts_anchor_height * 0.5

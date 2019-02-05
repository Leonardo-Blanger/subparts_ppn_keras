from keras.applications import vgg16
from keras.layers import Input, Lambda, MaxPooling2D, Conv2D, Reshape, Concatenate, Activation, Add
from keras.models import Model

from .subparts_ssd import SubParts_SSD

class SubParts_SSD_PPN_VGG16(SubParts_SSD):
    def __init__(self,
                class_labels,
                subparts_class_labels,
                input_shape = (300,300,3),
                scales = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05],
                aspect_ratios = [1.0/3.0, 0.5, 1.0, 2.0, 3.0],
                subparts_scales = 0.25,
                subparts_aspect_ratios = [1.0/3.0, 0.5, 1.0, 2.0, 3.0]):

        if isinstance(class_labels, str):
            with open(class_labels, 'r') as f:
                class_labels = [line.strip() for line in f]

        if isinstance(subparts_class_labels, str):
            with open(subparts_class_labels, 'r') as f:
                subparts_class_labels = [line.strip() for line in f]

        if not 'background' in class_labels:
            class_labels = ['background'] + class_labels

        if not 'background' in subparts_class_labels:
            subparts_class_labels = ['background'] + subparts_class_labels

        self.class_labels = class_labels
        self.subparts_class_labels = subparts_class_labels

        self.num_classes = len(class_labels)
        self.num_subpart_classes = len(subparts_class_labels)

        self.background_id = class_labels.index('background')
        self.subparts_background_id = subparts_class_labels.index('background')

        self.input_shape = input_shape

        self.scales = scales
        self.num_scales = len(scales) - 1
        self.aspect_ratios = aspect_ratios
        self.boxes_per_cell = len(aspect_ratios) + 1

        if not isinstance(subparts_scales, list):
            subparts_scales = [s * subparts_scales for s in scales]

        self.subparts_scales = subparts_scales
        self.num_subparts_scales = len(subparts_scales) - 1
        self.subparts_aspect_ratios = subparts_aspect_ratios
        self.subparts_boxes_per_cell = len(subparts_aspect_ratios) + 1

        self.build_model()
        self.build_anchors()
        self.build_subpart_anchors()

    def build_model(self):
        input = Input(shape=self.input_shape)
        preprocessed_input = Lambda(lambda x: vgg16.preprocess_input(x), name='preprocess')(input)

        base = vgg16.VGG16(include_top=False, weights='imagenet', input_tensor=preprocessed_input)
        featmap = base.get_layer('block4_pool').output

        feature_maps = [featmap]

        featmap_height, featmap_width = featmap.shape.as_list()[1:3]
        self.feature_map_sizes = [(featmap_height, featmap_width)]

        while len(feature_maps) < self.num_scales:
            featmap = MaxPooling2D(pool_size = (2, 2), padding = 'same', name='subparts_max_pool_%d' % len(feature_maps))(featmap)
            feature_maps.append(featmap)

            featmap_height, featmap_width = featmap.shape.as_list()[1:3]
            self.feature_map_sizes.append((featmap_height, featmap_width))

        # Sub parts detection parameters

        subparts_shared_conv = Conv2D(filters = 512,
                            kernel_size = (1, 1),
                            strides = (1, 1),
                            activation = 'relu',
                            name = 'subparts_shared_conv')

        subparts_box_classifier = Conv2D(filters = self.subparts_boxes_per_cell * self.num_subpart_classes,
                                kernel_size = (3, 3),
                                strides = (1, 1),
                                padding = 'same',
                                activation = 'linear',
                                name = 'subparts_box_classifier')

        subparts_box_regressor = Conv2D(filters = self.subparts_boxes_per_cell * 4,
                                kernel_size = (3, 3),
                                strides = (1, 1),
                                padding = 'same',
                                activation = 'linear',
                                name = 'subparts_box_regressor')

        subparts_cls_output = []
        subparts_loc_output = []
        subparts_featmaps = []
        self.subparts_tensor_sizes = []

        for featmap, featmap_size in zip(feature_maps, self.feature_map_sizes):
            featmap = subparts_shared_conv(featmap)
            cls = subparts_box_classifier(featmap)
            loc = subparts_box_regressor(featmap)

            featmap_height, featmap_width = featmap_size
            cls = Reshape((featmap_height, featmap_width, self.subparts_boxes_per_cell, self.num_subpart_classes))(cls)
            cls = Activation('softmax')(cls)
            cls = Reshape((featmap_height, featmap_width, self.subparts_boxes_per_cell * self.num_subpart_classes))(cls)

            subparts_featmaps.append(Concatenate(axis=-1)([cls, loc]))

            self.subparts_tensor_sizes.append(featmap_size)
            total_boxes = featmap_size[0] * featmap_size[1] * self.subparts_boxes_per_cell

            cls = Reshape((total_boxes, self.num_subpart_classes))(cls)
            loc = Reshape((total_boxes, 4))(loc)

            subparts_cls_output.append(cls)
            subparts_loc_output.append(loc)

        subparts_cls_output = Concatenate(axis = 1)(subparts_cls_output)
        subparts_loc_output = Concatenate(axis = 1)(subparts_loc_output)
        subparts_output = Concatenate(axis = -1, name='subparts_output')([subparts_cls_output, subparts_loc_output])

        featmap = base.get_layer('block5_conv3').output

        feature_maps = [featmap]

        featmap_height, featmap_width = featmap.shape.as_list()[1:3]
        self.feature_map_sizes = [(featmap_height, featmap_width)]

        while len(feature_maps) < self.num_scales:
            featmap = MaxPooling2D(pool_size = (2, 2), padding = 'same', name='max_pool_%d' % len(feature_maps))(featmap)
            feature_maps.append(featmap)

            featmap_height, featmap_width = featmap.shape.as_list()[1:3]
            self.feature_map_sizes.append((featmap_height, featmap_width))

        # Full object detection parameters

        shared_conv = Conv2D(filters = 512,
                            kernel_size = (1, 1),
                            strides = (1, 1),
                            activation = 'relu',
                            name = 'shared_conv')

        box_classifier = Conv2D(filters = self.boxes_per_cell * self.num_classes,
                                kernel_size = (3, 3),
                                strides = (1, 1),
                                padding = 'same',
                                activation = 'linear',
                                name = 'box_classifier')

        box_regressor = Conv2D(filters = self.boxes_per_cell * 4,
                                kernel_size = (3, 3),
                                strides = (1, 1),
                                padding = 'same',
                                activation = 'linear',
                                name = 'box_regressor')

        cls_output = []
        loc_output = []
        self.tensor_sizes = []

        for featmap, subparts_featmap, featmap_size in zip(feature_maps, subparts_featmaps, self.feature_map_sizes):
            featmap = Concatenate(axis = -1)([shared_conv(featmap), subparts_featmap])
            #featmap = shared_conv(featmap)
            cls = box_classifier(featmap)
            loc = box_regressor(featmap)

            self.tensor_sizes.append(featmap_size)
            total_boxes = featmap_size[0] * featmap_size[1] * self.boxes_per_cell

            cls = Reshape((total_boxes, self.num_classes))(cls)
            cls = Activation('softmax')(cls)
            loc = Reshape((total_boxes, 4))(loc)

            cls_output.append(cls)
            loc_output.append(loc)

        cls_output = Concatenate(axis = 1)(cls_output)
        loc_output = Concatenate(axis = 1)(loc_output)
        output = Concatenate(axis = -1, name='main_output')([cls_output, loc_output])

        self.model = Model(input, [subparts_output, output])
        #self.model.summary()

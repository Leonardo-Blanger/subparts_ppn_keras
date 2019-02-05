from keras.applications import resnet50
from keras.layers import Input, Lambda, MaxPooling2D, Conv2D, Reshape, Concatenate, Activation
from keras.models import Model

from .ssd import SSD

class SSD_PPN_ResNet50(SSD):
    def __init__(self,
                class_labels,
                input_shape = (300,300,3),
                scales = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05],
                aspect_ratios = [1.0/3.0, 0.5, 1.0, 2.0, 3.0]):

        if isinstance(class_labels, str):
            with open(class_labels, 'r') as f:
                class_labels = [line.strip() for line in f]

        if not 'background' in class_labels:
            class_labels = ['background'] + class_labels

        self.class_labels = class_labels
        self.num_classes = len(class_labels)
        self.background_id = class_labels.index('background')

        self.input_shape = input_shape
        self.scales = scales
        self.num_scales = len(scales) - 1
        self.aspect_ratios = aspect_ratios
        self.boxes_per_cell = len(aspect_ratios) + 1

        self.build_model()
        self.build_anchors()

    def build_model(self):
        input = Input(shape=self.input_shape)
        preprocessed_input = Lambda(lambda x: resnet50.preprocess_input(x), name='preprocess')(input)

        base = resnet50.ResNet50(include_top=False, weights='imagenet', input_tensor=preprocessed_input)
        featmap = base.get_layer('activation_40').output

        feature_maps = [featmap]

        featmap_height, featmap_width = featmap.shape.as_list()[1:3]
        self.feature_map_sizes = [(featmap_height, featmap_width)]

        print(featmap)

        while len(feature_maps) < self.num_scales:
            featmap = MaxPooling2D(pool_size = (2, 2), padding = 'same', name='max_pool_%d' % len(feature_maps))(featmap)
            feature_maps.append(featmap)
            print(featmap)
            featmap_height, featmap_width = featmap.shape.as_list()[1:3]
            self.feature_map_sizes.append((featmap_height, featmap_width))

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

        for featmap, featmap_size in zip(feature_maps, self.feature_map_sizes):
            featmap = shared_conv(featmap)
            total_boxes = featmap_size[0] * featmap_size[1] * self.boxes_per_cell

            cls = box_classifier(featmap)
            cls = Reshape((total_boxes, self.num_classes))(cls)
            cls = Activation('softmax')(cls)

            loc = box_regressor(featmap)
            loc = Reshape((total_boxes, 4))(loc)

            cls_output.append(cls)
            loc_output.append(loc)

        cls_output = Concatenate(axis = 1)(cls_output)
        loc_output = Concatenate(axis = 1)(loc_output)
        output = Concatenate(axis = -1)([cls_output, loc_output])

        self.model = Model(input, output)
        #self.model.summary()

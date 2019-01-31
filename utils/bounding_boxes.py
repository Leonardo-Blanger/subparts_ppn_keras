import imgaug as ia
import numpy as np

class BoundingBox(ia.BoundingBox):
    """
    Extend the imgaug.BoundingBox class with the
        addition of a confidence attribute.
    """
    def __init__(self, x1, y1, x2, y2, label = None, confidence = 1.0):
        if x1 > x2:
            x2, x1 = x1, x2
        ia.do_assert(x2 >= x1)
        if y1 > y2:
            y2, y1 = y1, y2
        ia.do_assert(y2 >= y1)

        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.label = label
        self.confidence = confidence

    # Aparently, there is a bug on the iou method on the pip3 version of imgaug.
    # So I copied this implementation from the latest version of code on GitHub.
    def iou(self, other):
        """
        Compute the IoU of this bounding box with another one.
        IoU is the intersection over union, defined as::
            ``area(intersection(A, B)) / area(union(A, B))``
            ``= area(intersection(A, B)) / (area(A) + area(B) - area(intersection(A, B)))``
        Parameters
        ----------
        other : imgaug.BoundingBox
            Other bounding box with which to compare.
        Returns
        -------
        float
            IoU between the two bounding boxes.
        """
        inters = self.intersection(other)
        if inters is None:
            return 0.0
        else:
            area_union = self.area + other.area - inters.area
            return inters.area / area_union if area_union > 0 else 0.0
    '''
    def cut_out_of_image(self, image):
        """
        Cut off all parts of the bounding box that are outside of the image.

        Parameters
        ----------
        image : (H,W,...) ndarray or tuple of int
            Image dimensions to use for the clipping of the bounding box.
            If an ndarray, its shape will be used.
            If a tuple, it is assumed to represent the image shape and must contain at least two integers.

        Returns
        -------
        result : imgaug.BoundingBox
            Bounding box, clipped to fall within the image dimensions.

        """
        if isinstance(image, tuple):
            shape = image
        else:
            shape = image.shape

        height, width = shape[0:2]
        ia.do_assert(height > 0)
        ia.do_assert(width > 0)

        #eps = np.finfo(np.float32).eps
        x1 = np.clip(self.x1, 0, width - 1)
        x2 = np.clip(self.x2, 0, width - 1)
        y1 = np.clip(self.y1, 0, height - 1)
        y2 = np.clip(self.y2, 0, height - 1)

        return self.copy(
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            label=self.label
        )
    '''

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "BoundingBox(x1=%.4f, y1=%.4f, x2=%.4f, y2=%.4f, label=%s, confidence=%.4f)" % (
            self.x1, self.y1, self.x2, self.y2, str(self.label), self.confidence)

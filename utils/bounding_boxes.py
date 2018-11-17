import imgaug as ia

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

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "BoundingBox(x1=%.4f, y1=%.4f, x2=%.4f, y2=%.4f, label=%s, confidence=%.4f)" % (
            self.x1, self.y1, self.x2, self.y2, str(self.label), self.confidence)

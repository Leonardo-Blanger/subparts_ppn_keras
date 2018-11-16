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

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "BoundingBox(x1=%.4f, y1=%.4f, x2=%.4f, y2=%.4f, label=%s, confidence=%.4f)" % (
            self.x1, self.y1, self.x2, self.y2, str(self.label), self.confidence)

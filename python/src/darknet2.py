from ctypes import *
import math
import random
import os
from PIL import Image


def sample(probs):
    s = sum(probs)
    probs = [a / s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs) - 1


def c_array(ctype, values):
    arr = (ctype * len(values))()
    arr[:] = values
    return arr


class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]


class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]


class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]


class Detection():
    """
    ('car', 0.6152912378311157, (572.1994018554688, 120.48184204101562, 214.3546600341797, 98.72494506835938))
    """

    def __init__(self, tuple):
        self.name = tuple[0]
        self.prob = tuple[1]
        self.x = tuple[2][0]
        self.y = tuple[2][1]
        self.width = tuple[2][2]
        self.height = tuple[2][3]

    def __repr__(self):
        return 'Detection(name = {}, prob = {}, x = {}, y = {}, width = {}, height = {})'.format(self.name,
                                                                                                 self.prob,
                                                                                                 self.x,
                                                                                                 self.y,
                                                                                                 self.width,
                                                                                                 self.height)


class Darknet():

    def __init__(self, libdarknet_so, cfg, weights, meta):
        self.lib = CDLL(libdarknet_so, RTLD_GLOBAL)
        self.lib.network_width.argtypes = [c_void_p]
        self.lib.network_width.restype = c_int
        self.lib.network_height.argtypes = [c_void_p]
        self.lib.network_height.restype = c_int

        self.predict = self.lib.network_predict
        self.predict.argtypes = [c_void_p, POINTER(c_float)]
        self.predict.restype = POINTER(c_float)

        self.load_net = self.lib.load_network
        self.load_net.argtypes = [c_char_p, c_char_p, c_int]
        self.load_net.restype = c_void_p

        self.load_meta = self.lib.get_metadata
        self.lib.get_metadata.argtypes = [c_char_p]
        self.lib.get_metadata.restype = METADATA

        self.load_image = self.lib.load_image_color
        self.load_image.argtypes = [c_char_p, c_int, c_int]
        self.load_image.restype = IMAGE

        self.predict_image = self.lib.network_predict_image
        self.predict_image.argtypes = [c_void_p, IMAGE]
        self.predict_image.restype = POINTER(c_float)

        self.get_network_boxes = self.lib.get_network_boxes
        self.get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int,
                                           POINTER(c_int)]
        self.get_network_boxes.restype = POINTER(DETECTION)

        self.do_nms_obj = self.lib.do_nms_obj
        self.do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

        self.free_image = self.lib.free_image
        self.free_image.argtypes = [IMAGE]

        self.free_detections = self.lib.free_detections
        self.free_detections.argtypes = [POINTER(DETECTION), c_int]

        # load weights
        self.net = self.load_net(cfg, weights, 0)

        # load meta
        self.meta = self.load_meta(meta)

    def detect(self, image, thresh=.5, hier_thresh=.5, nms=.45):
        im = self.load_image(image, 0, 0)
        num = c_int(0)
        pnum = pointer(num)
        self.predict_image(self.net, im)
        dets = self.get_network_boxes(self.net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
        num = pnum[0]
        if (nms): self.do_nms_obj(dets, num, self.meta.classes, nms);

        res = []
        for j in range(num):
            for i in range(self.meta.classes):
                if dets[j].prob[i] > 0:
                    b = dets[j].bbox
                    res.append((self.meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
        res = sorted(res, key=lambda x: -x[1])
        self.free_image(im)
        self.free_detections(dets, num)
        return res

    def detect_coco_item(self, image, item, thresh=.5, hier_thresh=.5, nms=.45):
        bounding_boxes = self.detect(image, thresh, hier_thresh, nms)
        detections = [Detection(bounding_box) for bounding_box in bounding_boxes if bounding_box[0] == item]
        return detections


    def crop(self, image, detection):
        img = Image.open(image)
        left = detection.x
        upper = detection.y
        right = detection.x + detection.width
        lower = detection.y + detection.height

        cropped = img.crop((left, upper, right, lower))
        return cropped

if __name__ == "__main__":
    HERE = os.path.dirname(os.path.realpath(__file__))

    libdarknet_so = os.path.join(HERE, "../../libdarknet.so")
    yolov3_tiny_cfg = os.path.join(HERE, "../../cfg/yolov3-tiny.cfg")
    yolov3_tiny_weights = os.path.join(HERE, "../../cfg/yolov3-tiny.weights")
    coco_data = os.path.join(HERE, "../../cfg/coco.data")

    drknet = Darknet(libdarknet_so=libdarknet_so, cfg=yolov3_tiny_cfg, weights=yolov3_tiny_weights, meta=coco_data)

    dog = os.path.join(HERE, "../../data/dog.jpg")
    dd = drknet.detect(dog)
    print dd

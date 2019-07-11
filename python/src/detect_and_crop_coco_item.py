from darknet import *
from PIL import Image


def crop_image(image_path, (x, y), width, height):
    img = Image.open(image_path)

    left = x
    upper = y
    right = x + width
    lower = y + height

    cropped = img.crop((left, upper, right, lower))
    cropped.show()


def detect_objects(image_path):
    yolov3_tiny_cfg = os.path.join(HERE, "../cfg/yolov3-tiny.cfg")
    yolov3_tiny_weights = os.path.join(HERE, "../cfg/yolov3-tiny.weights")
    yolov3_cfg = os.path.join(HERE, "../cfg/yolov3.cfg")
    yolov3_weights = os.path.join(HERE, "../cfg/yolov3.weights")
    coco_data = os.path.join(HERE, "../cfg/coco.data")

    net = load_net(yolov3_cfg, yolov3_weights, 0)
    meta = load_meta(coco_data)
    r = detect(net, meta, image_path)

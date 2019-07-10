from darknet import *

yolov3_tiny_cfg = os.path.join(HERE, "../cfg/yolov3-tiny.cfg")
yolov3_tiny_weights = os.path.join(HERE, "../cfg/yolov3-tiny.weights")
yolov3_cfg = os.path.join(HERE, "../cfg/yolov3.cfg")
yolov3_weights = os.path.join(HERE, "../cfg/yolov3.weights")
coco_data = os.path.join(HERE, "../cfg/coco.data")


net = load_net(yolov3_cfg, yolov3_weights, 0)
meta = load_meta(coco_data)

def detect_car(image_path):
    r = detect(net, meta, image_path)


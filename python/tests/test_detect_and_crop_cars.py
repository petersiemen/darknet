from context import *

HERE = os.path.dirname(os.path.realpath(__file__))


def test_crop_image():
    dog = os.path.join(HERE, "./resources/dog.jpg")
    cropped = crop_image(dog, (100, 200), 200, 400)

    cropped.show()


def test_darknet_tiny():
    HERE = os.path.dirname(os.path.realpath(__file__))

    libdarknet_so = os.path.join(HERE, "../../libdarknet.so")
    yolov3_tiny_cfg = os.path.join(HERE, "../../cfg/yolov3-tiny.cfg")
    yolov3_tiny_weights = os.path.join(HERE, "../../cfg/yolov3-tiny.weights")
    coco_data = os.path.join(HERE, "../../cfg/coco.data")

    drknet = Darknet(libdarknet_so=libdarknet_so, cfg=yolov3_tiny_cfg, weights=yolov3_tiny_weights, meta=coco_data)

    dog = os.path.join(HERE, "../../data/dog.jpg")
    r = drknet.detect(dog)
    print(r)


def test_darknet_tiny_car():
    HERE = os.path.dirname(os.path.realpath(__file__))

    libdarknet_so = os.path.join(HERE, "../../libdarknet.so")
    yolov3_tiny_cfg = os.path.join(HERE, "../../cfg/yolov3-tiny.cfg")
    yolov3_tiny_weights = os.path.join(HERE, "../../cfg/yolov3-tiny.weights")
    coco_data = os.path.join(HERE, "../../cfg/coco.data")

    drknet = Darknet(libdarknet_so=libdarknet_so, cfg=yolov3_tiny_cfg, weights=yolov3_tiny_weights, meta=coco_data)

    dog = os.path.join(HERE, "../../data/dog.jpg")
    cars = drknet.detect_coco_item(dog, 'car')

    for car in cars:
        cropped = drknet.crop(dog, car)
        cropped.show()

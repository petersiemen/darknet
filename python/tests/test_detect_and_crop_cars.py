from context import *

HERE = os.path.dirname(os.path.realpath(__file__))


def test_detect_car():
    dog = os.path.join(HERE, "./resources/dog.jpg")
    crop_image(dog, (100, 200), 200, 400)
    assert True

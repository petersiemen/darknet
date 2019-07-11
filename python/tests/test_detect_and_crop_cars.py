from context import *

HERE = os.path.dirname(os.path.realpath(__file__))


def test_crop_image():
    dog = os.path.join(HERE, "./resources/dog.jpg")
    cropped = crop_image(dog, (100, 200), 200, 400)

    cropped.show()

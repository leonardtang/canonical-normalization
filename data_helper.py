import cv2
import glob

IMAGENET_VAL = "~/imagenette/val/"
TEST_IMG = "n01440764/ILSVRC2012_val_00030740.JPEG"

def grayscale_images(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray

def load_data(img_dir=IMAGENET_VAL, single_image=True):
    images = []
    for file in glob.glob(img_dir + "*/*.JPEG"):
        print(f"Loading image from {file}...")
        images.append(grayscale_images(cv2.imread(file)))
        # Just to test on a single image
        if single_image:
            break

    return images


def load_light_data(file_path=IMAGENET_VAL + TEST_IMG):
    base_image = grayscale_images(cv2.imread(file_path))
    brightened = cv2.add(base_image, 30)
    darkened = cv2.add(base_image, -10)
    return base_image, brightened, darkened

if __name__ == "__main__":
    pass
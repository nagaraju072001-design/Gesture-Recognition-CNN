import random
from pathlib import Path

import cv2

from config import DATASET_DIR

# -----------------------------
# Configuration
# -----------------------------

GESTURES = [
    "Palm",
    "Fist",
    "Peace",
    "Thumb_Up",
    "OK",
]

# Number of augmented images per original image
AUGMENTATIONS_PER_IMAGE = 4


def rotate(img):
    angle = random.uniform(-15, 15)
    h, w = img.shape[:2]
    matrix = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(img, matrix, (w, h))


def brightness(img):
    alpha = random.uniform(0.8, 1.2)
    beta = random.randint(-25, 25)
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


def flip(img):
    return cv2.flip(img, 1)


def blur(img):
    return cv2.GaussianBlur(img, (3, 3), 0)


def noise(img):
    n = random.randint(5, 15)
    gaussian = random.normalvariate(0, n)

    noise_img = img.copy().astype("float32")
    noise_img += gaussian

    return noise_img.clip(0, 255).astype("uint8")


augmentations = [
    rotate,
    brightness,
    flip,
    blur,
    noise,
]

# -----------------------------
# Start Augmentation
# -----------------------------

print("\nStarting Data Augmentation...\n")

total = 0

for gesture in GESTURES:

    folder = DATASET_DIR / gesture

    images = list(folder.glob("*.jpg"))

    print(f"{gesture}: {len(images)} original images")

    for image_path in images:

        image = cv2.imread(str(image_path))

        if image is None:
            continue

        for i in range(AUGMENTATIONS_PER_IMAGE):

            aug = random.choice(augmentations)

            aug_img = aug(image)

            output = folder / f"{image_path.stem}_aug_{i}.jpg"

            cv2.imwrite(str(output), aug_img)

            total += 1

print("\n==============================")
print("Augmentation Complete")
print("==============================")
print("New Images :", total)

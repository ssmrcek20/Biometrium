import os
import shutil
import cv2
import numpy as np

def sort_images(fingerprints_source, training_fingerprints, test_fingerprints):
    images = []
    for img in os.listdir(fingerprints_source):
        if img.endswith(('.tif')):
            images.append(img)

    for i, img in enumerate(images, start=1):
        img_source = os.path.join(fingerprints_source, img)
        if i % 8 == 0:
            shutil.copy2(img_source, test_fingerprints)
        else:
            processed_image = process_image(img_source)
            cv2.imwrite(os.path.join(training_fingerprints, img), processed_image)

def process_image(image_path):
    gray_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    thresholded_image = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return thresholded_image

def main():
    fingerprints_source = './DB2_B'
    training_fingerprints = './TestDataset'
    test_fingerprints = './TrainingDataset'
    if os.listdir(training_fingerprints) or os.listdir(test_fingerprints):
        print("Data already sorted.")
    else:
        sort_images(fingerprints_source, training_fingerprints, test_fingerprints)


if __name__ == "__main__":
    main()
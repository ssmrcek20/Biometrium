import os
import shutil
import cv2
import numpy as np

def sort_images(fingerprintsSource, trainingFingerprints, testFingertips):
    images = []
    for img in os.listdir(fingerprintsSource):
        if img.endswith(('.tif')):
            images.append(img)

    for i, img in enumerate(images, start=1):
        if i % 8 == 0:
            shutil.copy2(os.path.join(fingerprintsSource, img), testFingertips)
        else:
            shutil.copy2(os.path.join(fingerprintsSource, img), trainingFingerprints)

def process_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    thresholded_image = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return thresholded_image

def main():
    fingerprintsSource = './DB2_B'
    trainingFingerprints = './TestDataset'
    testFingertips = './TrainingDataset'
    if os.listdir(trainingFingerprints) or os.listdir(testFingertips):
        print("Data already sorted.")
    else:
        sort_images(fingerprintsSource, trainingFingerprints, testFingertips)


if __name__ == "__main__":
    main()



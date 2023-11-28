import os
import shutil
import cv2
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def save_processed_image(fingerprint_image, fingerprint_label, proc_fingerprints_source):
    cv2.imwrite(os.path.join(proc_fingerprints_source, fingerprint_label), fingerprint_image)

def process_image(image_path):
    gray_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    thresholded_image = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return thresholded_image

def fingerprints_load(fingerprints_source, proc_fingerprints_source):
    images = []
    labels = []

    for img in os.listdir(fingerprints_source):
        if img.endswith('.tif'):
            img_source = os.path.join(fingerprints_source, img)
            processed_image = process_image(img_source)
            flattened_image = np.array(processed_image).flatten()
            images.append(flattened_image)

            img_labels = img.split('_')
            label = img_labels[0] 
            labels.append(label)

            save_processed_image(processed_image, img, proc_fingerprints_source)

    return images, labels

def main():
    fingerprints_source = './Fingerprints_DB'
    if not os.path.exists(fingerprints_source):
        print("Fingerprint database does not exist.")
        return
    proc_fingerprints_source = './Processed_Fingerprints'
    if os.path.exists(proc_fingerprints_source):
        shutil.rmtree(proc_fingerprints_source)
    os.makedirs(proc_fingerprints_source)

    fingerprints, labels = fingerprints_load(fingerprints_source, proc_fingerprints_source)
    norm_fingerprints = preprocessing.normalize(fingerprints, norm='l2')

    fingerprints_train, fingerprints_test, labels_train, labels_test = train_test_split(norm_fingerprints, labels, test_size=0.25, random_state=42)

    model = OneVsRestClassifier(svm.SVC(kernel='rbf', C=10.0, gamma=0.9, decision_function_shape='ovr'))
    model.fit(fingerprints_train, labels_train)

    labels_pred = model.predict(fingerprints_test)

    print(labels_pred)
    print(labels_test)

    accuracy = accuracy_score(labels_test, labels_pred)
    print("Accuracy: {:.2f}%".format(accuracy * 100))

if __name__ == "__main__":
    main()
import os
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

def process_image(image_path):
    gray_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    thresholded_image = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return thresholded_image

def fingerprints_load(fingerprints_source):
    images = []
    labels = []

    for img in os.listdir(fingerprints_source):
        if img.endswith(('.tif')):
            img_source = os.path.join(fingerprints_source, img)
            processed_image = process_image(img_source)

            images.append(processed_image)

            img_label = img.split('_')
            if len(img_label) == 3:
                label = img_label[0] + '_' + img_label[1]
            else:
                label = img_label[0]
                
            labels.append(label)

    return images, labels

def main():
    fingerprints_source = './Fingerprints_DB'
    if not os.path.exists(fingerprints_source):
        print("Fingerprint database does not exist.")
        return

    fingerprints, labels = fingerprints_load(fingerprints_source)
    fingerprints = np.array(fingerprints).reshape(len(fingerprints), -1)

    fingerprints_train, fingerprints_test, labels_train, labels_test = train_test_split(fingerprints, labels, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(fingerprints_train, labels_train)

    labels_pred = model.predict(fingerprints_test)

    accuracy = accuracy_score(labels_test, labels_pred)
    print("Točnost modela: {:.2f}%".format(accuracy * 100))

    cm = confusion_matrix(labels_test, labels_pred)
    epsilon = 1e-7
    FAR = cm[0, 1] / (cm[0, 0] + cm[0, 1] + epsilon)
    FRR = cm[1, 0] / (cm[1, 0] + cm[1, 1] + epsilon)

    print("False Acceptance Rate (FAR): {:.2f}%".format(FAR * 100))
    print("False Rejection Rate (FRR): {:.2f}%".format(FRR * 100))

if __name__ == "__main__":
    main()
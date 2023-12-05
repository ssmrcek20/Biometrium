import os
import shutil
import cv2
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def fingerprints_setup(fingerprints_source, proc_fingerprints_source):
    if os.path.exists(proc_fingerprints_source):
        shutil.rmtree(proc_fingerprints_source)
    os.makedirs(proc_fingerprints_source)

    fingerprints, labels = fingerprints_load(fingerprints_source, proc_fingerprints_source)
    norm_fingerprints = preprocessing.normalize(fingerprints, norm='l2')
    return labels,norm_fingerprints

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

def process_image(image_path):
    gray_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    thresholded_image = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return thresholded_image

def save_processed_image(fingerprint_image, fingerprint_label, proc_fingerprints_source):
    cv2.imwrite(os.path.join(proc_fingerprints_source, fingerprint_label), fingerprint_image)

def probabilitis_of_predictions(fingerprints_test, model, threshold):
    votes = np.array(model.decision_function(fingerprints_test))
    prob = np.exp(votes)/np.sum(np.exp(votes),axis=1, keepdims=True)
    thresholded_predictions = (np.max(prob, axis=1) > threshold).astype(int)
    return thresholded_predictions

def main():
    fingerprints_source = './Fingerprints_DB'
    proc_fingerprints_source = './Processed_Fingerprints'
    if not os.path.exists(fingerprints_source):
        print("Fingerprint database does not exist.")
        return
    
    labels, norm_fingerprints = fingerprints_setup(fingerprints_source, proc_fingerprints_source)

    fingerprints_train, fingerprints_test, labels_train, labels_test = train_test_split(norm_fingerprints, labels, test_size=0.2, random_state=42)
    print("Fingerprints are procesed and sorted into training and testing sets.\nNow starting traning of the model.")

    model = OneVsRestClassifier(svm.SVC(kernel='rbf', C=10.0, gamma=1.0, decision_function_shape='ovr'))
    model.fit(fingerprints_train, labels_train)

    labels_pred = model.predict(fingerprints_test)

    accuracy = accuracy_score(labels_test, labels_pred)
    print("Traning is finished. \nAccuracy of model: {:.2f}%".format(accuracy * 100))

    threshold = 0.1

    prob = probabilitis_of_predictions(fingerprints_test, model, threshold)

    correct_predictions = 0
    for i in range(len(labels_test)):
        if labels_test[i] == labels_pred[i] and prob[i] == 1:
            correct_predictions += 1

    accuracy = correct_predictions / len(labels_test)
    print("FRR: {:.2%}".format(1 - accuracy))


    fake_fingerprints_source = './Imposter_Fingerprints_DB'
    proc_fake_fingerprints_source = './Imposter_Processed_Fingerprints'
    if not os.path.exists(fake_fingerprints_source):
        print("Fake fingerprint database does not exist.")
        return
    
    fake_labels, fake_norm_fingerprints = fingerprints_setup(fake_fingerprints_source, proc_fake_fingerprints_source)

    prob = probabilitis_of_predictions(fake_norm_fingerprints, model, threshold)

    false_predictions = 0
    for i in range(len(fake_labels)):
        if prob[i] == 1:
            false_predictions += 1

    far = false_predictions / len(labels_test)
    print("FAR: {:.2%}".format(far))


if __name__ == "__main__":
    main()
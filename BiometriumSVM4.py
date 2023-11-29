import os
import shutil
import cv2
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def get_image_dimensions(image_path):
    img = cv2.imread(image_path)
    height, width = img.shape[:2]
    return height, width

def resize_image(image_path, new_width, new_height):
    img = cv2.imread(image_path)
    resized_img = cv2.resize(img, (new_width, new_height))
    return resized_img

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
    # FRR: False Rejection Rate
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

    accuracy = accuracy_score(labels_test, labels_pred)
    print("Accuracy: {:.2f}%".format(accuracy * 100))

    threshold = 0.17

    votes = np.array(model.decision_function(fingerprints_test))
    prob = np.exp(votes)/np.sum(np.exp(votes),axis=1, keepdims=True)
    thresholded_predictions = (np.max(prob, axis=1) > threshold).astype(int)
    print(thresholded_predictions)

    correct_predictions = []
    for i in range(len(labels_test)):
        if(labels_test[i] == labels_pred[i] and thresholded_predictions[i] == 1):
            correct_predictions.append(True)
        elif(labels_test[i] == labels_pred[i] and thresholded_predictions[i] == 0):
            correct_predictions.append(False)
        elif(labels_test[i] != labels_pred[i] and thresholded_predictions[i] == 1):
            correct_predictions.append(False)
        else:
            correct_predictions.append(True)
    print(correct_predictions)
    accuracy = np.mean(correct_predictions)
    print("Accuracy: {:.2f}%".format(accuracy * 100))
    
    frr = 1 - accuracy
    print("FRR: {:.2%}".format(frr))

    # FAR: False Acceptance Rate
    fake_fingerprints_source = './Imposter_Fingerprints_DB'
    if not os.path.exists(fake_fingerprints_source):
        print("Fake fingerprint database does not exist.")
        return
    proc_fake_fingerprints_source = './Imposter_Processed_Fingerprints'
    if os.path.exists(proc_fake_fingerprints_source):
        shutil.rmtree(proc_fake_fingerprints_source)
    os.makedirs(proc_fake_fingerprints_source)

    fing_img = os.listdir(fingerprints_source)
    height, width = get_image_dimensions(os.path.join(fingerprints_source, fing_img[0]))

    for img in os.listdir(fake_fingerprints_source):
        if img.endswith('.tif'):
            fake_image_path = os.path.join(fake_fingerprints_source, img)
            resized_fake_image = resize_image(fake_image_path, width, height)
            save_processed_image(resized_fake_image, img, fake_fingerprints_source)

    fake_fingerprints, fake_labels = fingerprints_load(fake_fingerprints_source, proc_fake_fingerprints_source)
    fake_norm_fingerprints = preprocessing.normalize(fake_fingerprints, norm='l2')

    fake_labels_pred = model.predict(fake_norm_fingerprints)

    accuracy = accuracy_score(fake_labels, fake_labels_pred)
    print("Accuracy: {:.2f}%".format(accuracy * 100))

    votes = np.array(model.decision_function(fake_norm_fingerprints))
    prob = np.exp(votes)/np.sum(np.exp(votes),axis=1, keepdims=True)
    thresholded_predictions = (np.max(prob, axis=1) > threshold).astype(int)
    print(thresholded_predictions)

    correct_predictions = []
    for i in range(len(fake_labels)):
        if(fake_labels[i] == fake_labels_pred[i] and thresholded_predictions[i] == 1):
            correct_predictions.append(True)
        elif(fake_labels[i] == fake_labels_pred[i] and thresholded_predictions[i] == 0):
            correct_predictions.append(False)
        elif(fake_labels[i] != fake_labels_pred[i] and thresholded_predictions[i] == 1):
            correct_predictions.append(False)
        else:
            correct_predictions.append(True)
    print(correct_predictions)
    accuracy = np.mean(correct_predictions)
    print("Accuracy: {:.2f}%".format(accuracy * 100))
    
    far = 1 - accuracy
    print("FAR: {:.2%}".format(far))



if __name__ == "__main__":
    main()
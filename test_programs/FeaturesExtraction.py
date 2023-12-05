import os
import shutil
import cv2
import numpy as np
import fingerprint_feature_extractor as ffe
import fingerprint_enhancer as fe
from sklearn import svm, preprocessing
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

def process_image(image_path):
    img = cv2.imread(image_path, 0)
    return fe.enhance_Fingerprint(img)

def process_image_2(image_path):
    gray_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    thresholded_image = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return thresholded_image

def fingerprints_load(fingerprints_source, processed_fingerprints_folder):
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

            cv2.imwrite(os.path.join(processed_fingerprints_folder, img), processed_image)

    return images, labels

def create_folders(processed_fingerprints_folder, features_fingerprints_folder):
    if os.path.exists(processed_fingerprints_folder):
        shutil.rmtree(processed_fingerprints_folder)
    os.makedirs(processed_fingerprints_folder)

    if os.path.exists(features_fingerprints_folder):
        shutil.rmtree(features_fingerprints_folder)
    os.makedirs(features_fingerprints_folder)

def save_extracted_fingerprint(features_fingerprints_folder,labels,counter):
    destination_path = os.path.join(features_fingerprints_folder, f"{labels[counter]}_{counter}.tif")
    
    shutil.copy("./result.png", destination_path)
    os.remove("./result.png")

def extract_features(fingerprints, labels, features_fingerprints_folder):
    features = []
    counter = 0

    for fingerprint in fingerprints:
        Terminations, Bifurcations = ffe.extract_minutiae_features(fingerprint, spuriousMinutiaeThresh=10, invertImage=False, showResult=False, saveResult=True)
        
        save_extracted_fingerprint(features_fingerprints_folder, labels, counter)
        counter += 1

        fingerprint_features = []
        for termination in Terminations:
            orientation = termination.Orientation[0]
            loc_x = termination.locX
            loc_y = termination.locY

            fingerprint_features.extend([orientation, loc_x, loc_y])

        pad_length = 300 - len(fingerprint_features)
        fingerprint_features += [0] * pad_length

        print(labels[counter-1])
        print(fingerprint_features)


        features.append(fingerprint_features)
    return features


def main():
    fingerprints_folder = './Fingerprints_DB'
    processed_fingerprints_folder = './Processed_Fingerprints'
    features_fingerprints_folder = './Features_Processed_Fingerprints'

    if not os.path.exists(fingerprints_folder):
        print("Fingerprint database does not exist.")
        return
    
    create_folders(processed_fingerprints_folder,features_fingerprints_folder)

    fingerprints, labels = fingerprints_load(fingerprints_folder,processed_fingerprints_folder)
    #processed_images = np.asarray([np.array(im).flatten() for im in fingerprints])
    #features = preprocessing.normalize(processed_images, norm='l2')

    features = extract_features(fingerprints, labels, features_fingerprints_folder)

    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    clf = OneVsRestClassifier(svm.SVC(kernel='rbf', C=1.0, gamma=0.9, decision_function_shape='ovr'))
    clf.fit(features_train, labels_train)
    labels_pred = clf.predict(features_test)

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
import os
import shutil
import cv2
import numpy as np
import fingerprint_feature_extractor as ffe
import fingerprint_enhancer as fe
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

def save_processed_image(fingerprint_image, fingerprint_label, processed_fingerprints_folder):
    cv2.imwrite(os.path.join(processed_fingerprints_folder, fingerprint_label), fingerprint_image)

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
            processed_image = process_image_2(img_source)

            images.append(processed_image)

            img_label = img.split('_')
            if len(img_label) == 3:
                label = img_label[0] + '_' + img_label[1]
            else:
                label = img_label[0]
                
            labels.append(label)

            save_processed_image(processed_image, img, processed_fingerprints_folder)

    return images, labels

def main():
    fingerprints_source = './Fingerprints_DB'
    if not os.path.exists(fingerprints_source):
        print("Fingerprint database does not exist.")
        return
    
    processed_fingerprints_folder = './Processed_Fingerprints'
    if os.path.exists(processed_fingerprints_folder):
        shutil.rmtree(processed_fingerprints_folder)
    os.makedirs(processed_fingerprints_folder)

    fingerprints, labels = fingerprints_load(fingerprints_source,processed_fingerprints_folder)

    features = []

    for fingerprint in fingerprints:
        FeaturesTerminations, FeaturesBifurcations = ffe.extract_minutiae_features(fingerprint, spuriousMinutiaeThresh=10, invertImage=False, showResult=False, saveResult=False)
        
        print("Broj terminacija: {}, broj bifurkacija: {}".format(len(FeaturesTerminations), len(FeaturesBifurcations)))

        max_len = max(len(FeaturesTerminations), len(FeaturesBifurcations))
        pad_zeros = np.zeros(max_len)

        feature = np.concatenate([
            np.concatenate([FeaturesTerminations, pad_zeros]),
            np.concatenate([FeaturesBifurcations, pad_zeros])
        ])
        features.append(feature)

    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    model = SVC(kernel='linear', C=1.0)
    model.fit(features_train, labels_train)

    labels_pred = model.predict(features_test)

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
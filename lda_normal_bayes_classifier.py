# @brief LdaNormalBayesClassifier
# @author Jose M. Buenaposada (josemiguel.buenaposada@urjc.es)
# @date 2025

# A continuación se presenta un esquema de la clase necesaria para implementar el clasificador
# propuesto en el Ejercicio1 de la práctica. Habrá que terminar la implementación
# Modificar como se crea conveniente (incluyendo métodos y parámetros), únicamente es una guía.

import string
import cv2
import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from ocr_classifier import OCRClassifier

class LdaNormalBayesClassifier(OCRClassifier):
    """
    Classifier for Optical Character Recognition using differents reduction algorithms and classifiers.
    """

    def __init__(self, ocr_char_size=(25, 25), classifier_type="sklearn_bayes", reduction="lda"):
        super().__init__(ocr_char_size)
        self.reduction = reduction
        self.pca = None
        self.lda = None
        self.svd = None
        self.classifier = None
        self.classifier_type = classifier_type

        if classifier_type == "sklearn_bayes":
            self.classifier_name = "Gaussian Naive Bayes (sklearn)"
        elif classifier_type == "sklearn_knn":
            self.classifier_name = "KNN (sklearn)"
        elif classifier_type == "sklearn_svm":
            self.classifier_name = "SVM (sklearn)"
        elif classifier_type == "sklearn_rf":
            self.classifier_name = "Random Forest (sklearn)"
        elif classifier_type == "sklearn_lr":
            self.classifier_name = "Logistic Regresion (sklearn)"
        elif classifier_type == "opencv_knn":
            self.classifier_name = "KNN (OpenCV)"
        elif classifier_type == "opencv_svm":
            self.classifier_name = "SVM (OpenCV)"
        else:
            raise ValueError(
                f"Unknown classifier type: '{classifier_type}' Valid options: "
                "sklearn_bayes | sklearn_knn | sklearn_svm | sklearn_rf | sklearn_lr | opencv_knn | opencv_svm"
            )

        valid_reductions = ["lda", "pca", "svd"]

        if self.reduction not in valid_reductions:
            raise ValueError(
                f"Unknown reduction type: '{self.reduction}' "
                f"Valid options: {' | '.join(valid_reductions)}"
            )
        
        red_name = reduction.upper()
        self.classifier_name = f'{red_name} + {self.classifier_name}'

    def train(self, images_dict):
        """
        Given character images in a dictionary of list of char images, 
        train the OCR classifier. The dictionary keys are the class of the list of images 
        (or corresponding char).

        :images_dict is a dictionary of images (name of the images is the key)
        """

        # Take training images and do feature extraction
        print("\n[TRAIN] Extracting features from training images...")
        X = []
        y = []
        
        all_chars = string.digits + string.ascii_lowercase + string.ascii_uppercase
        total_images = 0
        
        for char in all_chars:
            if char not in images_dict:
                continue
            
            char_count = 0
            for img in images_dict[char]:
                # Apply preprocessing
                features = self.preprocess(img)
                if features is None:
                    continue
                
                X.append(features)
                y.append(self.char2label(char))
                char_count += 1
            
            print(f"[TRAIN] Character '{char}': {char_count} images processed ({self.ocr_char_size[0] * self.ocr_char_size[1]} pixels each)")
            total_images += char_count
        
        if len(X) == 0:
            raise ValueError("No features extracted from training images")
        
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int32)
        
        print(f"[TRAIN] Total: {len(X)} feature vectors extracted")
        print(f"[TRAIN] Feature matrix shape: {X.shape[0]} x {X.shape[1]}")
        
        # Perform Classifier training
        n_components = min(len(all_chars) - 1, X.shape[1] // 4)
        X_reduced = self.reduce(n_components, X, y)
        
        print(f"[TRAIN] Training dimensions: {X.shape[1]} -> {X_reduced.shape[1]}")
        
        # Perform Classifier training
        print(f"[TRAIN] Training {self.classifier_name}...")
        self.train_classifier(X_reduced, y)
        print(f"[TRAIN] Classifier trained successfully!")

    def predict(self, img):
        """
        Given a single image of a character classify it.

        :img Image to classify
        
        """

        if (self.lda is None and self.pca is None and self.svd is None) or self.classifier is None:
            raise RuntimeError("Classifier not trained")
        
        # Apply same preprocessing as in training
        features = self.preprocess(img)
        if features is None:
            return 1
        
        # Obtain the estimated label by the classifier
        x = features.reshape(1, -1)

        if self.reduction == "lda":
            features_reduced = self.lda.transform(x)
        elif self.reduction == "pca":
            features_reduced = self.pca.transform(x)
        elif self.reduction == "svd":
            features_reduced = self.svd.transform(x)

        if self.classifier_type in ["opencv_svm", "opencv_knn"]:
            # OpenCV classifiers require float32
            features_reduced_f32 = features_reduced.astype(np.float32)
            _, y = self.classifier.predict(features_reduced_f32)
            y = int(y[0][0])
        else:
            # sklearn classifiers
            y = self.classifier.predict(features_reduced)[0]
 
        return int(y)
    
    def predict_dict(self, images_dict):
        """
        Override parent method to add prints for tracking
        """
        responses = []
        total = 0
        for char in images_dict.keys():
            char_count = 0
            for img in images_dict[char]:
                responses.append(self.predict(img))
                char_count += 1
                total += 1
            if char_count > 0:
                print(f"[PREDICT] Character '{char}': predicted {char_count} images")
        
        print(f"[PREDICT] Total predictions: {total}")
        return responses
    
    def preprocess(self, img):
        """
        Preprocess the input image to extract features. This includes adaptive thresholding, contour detection,
        and resizing to a fixed size. The output is a flattened feature vector.
        """
        if img.dtype != np.uint8:
            img = cv2.convertScaleAbs(img)

        binary = cv2.adaptiveThreshold(
            img, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11, 2
        )

        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) == 0:
            return None

        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)

        # Extract ROI
        roi = img[y:y+h, x:x+w]
        if roi.size == 0:
            return None

        # Resize to fixed size and convert to vector and normalize
        roi = cv2.resize(roi, self.ocr_char_size, interpolation=cv2.INTER_AREA)
        features = roi.flatten().astype(np.float32) / 255

        return features
    
    def reduce(self, n_components, X, y):
        """
        Apply dimensionality reduction to the feature matrix X using the specified reduction method.
        """
        if self.reduction == "lda":
            print(f"[TRAIN] Applying LDA with {n_components} components...")
            self.lda = LinearDiscriminantAnalysis(n_components=min(len(set(y))-1, X.shape[1]))
            X_reduced = self.lda.fit_transform(X, y)

        elif self.reduction == "pca":
            print(f"[TRAIN] Applying PCA with {n_components} components...")
            self.pca = PCA(n_components=0.95)
            X_reduced = self.pca.fit_transform(X)

        elif self.reduction == "svd":
            print(f"[TRAIN] Applying Truncated SVD with {n_components} components...")
            self.svd = TruncatedSVD(n_components=50)
            X_reduced = self.svd.fit_transform(X)
        
        return X_reduced
    
    def train_classifier(self, X_reduced, y):
        """
        Train the specified classifier using the reduced feature matrix X_reduced and labels y.
        """
        if self.classifier_type == "sklearn_bayes":
            self.classifier = GaussianNB()
            self.classifier.fit(X_reduced, y)

        elif self.classifier_type == "sklearn_knn":
            self.classifier = KNeighborsClassifier(n_neighbors=3)
            self.classifier.fit(X_reduced, y)
        
        elif self.classifier_type == "sklearn_svm":
            self.classifier = SVC(kernel='rbf', C=1.0)
            self.classifier.fit(X_reduced, y)

        elif self.classifier_type == "sklearn_rf":
            self.classifier = RandomForestClassifier(n_estimators=100)
            self.classifier.fit(X_reduced, y)

        elif self.classifier_type == "sklearn_lr":
            self.classifier = LogisticRegression(max_iter=1000)
            self.classifier.fit(X_reduced, y)
        
        elif self.classifier_type == "opencv_svm":
            # OpenCV SVM requires float32
            X_reduced_f32 = X_reduced.astype(np.float32)
            y_f32 = y.astype(np.int32)
            self.classifier = cv2.ml.SVM_create()
            self.classifier.setType(cv2.ml.SVM_C_SVC)
            self.classifier.setKernel(cv2.ml.SVM_LINEAR)
            self.classifier.setC(2.5)
            self.classifier.setGamma(0.01)
            self.classifier.train(X_reduced_f32, cv2.ml.ROW_SAMPLE, y_f32)
        
        elif self.classifier_type == "opencv_knn":
            # OpenCV KNN requires float32
            X_reduced_f32 = X_reduced.astype(np.float32)
            y_f32 = y.astype(np.int32)
            self.classifier = cv2.ml.KNearest_create()
            self.classifier.train(X_reduced_f32, cv2.ml.ROW_SAMPLE, y_f32)
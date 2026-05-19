# Asignatura de Visión Artificial (URJC). Script de evaluación.
# @author Jose M. Buenaposada (josemiguel.buenaposada@urjc.es)
# @date 2025


import argparse
import os
import string
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import numpy as np
import sklearn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from lda_normal_bayes_classifier import LdaNormalBayesClassifier

def plot_confusion_matrix(cm, labels, title='Confusion matrix', cmap='Blues'):
    '''
    Given a confusión matrix in cm (np.array) it plots it in a fancy way.
    '''
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    ax = plt.gca()
    width = cm.shape[1]
    height = cm.shape[0]

    for x in range(width):
        for y in range(height):
            ax.annotate(str(cm[y,x]), xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center')
            
def load_character_images(data_path):
    """
    Load character images from directory {data_path} with the right structure.
    
    Returns: dictionary {character: [list of images]}
    """
    images_dict = {}
    all_chars = string.digits + string.ascii_lowercase + string.ascii_uppercase
    total_images = 0
    
    for char in all_chars:
        if char.isdigit():
            char_path = os.path.join(data_path, char)
        elif char.islower():
            char_path = os.path.join(data_path, 'min', char)
        elif char.isupper():
            char_path = os.path.join(data_path, 'may', char)

        images = []

        if os.path.isdir(char_path):
            for filename in os.listdir(char_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(char_path, filename)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        images.append(img)

        if len(images) > 0:
            print(f"[LOADING] Character '{char}': {len(images)} images loaded")
            total_images += len(images)
        images_dict[char] = images
    
    print(f"[LOADING] Total images loaded: {total_images}")
    return images_dict

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Trains and executes a given classifier for OCR over testing images')
    parser.add_argument(
        '--classifier', type=str, default="sklearn_rf", help='Classifier string name: ' \
        'sklearn_bayes | sklearn_knn | sklearn_svm | sklearn_rf | sklearn_lr | opencv_knn | opencv_svm')
    
    parser.add_argument(
        '--train_path', default="./train_ocr", help='Select the training data dir')
    parser.add_argument(
        '--validation_path', default="./test_ocr", help='Select the validation data dir')
    parser.add_argument(
        '--reduction', type=str, default="lda", help='Reduction algorithm name: lda | pca | svd')

    args = parser.parse_args()

    print("=" * 60)
    print("OCR CLASSIFIER")
    print("=" * 60)

    classifier = LdaNormalBayesClassifier()
    classifier = LdaNormalBayesClassifier(classifier_type=args.classifier, reduction=args.reduction)

    # 1) Load training images and labels
    # Also extract feature vectors (basic: thresholding + findContours + resize)
    print("\n[LOADING] Loading training images from:", args.train_path)
    train_images_dict = load_character_images(args.train_path)
    print(f"[LOADING] Training images loaded")
    
    # 2) Load validation images and labels
    # Also extract feature vectors (basic: thresholding + findContours + resize)
    print("\n[LOADING] Loading validation images from:", args.validation_path)
    val_images_dict = load_character_images(args.validation_path)
    print(f"[LOADING] Validation images loaded")
    
    # Get ground truth labels
    print("\n[PREPARING] Extracting ground truth labels...")
    gt_labels = []
    for char in val_images_dict.keys():
        for img in val_images_dict[char]:
            gt_labels.append(classifier.char2label(char))

    print(f"[PREPARING] Total ground truth labels: {len(gt_labels)}")

    # 3) Train classifier
    print("\n[TRAINING] Starting classifier training...")
    classifier.train(train_images_dict)

    # 4) Execute classifier on validation data
    print("\n[EVALUATING] Making predictions on validation data...")
    predicted_labels = classifier.predict_dict(val_images_dict)

    # 5) Evaluate results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    accuracy = accuracy_score(gt_labels, predicted_labels)
    print(f"\nAccuracy = {accuracy} ({accuracy*100:.2f}%)")

    # Additional evaluation metrics
    labels = [classifier.label2char(i) for i in sorted(set(gt_labels))]
    print("\nClassification Report:")
    print(classification_report(gt_labels, predicted_labels, target_names=labels, digits=4))
    
    cm = confusion_matrix(gt_labels, predicted_labels)
    print(f"Confusion matrix shape: {cm.shape}")
    
    # Plot confusion matrix
    print("\n[PLOTTING] Displaying confusion matrix...")
    plt.figure(figsize=(12, 9))
    plot_confusion_matrix(cm, labels, title=f'Confusion Matrix - {classifier.classifier_name}')
    plt.gca().set_aspect('auto')
    plt.tight_layout()
    plt.show()
    
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
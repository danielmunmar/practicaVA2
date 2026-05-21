import argparse
import os
import cv2
import numpy as np
from lda_normal_bayes_classifier import LdaNormalBayesClassifier
from evaluar_clasificadores_OCR import load_character_images


def preprocess_panel(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    _, thresh = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (2,2)
    )

    thresh = cv2.morphologyEx(
        thresh,
        cv2.MORPH_OPEN,
        kernel
    )

    thresh = cv2.bitwise_not(thresh)

    return gray, thresh

def detect_characters(thresh):
    thresh = thresh.astype(np.uint8)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        thresh,
        connectivity=8
    )

    chars = []

    img_h, img_w = thresh.shape

    for i in range(1, num_labels):

        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        if area < 30:
            continue

        if area > 0.15 * img_h * img_w:
            continue

        if w > img_w * 0.5:
            continue

        if h > img_h * 0.4:
            continue

        if h < 8 or w < 2:
            continue

        aspect_ratio = w / float(h)

        if aspect_ratio < 0.08 or aspect_ratio > 3.5:
            continue

        density = area / float(w * h)

        if density < 0.15:
            continue

        chars.append((x, y, w, h))

    return chars

def group_lines(chars, y_threshold=15):
    """
    Agrupa caracteres en líneas según cercanía vertical
    """

    if len(chars) == 0:
        return []

    # order by y coordinate
    chars = sorted(chars, key=lambda c: c[1])

    lines = []

    for char in chars:

        x, y, w, h = char
        cy = y + h // 2

        added = False

        for line in lines:

            line_centers = [
                ly + lh // 2 for (_, ly, _, lh) in line
            ]

            mean_y = np.mean(line_centers)

            if abs(cy - mean_y) < y_threshold:
                line.append(char)
                added = True
                break

        if not added:
            lines.append([char])

    # order characters from left to right characters in each line
    for line in lines:
        line.sort(key=lambda c: c[0])

    # order from top to bottom
    lines.sort(key=lambda line: np.mean([c[1] for c in line]))

    return lines

def extract_char(gray, bbox):

    x, y, w, h = bbox

    pad = 3
    x = max(x - pad, 0)
    y = max(y - pad, 0)
    w = min(w + 2*pad, gray.shape[1] - x)
    h = min(h + 2*pad, gray.shape[0] - y)

    crop = gray[y:y+h, x:x+w]

    if crop.size == 0:
        return np.zeros((25, 25), dtype=np.uint8)

    _, crop = cv2.threshold(
        crop,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    crop = cv2.resize(
        crop,
        (25,25),
        interpolation=cv2.INTER_NEAREST
    )
    crop = cv2.bitwise_not(crop)

    return crop

def build_ocr_text(gray, lines,classifier):
    """
    Construye el string OCR final
    """

    text_lines = []

    for line in lines:

        line_text = ""

        for bbox in line:

            char_img = extract_char(gray, bbox)
            #cv2.imshow("ROI", char_img)
            #cv2.waitKey(0)

            pred_label = classifier.predict(char_img)

            pred_char = classifier.label2char(pred_label)

            line_text += pred_char

        text_lines.append(line_text)

    return "+".join(text_lines)



def visualize(img, thresh, lines,classifier):
    """
    Visualización de resultados
    """

    vis = img.copy()

    colors = [
        (0, 255, 0),
        (255, 0, 0),
        (0, 0, 255),
        (255, 255, 0)
    ]

    for idx, line in enumerate(lines):

        color = colors[idx % len(colors)]

        centers = []

        for bbox in line:

            x, y, w, h = bbox

            cv2.rectangle(
                vis,
                (x, y),
                (x+w, y+h),
                color,
                2
            )

            char_img = extract_char(gray, bbox)

            pred_label = classifier.predict(char_img)

            pred_char = classifier.label2char(pred_label)

            cv2.putText(
                vis,
                pred_char,
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

            centers.append((x + w//2, y + h//2))

        # draw line
        if len(centers) >= 2:

            p1 = centers[0]
            p2 = centers[-1]

            cv2.line(vis, p1, p2, color, 2)

    cv2.imshow("threshold", thresh)
    cv2.imshow("ocr", vis)

    cv2.waitKey(0)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Trains and executes a given detector over a set of testing images')
    parser.add_argument(
        '--detector', type=str, nargs="?", default="", help='Detector string name')
    parser.add_argument(
        '--train_path', default="", help='Select the training data dir')
    parser.add_argument(
        '--test_path', default="", help='Select the testing data dir')

    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Visualize OCR results'
    )

    args = parser.parse_args()

    # Load training data

    print("[INFO] Loading OCR training data...")

    train_images = load_character_images(args.train_path)

    print("[INFO] Training images loaded")

    # Create the OCR classifier


    classifier = LdaNormalBayesClassifier(
        ocr_char_size=(25,25),
        classifier_type="sklearn_knn",
        reduction="pca"
    )

    print("[INFO] Training OCR classifier...")

    classifier.train(train_images)

    print("[INFO] OCR classifier trained")

    # Load testing data
    test_images = sorted([
        f for f in os.listdir(args.test_path)
        if f.endswith(".png") or f.endswith(".jpg")
    ])

    results_file = open("resultado.txt", "w")


    # Evaluate OCR over road panels
    for filename in test_images:

        img_path = os.path.join(args.test_path, filename)

        img = cv2.imread(img_path)

        if img is None:
            continue

        h, w = img.shape[:2]

        # Preprocess image
        gray, thresh = preprocess_panel(img)

        # Detect characters
        chars = detect_characters(thresh)

        # Group text lines
        lines = group_lines(chars)

        # Build OCR text
        ocr_text = build_ocr_text(gray, lines,classifier)

        result_line = (
            f"{filename};"
            f"0;0;{w};{h};"
            f"panel;1.0;"
            f"{ocr_text}\n"
        )

        results_file.write(result_line)

        print(result_line.strip())

        if args.visualize:
            visualize(img, thresh, lines,classifier)

    results_file.close()

    cv2.destroyAllWindows()

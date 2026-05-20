import argparse
import os
import cv2
import numpy as np
from lda_normal_bayes_classifier import LdaNormalBayesClassifier
from evaluar_clasificadores_OCR import load_character_images


def preprocess_panel(img):

    # 1. Escala de grises
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Suavizado ligero (reduce ruido antes de edges)
    blur = cv2.bilateralFilter(gray, 9, 75, 75)

    # 3. Detección de bordes (clave del nuevo enfoque)
    edges = cv2.Canny(blur, 50, 150)

    # 4. Operación morfológica para unir fragmentos de contornos
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 5. Encontrar contornos externos
    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # 6. Crear máscara vacía
    mask = np.zeros_like(gray)

    # 7. Filtrar y dibujar SOLO contornos relevantes
    h, w = gray.shape

    for cnt in contours:

        area = cv2.contourArea(cnt)
        if area < 80:   # filtro ruido
            continue

        x, y, cw, ch = cv2.boundingRect(cnt)

        # filtrar contornos absurdos (muy grandes o muy pequeños)
        if ch < 10 or cw < 3:
            continue
        if area > 0.9 * (h * w):
            continue

        # rellenar contorno en la máscara
        cv2.drawContours(mask, [cnt], -1, 255, -1)

    # 8. Aplicar máscara a la imagen original
    masked = cv2.bitwise_and(gray, gray, mask=mask)

    # 9. Binarización final (más estable para OCR + detect_characters)
    thresh = cv2.threshold(
        masked,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]

    return gray, thresh

def detect_characters(thresh):
    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )

    chars = []

    img_h, img_w = thresh.shape

    heights = [cv2.boundingRect(c)[3] for c in contours]

    if len(heights) == 0:
        return []

    median_h = np.median(heights)

    for cnt in contours:

        x, y, w, h = cv2.boundingRect(cnt)

        area = w * h

        aspect_ratio = w / float(h)

        # filtros básicos

        

        if h < 0.4 * median_h:
            continue


        if w < 5 or h < 12:
            continue

        if area < 80:
            continue

        if h > img_h * 0.8:
            continue

        
        if aspect_ratio < 0.15 or aspect_ratio > 2.5:
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

    # --------------------------------------------------------
    # MISMO PREPROCESADO QUE TRAIN
    # --------------------------------------------------------

    binary = cv2.adaptiveThreshold(
        crop,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )
    binary_inv = 255 - binary
    
    contours, _ = cv2.findContours(
        binary_inv,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return cv2.resize(crop, (25,25))

    largest = max(contours, key=cv2.contourArea)

    x2, y2, w2, h2 = cv2.boundingRect(largest)

    char_crop = binary_inv[y2:y2+h2, x2:x2+w2]

    if char_crop.size == 0:
        return np.zeros((25,25), dtype=np.uint8)

    # ------------------------------------------------
    # MANTENER ASPECT RATIO
    # ------------------------------------------------

    size = max(w2, h2)

    canvas = np.zeros((size, size), dtype=np.uint8)

    x_offset = (size - w2) // 2
    y_offset = (size - h2) // 2

    canvas[
        y_offset:y_offset+h2,
        x_offset:x_offset+w2
    ] = char_crop

    char_crop = cv2.resize(
        canvas,
        (40,40),
        interpolation=cv2.INTER_NEAREST
    )
    return char_crop

def build_ocr_text(gray, lines,classifier):
    """
    Construye el string OCR final
    """

    text_lines = []

    for line in lines:

        line_text = ""

        for bbox in line:

            char_img = extract_char(gray, bbox)

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
        ocr_char_size=(40,40),
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

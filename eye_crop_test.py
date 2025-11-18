import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

def sharpness_score(gray_img):
    """Return how sharp the image is using Laplacian variance."""
    return cv2.Laplacian(gray_img, cv2.CV_64F).var()

def contrast_score(gray_img):
    """Return contrast score using intensity distribution."""
    non_zero = gray_img[gray_img > 15]

    if len(non_zero) > 0:
        p_low = np.percentile(non_zero, 10)
        p_high = np.percentile(non_zero, 90)
        dynamic_range = p_high - p_low
    else:
        dynamic_range = 0

    std_dev = gray_img.std()
    return std_dev * 0.5 + dynamic_range * 0.5

def completeness_score(gray_img):
    """Check iris/pupil shape consistency using circle detection."""
    h, w = gray_img.shape
    center_region = gray_img[h//4:3*h//4, w//4:3*w//4]
    center_mean = center_region.mean()

    # Detect circular pattern
    circles = cv2.HoughCircles(
        gray_img,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=w//2,
        param1=50,
        param2=30,
        minRadius=w//6,
        maxRadius=w//2
    )

    circle_score = 50 if circles is not None else 0
    return abs(center_mean - gray_img.mean()) + circle_score

def combined_quality_score(gray_img):
    """Weighted combination of sharpness, contrast, and completeness."""
    if gray_img.mean() < 30 or gray_img.mean() > 220:
        return 0

    sharp = sharpness_score(gray_img)
    contr = contrast_score(gray_img)
    compl = completeness_score(gray_img)

    return 0.4 * sharp + 0.3 * contr + 0.3 * compl

def detect_best_eye(image_path, visualize=False):
    """Detect the best-quality eye region from the input face image."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Could not load image: {image_path}")
        return None, 0, None, []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Load OpenCV’s built-in eye detector
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

    # Detect eyes
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=6, minSize=(25, 25))

    if len(eyes) == 0:
        print("⚠ No eyes detected")
        return None, 0, None, []

    scores = []
    crops = []

    # Evaluate each eye
    for (x, y, w, h) in eyes:
        crop_gray = gray[y:y+h, x:x+w]
        crop_color = img[y:y+h, x:x+w]
        score = combined_quality_score(crop_gray)

        scores.append(score)
        crops.append(crop_color)

    # Select the best eye
    best_idx = np.argmax(scores)
    best_eye = crops[best_idx]
    best_score = scores[best_idx]

    # Optional visualization
    if visualize:
        fig, axes = plt.subplots(1, len(crops) + 1, figsize=(15, 5))
        axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Detected Eyes")
        axes[0].axis("off")

        for i, crop in enumerate(crops):
            axes[i+1].imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            axes[i+1].set_title(f"Eye {i+1}\nScore: {scores[i]:.1f}")
            axes[i+1].axis("off")

        plt.show()

    return best_eye, best_score, crops, scores
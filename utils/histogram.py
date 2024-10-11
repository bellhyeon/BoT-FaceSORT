import cv2
import numpy as np


def calculate_histogram(image, bins):
    # Calculate histogram for each color channel
    histogram = [cv2.calcHist([image], [i], None, [bins], [0, 256]) for i in range(3)]
    # Normalize the histogram
    histogram = [cv2.normalize(h, h).flatten() for h in histogram]
    return np.hstack(histogram)


def chi_square_distance(histA, histB):
    # Calculate the Chi-Square distance
    return 0.5 * np.sum(
        [((a - b) ** 2) / (a + b + 1e-10) for (a, b) in zip(histA, histB)]
    )

import cv2
import numpy as np

def detect_forgery_overlay(image_path):
    image = cv2.imread(image_path)
    original = image.copy()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=400)
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    mask = np.zeros(gray.shape, dtype=np.uint8)

    if descriptors is not None and len(descriptors) > 2:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(descriptors, descriptors)

        matches = sorted(matches, key=lambda x: x.distance)[:15]

        for m in matches:
            if abs(m.queryIdx - m.trainIdx) < 10:
                continue

            pt1 = keypoints[m.queryIdx].pt
            pt2 = keypoints[m.trainIdx].pt

            cv2.circle(mask, (int(pt1[0]), int(pt1[1])), 5, 255, -1)
            cv2.circle(mask, (int(pt2[0]), int(pt2[1])), 5, 255, -1)

    # CLEAN NOISE
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

    clean_mask = np.zeros_like(mask)

    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > 150:
            clean_mask[labels == i] = 255

    mask = clean_mask

    # CLUSTER
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # FINAL OUTPUT (SHARP OUTLINE)
    output = original.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(output, contours, -1, (0, 0, 255), 3)

    return output
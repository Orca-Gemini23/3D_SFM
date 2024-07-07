import cv2
import os
import numpy as np
from itertools import combinations

# Define the directory containing the images
input_folder = r'C:\Users\csyas\OneDrive\Desktop\projects\3D_SFM\IMG_1_featExtraction_Output'
output_folder = r'C:\Users\csyas\OneDrive\Desktop\projects\3D_SFM\IMG_2'

# Ensure the output folder exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Create a SIFT detector and BFMatcher
sift = cv2.SIFT_create()
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# Load images and detect features
images = []
keypoints = []
descriptors = []
filenames = []

for filename in os.listdir(input_folder):
    if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.webp')):
        filepath = os.path.join(input_folder, filename)
        img = cv2.imread(filepath)
        if img is None:
            print(f'Failed to load image: {filepath}')
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = sift.detectAndCompute(gray, None)
        images.append(img)
        keypoints.append(kp)
        descriptors.append(des)
        filenames.append(filename)
        print(f'Processed {filename}, found {len(kp)} keypoints')

# Perform exhaustive matching between each pair of images
for (i, j) in combinations(range(len(images)), 2):
    img1 = images[i]
    img2 = images[j]
    kp1 = keypoints[i]
    kp2 = keypoints[j]
    des1 = descriptors[i]
    des2 = descriptors[j]

    # Match descriptors
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # Draw matches
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Save the matching image
    output_path = os.path.join(output_folder, f'matches_{filenames[i]}_{filenames[j]}.jpg')
    cv2.imwrite(output_path, img_matches)
    print(f'Saved matching result: {output_path}')

print('Feature matching completed!')

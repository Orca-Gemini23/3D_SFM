import cv2
import numpy as np
import os

# Define input folder containing feature matched images
input_folder = r'C:\Users\csyas\OneDrive\Desktop\projects\3d_Construction\IMG_2_featMatching_Output'
output_folder = r'C:\Users\csyas\OneDrive\Desktop\projects\3d_Construction\IMG_3_SFM'
output_file = os.path.join(output_folder, 'camera_poses.npy')

# Ensure output folder exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Create SIFT detector and BFMatcher
sift = cv2.SIFT_create()
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# Function to perform camera pose estimation
def estimate_camera_poses(images_folder):
    images = []
    keypoints = []
    descriptors = []

    # Load images and detect features
    for filename in os.listdir(images_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.webp')):
            filepath = os.path.join(images_folder, filename)
            img = cv2.imread(filepath)
            if img is None:
                print(f'Failed to load image: {filepath}')
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            kp, des = sift.detectAndCompute(gray, None)
            images.append(img)
            keypoints.append(kp)
            descriptors.append(des)
            print(f'Processed {filename}, found {len(kp)} keypoints')

    # Estimate camera poses
    camera_poses = []
    for i in range(len(images) - 1):
        img1 = images[i]
        img2 = images[i + 1]
        kp1 = keypoints[i]
        kp2 = keypoints[i + 1]
        des1 = descriptors[i]
        des2 = descriptors[i + 1]

        # Match descriptors
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        # Get matched keypoints
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        if len(pts1) == 0 or len(pts2) == 0:
            continue

        # Estimate essential matrix using RANSAC
        E, mask = cv2.findEssentialMat(pts1, pts2, method=cv2.RANSAC, prob=0.999, threshold=1.0)

        # Recover the relative camera pose (rotation and translation)
        _, R, t, mask = cv2.recoverPose(E, pts1, pts2)

        # Store the camera pose (rotation matrix and translation vector)
        camera_pose = {
            'rotation': R,
            'translation': t
        }
        camera_poses.append(camera_pose)

    return camera_poses

# Perform camera pose estimation
camera_poses = estimate_camera_poses(input_folder)

# Save camera poses to .npy file
np.save(output_file, camera_poses)

print(f'Camera poses saved to {output_file}')

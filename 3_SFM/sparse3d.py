import cv2
import numpy as np
import os

# Define input folders and files
input_folder = r'C:\Users\csyas\OneDrive\Desktop\projects\3D_SFM\IMG_2'
camera_poses_file = r'C:\Users\csyas\OneDrive\Desktop\projects\3D_SFM\3_SFM\camera_poses.npy'
output_file = r'C:\Users\csyas\OneDrive\Desktop\projects\3D_SFM\3_SFM\sparse.npy'

# Load camera poses
camera_poses = np.load(camera_poses_file, allow_pickle=True)

# Create SIFT detector and BFMatcher
sift = cv2.SIFT_create()
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# Function to perform triangulation
def triangulate_points(images_folder, camera_poses):
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

    points_3D = []
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

        # Get camera poses for the two images
        pose1 = camera_poses[i]
        pose2 = camera_poses[i + 1]
        R1, t1 = pose1['rotation'], pose1['translation']
        R2, t2 = pose2['rotation'], pose2['translation']

        # Create projection matrices
        P1 = np.hstack((R1, t1))
        P2 = np.hstack((R2, t2))

        # Triangulate points to get 3D coordinates
        pts4D_hom = cv2.triangulatePoints(P1, P2, pts1, pts2)
        pts4D = pts4D_hom / pts4D_hom[3]

        # Convert points to 3D
        points_3D.extend(pts4D[:3].T.tolist())

    return np.array(points_3D)

# Perform triangulation
sparse_point_cloud = triangulate_points(input_folder, camera_poses)

# Save sparse point cloud to file
np.save(output_file, sparse_point_cloud)

print(f'Sparse point cloud saved: {output_file}')

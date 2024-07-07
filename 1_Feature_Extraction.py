import cv2
import os

# Define the directory containing the images
input_folder = r'C:\Users\csyas\OneDrive\Desktop\projects\3D_SFM\images'
output_folder = r'C:\Users\csyas\OneDrive\Desktop\projects\3D_SFM\IMG_1_featExtraction_Output'
# Ensure the output folder exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Create a SIFT detector
sift = cv2.SIFT_create()

# Loop through all files in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif',)):
        # Construct the full file path
        filepath = os.path.join(input_folder, filename)

        # Debug: Print the file path
        print(f'Processing file: {filepath}')

        # Load the image
        img = cv2.imread(filepath)

        # Debug: Check if the image is loaded
        if img is None:
            print(f'Failed to load image: {filepath}')
            continue

        # Convert the image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect keypoints and compute descriptors
        kp, des = sift.detectAndCompute(gray, None)

        # Debug: Print number of keypoints detected
        print(f'Number of keypoints detected: {len(kp)}')

        # Draw keypoints on the image
        img_with_keypoints = cv2.drawKeypoints(gray, kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # Save the image with keypoints
        output_path = os.path.join(output_folder, 'keypoints_' + filename)
        cv2.imwrite(output_path, img_with_keypoints)

        # Debug: Confirm saving the image
        if os.path.exists(output_path):
            print(f'Successfully saved: {output_path}')
        else:
            print(f'Failed to save: {output_path}')

print('Feature extraction completed!')

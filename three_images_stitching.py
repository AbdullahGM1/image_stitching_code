import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load the images
ref_image = cv2.imread('/Users/taz/Documents/GitHub/image_stitching/Examples_Images/office1.jpg', 0)  # Reference image
center_image = cv2.imread('/Users/taz/Documents/GitHub/image_stitching/Examples_Images/office2.jpg', 0)  # Center image
right_image = cv2.imread('/Users/taz/Documents/GitHub/image_stitching/Examples_Images/office3.jpg', 0)  # Right image

def H_out(img1, img2):
    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Compute keypoints and descriptors for both images
    kp_ref, des_ref = sift.detectAndCompute(img1, None)
    kp_right, des_right = sift.detectAndCompute(img2, None)

    # Match descriptors using BFMatcher
    bf = cv2.BFMatcher()
    matches = bf.match(des_right, des_ref)
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract the locations of the good matches
    pts_right = np.float32([kp_right[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts_ref_right = np.float32([kp_ref[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Compute homography matrix
    H, _ = cv2.findHomography(pts_right, pts_ref_right, cv2.RANSAC)

    return H

def warp(img1, img2, H):
    # Get dimensions of the reference and center images
    h_img1, w_img1 = img1.shape
    h_img2, w_img2 = img2.shape

    # Create an output canvas large enough to fit the stitched images
    output_width = w_img1 + w_img2
    output_height = max(h_img1, h_img2)
    output_image = np.zeros((output_height, output_width), dtype=np.uint8)

    # Place the reference image on the canvas
    output_image[:h_img1, :w_img1] = img1

    # Warp the center image using the homography matrix
    warped_image = cv2.warpPerspective(img2, H, (output_width, output_height))

    # Blend the warped center image with the reference image
    stitched_image = np.maximum(output_image, warped_image)

    return stitched_image

# Homography Matrix Function
H = H_out(ref_image, center_image)

# Warp Function
stitched_image = warp(ref_image, center_image, H)

# Homography Matrix Function
H = H_out(stitched_image, right_image)

# Warp Function
stitched_image = warp(stitched_image, right_image, H)

# Show the final stitched image
plt.imshow(stitched_image, cmap='gray')
plt.title('Stitched Image')
plt.axis('off')
plt.show()

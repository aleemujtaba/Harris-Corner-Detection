import cv2
import numpy as np

def calculate_gradients(image):
    Ix = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    return Ix, Iy

def calculate_corner_response(Ix, Iy, k=0.04):
    IxIx = Ix ** 2
    IyIy = Iy ** 2
    IxIy = Ix * Iy

    detM = IxIx * IyIy - IxIy ** 2
    traceM = IxIx + IyIy
    R = detM - k * (traceM ** 2)
    return R

def detect_corners(R, threshold=0.25):
    R_max = cv2.dilate(R, None)
    R_nonmax = cv2.compare(R, R_max, cv2.CMP_EQ)
    corners = np.where(R_nonmax > threshold)
    return corners

def mark_corners_on_image(image, corners):
    image_with_corners = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    image_with_corners[corners[0], corners[1]] = [0, 0, 255]  # Red-plus signs for corners
    return image_with_corners

def main(image_path, save_path):
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print("Error: Image not loaded successfully.")
        return

    # Calculate gradients Ix and Iy
    Ix, Iy = calculate_gradients(image)

    # Calculate IxIx, IyIy, and IxIy
    IxIx = Ix ** 2
    IyIy = Iy ** 2
    IxIy = Ix * Iy

    # Save images of Ix, Iy, IxIx, IyIy, and IxIy
    cv2.imwrite(save_path + 'Ix.jpg', Ix)
    cv2.imwrite(save_path + 'Iy.jpg', Iy)
    cv2.imwrite(save_path + 'IxIx.jpg', IxIx)
    cv2.imwrite(save_path + 'IyIy.jpg', IyIy)
    cv2.imwrite(save_path + 'IxIy.jpg', IxIy)

    # Calculate the Harris Corner Response (R) matrix
    R = calculate_corner_response(Ix, Iy)

    # Calculate the size of the R matrix
    R_height, R_width = R.shape

    # Print the sizes
    print(f"Size of the R matrix: {R_width} x {R_height}")

    # Save the R matrix before non-maximum suppression
    cv2.imwrite(save_path + 'R_before.jpg', R)

    # Detect corners
    corners = detect_corners(R)

    # Overlay detected corners on the original image
    image_with_corners = mark_corners_on_image(image, corners)

    # Save the final image with detected corners
    cv2.imwrite(save_path + 'Corners_Detected.jpg', image_with_corners)

    # Apply non-maximum suppression and save the R matrix after non-maximum suppression
    R_max = cv2.dilate(R, None)
    R_nonmax = cv2.compare(R, R_max, cv2.CMP_EQ)
    R_after_nonmax = np.copy(R)
    R_after_nonmax[R_nonmax == 0] = 0  # Apply non-maximum suppression
    cv2.imwrite(save_path + 'R_after.jpg', R_after_nonmax)


if __name__ == "__main__":
    image_path = '/Users/alimujtaba/Documents/ITU/Computer Vision/Assignment 2/Img3.png'
    save_path = '/Users/alimujtaba/Documents/ITU/Computer Vision/Assignment 2/Output_Images/Img3/'  # Set the path to the directory where you want to save the images
    main(image_path, save_path)

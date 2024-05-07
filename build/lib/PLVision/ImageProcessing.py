import cv2
import numpy as np
class ImageProcessor:
    """
    A class for processing images, including undistorting, cropping, and converting images to grayscale.

    This class utilizes OpenCV functions to perform various image processing tasks, which are commonly
    required in computer vision applications.
    """

    def brightness_contrast(self, img, brightness=0, contrast=0):

        brightness = int((brightness - 0) * (255 - (-255)) / (510 - 0) + (-255))
        contrast = int((contrast - 0) * (127 - (-127)) / (254 - 0) + (-127))

        if brightness != 0:
            if brightness > 0:
                shadow = brightness
                max_val = 255
            else:
                shadow = 0
                max_val = 255 + brightness

            alpha = (max_val - shadow) / 255
            gamma = shadow
            img = cv2.addWeighted(img, alpha, img, 0, gamma)

        if contrast != 0:
            alpha = float(131 * (contrast + 127)) / (127 * (131 - contrast))
            gamma = 127 * (1 - alpha)
            img = cv2.addWeighted(img, alpha, img, 0, gamma)

        return img

    def apply_affine_transformation(self, distorted, offset_x, offset_y, rotation_angle, scale_x, scale_y):
        rows, cols = distorted.shape[:2]
        # Compute the scaling part of the transformation matrix
        scaling_matrix = np.array([[scale_x, 0], [0, scale_y]], dtype=np.float32)
        # Create the full transformation matrix combining scaling and rotation
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), rotation_angle, 1)
        M[:, :2] = M[:, :2] @ scaling_matrix  # Apply scaling
        # adjust the translation part of M here
        M[:, 2] += np.array([offset_x, offset_y])
        # Applying the affine transformation
        affine_transformed = cv2.warpAffine(distorted, M, (cols, rows))
        return affine_transformed

    def undistorted_image(self, image, mtx, dist,imageWidth=1920,imageHeight=1080,crop=False):
        if image is None:
            raise ValueError("Image can not be None")

        """
        Undistorts an image given the camera matrix and distortion coefficients.

        Parameters:
            crop:
            image (np.ndarray): The distorted image to be undistorted.
            mtx (np.ndarray): The camera matrix.
            dist (np.ndarray): The distortion coefficients.

        Returns:
            np.ndarray: The undistorted image.

        Raises:
            Exception: If either `mtx` or `dist` is None.
        """
        if mtx is None or dist is None:
            raise Exception("Matrix and Dist is none")

        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (imageWidth, imageHeight), 0.5,
                                                          (imageWidth, imageHeight))

        distorted = cv2.undistort(image, mtx, dist, None, newcameramtx)

        # crop the image
        if crop is True:
            x, y, w, h = roi
            dst = distorted[y:y + h, x:x + w]
            return dst

        return distorted

    def crop_image(self, image, left_crop=0, right_crop=0, top_crop=0, bottom_crop=0, pad=True):
        """
        Crops and then pads an image to maintain its original dimensions.

        Parameters:
            image (np.ndarray): The image to be cropped and padded.
            left_crop (int): The number of pixels to crop from the left.
            right_crop (int): The number of pixels to crop from the right.
            top_crop (int): The number of pixels to crop from the top.
            bottom_crop (int): The number of pixels to crop from the bottom.

        Returns:
            np.ndarray: The cropped and padded image.

        Notes:
            This function validates the crop values to ensure they are positive integers
            before performing the crop and pad operations.
        """

        height, width = image.shape[:2]
        cropped_image = image[top_crop:height - bottom_crop, left_crop:width - right_crop]
        if pad:
            padded_image = cv2.copyMakeBorder(cropped_image, top_crop, bottom_crop, left_crop, right_crop,
                                              cv2.BORDER_CONSTANT, value=[0, 0, 0])

            return padded_image

        return cropped_image

    def grayscale_image(self, image):
        """
        Converts an image to grayscale.

        Parameters:
            image (np.ndarray): The image to be converted to grayscale.

        Returns:
            np.ndarray: The grayscale image.
        """
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def zoom(self, image, scale_factor, x_offset=0, y_offset=0):
        """
        Zooms into or out of the center of the given image based on the scale factor.

        Parameters:
            image: numpy.ndarray
                Input image as a NumPy array.
            scale_factor: float
                Scale factor for zooming.
                - Values greater than 1 zoom in.
                - Values less than 1 zoom out.
            x_offset: int, optional
                Horizontal offset from the center.
            y_offset: int, optional
                Vertical offset from the center.

        Returns:
            numpy.ndarray
                Zoomed image as a NumPy array.
        """
        # Get the height and width of the image
        print("Scale factor:", scale_factor)
        height, width = image.shape[:2]
        center_x, center_y = width // 2, height // 2

        # Calculate new dimensions
        new_width = int(width / scale_factor)
        new_height = int(height / scale_factor)

        # Calculate the ROI coordinates
        start_x = max(center_x - new_width // 2, 0)
        start_y = max(center_y - new_height // 2, 0)
        end_x = min(start_x + new_width, width)
        end_y = min(start_y + new_height, height)

        # Extract ROI
        zoomed_region = image[start_y:end_y, start_x:end_x]
        print("ROI shape:", zoomed_region.shape)

        # Resize back to original dimensions
        zoomed_image = cv2.resize(zoomed_region, (width, height), interpolation=cv2.INTER_LINEAR)
        return zoomed_image

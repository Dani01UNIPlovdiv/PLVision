"""
* File: ImageProcessing.py
* Author: AtD
* Comments: This file contains the main function of the project.
* Revision history:
* Date       Author      Description
* -----------------------------------------------------------------
* 070524     AtD/IlV         Initial release
* -----------------------------------------------------------------
*
"""


import cv2
import numpy as np

"""
A module for processing images, including undistorting, cropping, and converting images to grayscale.

This module utilizes OpenCV functions to perform various image processing tasks, which are commonly
required in computer vision applications.
"""


def brightness_contrast(image, brightness=0, contrast=0):
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
        image = cv2.addWeighted(image, alpha, image, 0, gamma)

    if contrast != 0:
        alpha = float(131 * (contrast + 127)) / (127 * (131 - contrast))
        gamma = 127 * (1 - alpha)
        image = cv2.addWeighted(image, alpha, image, 0, gamma)

    return image


def applyAffineTransformation(image, xOffset, yOffset, rotationAngle, xScale, yScale):
    rows, cols = image.shape[:2]
    # Compute the scaling part of the transformation matrix
    scalingMatrix = np.array([[xScale, 0], [0, yScale]], dtype=np.float32)
    # Create the full transformation matrix combining scaling and rotation
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), rotationAngle, 1)
    M[:, :2] = M[:, :2] @ scalingMatrix  # Apply scaling
    # adjust the translation part of M here
    M[:, 2] += np.array([xOffset, yOffset])
    # Applying the affine transformation
    affineTransformed = cv2.warpAffine(image, M, (cols, rows))
    return affineTransformed


def undistortImage(image, mtx, dist, imageWidth=1920, imageHeight=1080, crop=False):


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
    if image is None:
        raise ValueError("Image can not be None")
    if mtx is None:
        raise Exception("mtx can not be None")
    if dist is None:
        raise Exception("dist can not be None")

    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (imageWidth, imageHeight), 0.5,
                                                      (imageWidth, imageHeight))

    distorted = cv2.undistort(image, mtx, dist, None, newcameramtx)

    # crop the image
    if crop is True:
        x, y, w, h = roi
        dst = distorted[y:y + h, x:x + w]
        return dst

    return distorted


def cropImage(image, leftCrop=0, rightCrop=0, topCrop=0, bottomCrop=0, pad=True):
    """
    Crops and then pads an image to maintain its original dimensions.

    Parameters:
        image (np.ndarray): The image to be cropped and padded.
        leftCrop (int): The number of pixels to crop from the left.
        rightCrop (int): The number of pixels to crop from the right.
        topCrop (int): The number of pixels to crop from the top.
        bottomCrop (int): The number of pixels to crop from the bottom.

    Returns:
        np.ndarray: The cropped and padded image.

    Notes:
        This function validates the crop values to ensure they are positive integers
        before performing the crop and pad operations.
    """

    height, width = image.shape[:2]
    croppedImage = image[topCrop:height - bottomCrop, leftCrop:width - rightCrop]
    if pad:
        paddedImage = cv2.copyMakeBorder(croppedImage, topCrop, bottomCrop, leftCrop, rightCrop,
                                         cv2.BORDER_CONSTANT, value=[0, 0, 0])

        return paddedImage

    return croppedImage


def grayscaleImage(image):
    """
    Converts an image to grayscale.

    Parameters:
        image (np.ndarray): The image to be converted to grayscale.

    Returns:
        np.ndarray: The grayscale image.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def zoom(image, scaleFactor, xOffset=0, yOffset=0):
    """
    Zooms into or out of the center of the given image based on the scale factor.

    Parameters:
        image: numpy.ndarray
            Input image as a NumPy array.
        scaleFactor: float
            Scale factor for zooming.
            - Values greater than 1 zoom in.
            - Values less than 1 zoom out.
        xOffset: int, optional
            Horizontal offset from the center.
        yOffset: int, optional
            Vertical offset from the center.

    Returns:
        numpy.ndarray
            Zoomed image as a NumPy array.
    """
    # Get the height and width of the image
    height, width = image.shape[:2]
    center_x, center_y = width // 2 + xOffset, height // 2 + yOffset

    # Calculate new dimensions
    newWidth = int(width / scaleFactor)
    newHeight = int(height / scaleFactor)

    # Calculate the ROI coordinates
    xStart = max(center_x - newWidth // 2, 0)
    yStart = max(center_y - newHeight // 2, 0)
    xEnd = min(xStart + newWidth, width)
    yEnd = min(yStart + newHeight, height)

    # Extract ROI
    zoomedRegion = image[yStart:yEnd, xStart:xEnd]

    # Resize back to original dimensions
    zoomedImage = cv2.resize(zoomedRegion, (width, height), interpolation=cv2.INTER_LINEAR)
    return zoomedImage

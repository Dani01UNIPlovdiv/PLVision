"""
* File: Contouring.py
* Author: AtD, IlV
* Comments:
* Revision history:
* Date       Author      Description
* -----------------------------------------------------------------
** 070524     AtD/IlV         Initial release
* -----------------------------------------------------------------
*
"""
import cv2
import numpy as np


def calculateCentroid(contour):
    """
       Calculates the centroid of a contour.

       Parameters:
           contour (np.ndarray): The contour for which to calculate the centroid.

       Returns:
           tuple: A tuple containing the (x, y) coordinates of the centroid.
    """
    M = cv2.moments(contour)
    if M['m00'] == 0:
        return (0, 0)  # Avoid division by zero
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    return (cx, cy)


def scaleContourAndChildren(contours, hierarchy, ppm, mmToScale):
    """
        Scales contours and their children based on a given scaling factor.

        Parameters:
            contours (list): A list of contours.
            hierarchy (np.ndarray): Contour hierarchy as returned by cv2.findContours().
            ppm (float): Pixels per millimeter scale factor.
            mmToScale (float): Millimeters to scale by.

        Note:
            This function modifies the input contours in place.
    """
    # Calculate scale factor from mm to scale and ppm (pixels per millimeter)
    if mmToScale == 0 or ppm == 0:  # Avoid division by zero or no scaling case
        scaleFactor = 1  # No scaling
        return contours
    else:
        pixelsToScale = mmToScale * ppm  # Convert mm to pixels
        scaleFactor = 1 + (pixelsToScale / 100.0)  # Assuming you want to scale based on a percentage
        # scale_factor = pixelsToScale

    def scaleRecursive(contourIndex, parentCentroid):
        if contourIndex == -1:
            return
        if contourIndex > len(contours) - 1:
            return
        contour = contours[contourIndex]

        # Use parent centroid for scaling if available
        cx, cy = parentCentroid if parentCentroid else calculateCentroid(contour)

        # Scale each point in the contour
        for i in range(len(contour)):
            x, y = contour[i][0]
            xScaled = cx + int((x - cx) * scaleFactor)
            yScaled = cy + int((y - cy) * scaleFactor)
            contour[i][0][0] = xScaled
            contour[i][0][1] = yScaled

        # Recursively scale all child contours, passing the parent centroid down
        childIndex = hierarchy[0][contourIndex][2]  # First child
        while childIndex != -1:
            scaleRecursive(childIndex, (cx, cy))  # Keep using the same parent centroid for children
            childIndex = hierarchy[0][childIndex][0]  # Next sibling

    # Iterate over each top-level contour and scale
    for i in range(len(contours)):
        if hierarchy[0][i][3] == -1:  # If contour has no parent
            parentCentroid = calculateCentroid(contours[i])
            scaleRecursive(i, parentCentroid)


def translateContourAndChildren(contours, hierarchy, xOffset, yOffset, parentIndex=None):
    """
        Translates contours and their children by specified offsets.

        Parameters:
            contours (list): A list of contours.
            hierarchy (np.ndarray): Contour hierarchy as returned by cv2.findContours().
            xOffset (int): Horizontal offset in pixels.
            yOffset (int): Vertical offset in pixels.
            parentIndex (int): Index of the parent contour. Defaults to None.

        Returns:
            list: A list of indices of translated child contours.

        Note:
            This function modifies the input contours in place.
    """
    childIndices = [] # List to store indices of child contours

    # If parent_index is not provided, start from the root of the hierarchy
    if parentIndex is None:
        parentIndex = 0  # Assuming the first contour is the root

    # Translate the parent contour by the specified offsets
    contours[parentIndex] += (int(xOffset), int(yOffset))  # Convert offsets to integers

    # Get the first child index of the parent contour
    childIndex = hierarchy[0][parentIndex][2]

    # Recursively translate child contours and collect child indices
    while childIndex != -1: # While there are children
        childIndices.append(childIndex) # Collect child index
        childIndices.extend(translateContourAndChildren(contours, hierarchy, xOffset, yOffset, childIndex)) # Recursively translate children
        childIndex = hierarchy[0][childIndex][0]  # Get the next sibling index

    return childIndices


def drawContours(image, contours, contourColor, thickness, lineType):
    """
       Draws contours on an image.

       Parameters:
           image (np.ndarray): The input image.
           contours (list): A list of contours.
           contourColor (tuple): The color of the contours in BGR format.
           thickness (int): The thickness of the contour lines.

       Returns:
           np.ndarray: The image with contours drawn on it.

       Raises:
           ValueError: If thickness is not a positive integer.
    """

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    index = 0
    for contour in contours:
        if index == 0:
            index = 1
            continue
        cv2.drawContours(image, [contour], -1, contourColor, thickness, lineType) # Draw the contour

    return image


def findContours(image, kSize, sigmaX, threshold, maxval, type, mode, method, edged=None, kernel=None):
    """
        Finds contours in an image.

        Parameters:
            threshold:
            image (np.ndarray): The input image.

        Returns:
            tuple: A tuple containing a list of contours and their hierarchy.
    """

    # Convert the image to grayscale if it's not already
    if len(image.shape) == 3:  # Color image
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        grayImage = image  # Assuming the image is already in grayscale
    if edged is not None:
        edgedImage = cv2.Canny(grayImage, edged[0], edged[1])  # Apply Canny edge detection
    if kernel is not None:
        kernelImage = np.ones(kernel[0], np.uint8)  # Create kernel
    # Apply Gaussian blur
    bluredImage = cv2.GaussianBlur(grayImage, kSize, sigmaX)
    # Apply binary inversion thresholding
    _, binary = cv2.threshold(bluredImage, threshold, maxval, type)

    # inverted_binary = ~binary

    # Find contours
    contours, hierarchy = cv2.findContours(binary, mode, method)

    return contours, hierarchy


def rotateContour(contour, angle, pivot):
    """
    Rotates a single contour around a pivot point.

    Parameters:
        contour (np.ndarray): The contour to rotate.
        angle (float): The rotation angle in degrees.
        pivot (tuple): The pivot point (x, y) around which to rotate the contour.

    Returns:
        np.ndarray: The rotated contour.
    """
    # Convert angle from degrees to radians
    angleRad = np.deg2rad(angle)

    # Calculate the cosine and sine of the rotation angle
    cosAngle = np.cos(angleRad)
    sinAngle = np.sin(angleRad)

    # Create a new contour to hold the rotated points
    rotatedContour = np.zeros_like(contour)

    # Perform the rotation
    for i in range(contour.shape[0]):
        x, y = contour[i, 0, :]
        xRotated = cosAngle * (x - pivot[0]) - sinAngle * (y - pivot[1]) + pivot[0]
        yRotated = sinAngle * (x - pivot[0]) + cosAngle * (y - pivot[1]) + pivot[1]
        rotatedContour[i, 0, :] = [int(round(xRotated)), int(round(yRotated))]

    return rotatedContour


def rotateContourAndChildren(contours, hierarchy, angle):
    """
    Rotates contours and their children around their centroid by a given angle.

    Parameters:
        contours (list): A list of contours.
        hierarchy (np.ndarray): Contour hierarchy as returned by cv2.findContours().
        angle (float): The rotation angle in degrees.

    Note:
        This function modifies the input contours in place.
    """

    # Function to recursively rotate a contour and its children
    def rotateRecursive(contourIndex, pivot):
        if contourIndex == -1:
            return

        contour = contours[contourIndex]

        # Rotate contour
        rotated_contour = rotateContour(contour, angle, pivot)
        contours[contourIndex] = rotated_contour

        # Rotate child contours
        childIndex = hierarchy[0][contourIndex][2]  # First child
        while childIndex != -1:
            rotateRecursive(childIndex, pivot)  # Children are rotated around the same pivot
            childIndex = hierarchy[0][childIndex][0]  # Next sibling

    # Iterate over each top-level contour and rotate
    for i in range(len(contours)):
        if hierarchy[0][i][3] == -1:  # If contour has no parent
            # Calculate centroid of the top-level contour to use as a pivot
            M = cv2.moments(contours[i])
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                pivot = (cx, cy)
                rotateRecursive(i, pivot)


def isContourWithinBbox(contour, bbox):
    # Extract bounding box coordinates
    x_min, y_min, x_max, y_max = bbox

    # Iterate through contour points
    for point in contour:
        x, y = point[0]  # Extract x and y coordinates
        # Check if the point is within the bounding box
        if x < x_min or x > x_max or y < y_min or y > y_max:
            return False  # Contour point lies outside the bounding box
    return True  # All contour points lie within the bounding box

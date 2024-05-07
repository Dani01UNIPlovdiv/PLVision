"""
* File: Contouring.py
* Author: AtD, IlV
* Comments:
* Revision history:
* Date       Author      Description
* -----------------------------------------------------------------
*
* -----------------------------------------------------------------
*
"""
import cv2
import numpy as np
sss
def calculate_centroid(contour):
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
def scale_contour_and_children(contours, hierarchy, ppm, mm_to_scale):
    """
        Scales contours and their children based on a given scaling factor.

        Parameters:
            contours (list): A list of contours.
            hierarchy (np.ndarray): Contour hierarchy as returned by cv2.findContours().
            ppm (float): Pixels per millimeter scale factor.
            mm_to_scale (float): Millimeters to scale by.

        Note:
            This function modifies the input contours in place.
    """
    # Calculate scale factor from mm to scale and ppm (pixels per millimeter)
    if mm_to_scale == 0 or ppm == 0:  # Avoid division by zero or no scaling case
        scale_factor = 1  # No scaling
        return contours
    else:
        pixels_to_scale = mm_to_scale * ppm  # Convert mm to pixels
        scale_factor = 1 + (pixels_to_scale / 100.0)  # Assuming you want to scale based on a percentage
        # scale_factor = pixels_to_scale

    def scale_recursive(contour_idx, parent_centroid):
        if contour_idx == -1:
            return
        if contour_idx > len(contours) - 1:
            return
        contour = contours[contour_idx]

        # Use parent centroid for scaling if available
        cx, cy = parent_centroid if parent_centroid else calculate_centroid(contour)

        # Scale each point in the contour
        for i in range(len(contour)):
            x, y = contour[i][0]
            x_scaled = cx + int((x - cx) * scale_factor)
            y_scaled = cy + int((y - cy) * scale_factor)
            contour[i][0][0] = x_scaled
            contour[i][0][1] = y_scaled

        # Recursively scale all child contours, passing the parent centroid down
        child_idx = hierarchy[0][contour_idx][2]  # First child
        while child_idx != -1:
            scale_recursive(child_idx, (cx, cy))  # Keep using the same parent centroid for children
            child_idx = hierarchy[0][child_idx][0]  # Next sibling

    # Iterate over each top-level contour and scale
    for i in range(len(contours)):
        if hierarchy[0][i][3] == -1:  # If contour has no parent
            parent_centroid = calculate_centroid(contours[i])
            scale_recursive(i, parent_centroid)


def translate_contour_and_children(contours, hierarchy, x_offset, y_offset, parent_index=None):
    """
        Translates contours and their children by specified offsets.

        Parameters:
            contours (list): A list of contours.
            hierarchy (np.ndarray): Contour hierarchy as returned by cv2.findContours().
            x_offset (int): Horizontal offset in pixels.
            y_offset (int): Vertical offset in pixels.
            parent_index (int): Index of the parent contour. Defaults to None.

        Returns:
            list: A list of indices of translated child contours.

        Note:
            This function modifies the input contours in place.
    """
    child_indices = []

    # If parent_index is not provided, start from the root of the hierarchy
    if parent_index is None:
        parent_index = 0  # Assuming the first contour is the root

    # Translate the parent contour by the specified offsets
    contours[parent_index] += (int(x_offset), int(y_offset))  # Convert offsets to integers

    # Get the first child index of the parent contour
    child_index = hierarchy[0][parent_index][2]

    # Recursively translate child contours and collect child indices
    while child_index != -1:
        child_indices.append(child_index)
        child_indices.extend(translate_contour_and_children(contours, hierarchy, x_offset, y_offset, child_index))
        child_index = hierarchy[0][child_index][0]  # Get the next sibling index

    return child_indices


def draw_contours(image, contours, contour_color, thickness):
    """
       Draws contours on an image.

       Parameters:
           area_filter:
           image (np.ndarray): The input image.
           contours (list): A list of contours.
           contour_color (tuple): The color of the contours in BGR format.
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
        # area = cv2.contourArea(contour)
        # if area > 100:
        # Draw contours
        cv2.drawContours(image, [contour], -1, contour_color, thickness,cv2.LINE_AA)

    return image


# def draw_contours(image, contours, contour_thickness):
#     """
#     Draws contours and fills them with their dominant color on an image.
#
#     Parameters:
#         image (np.ndarray): The input image.
#         contours (list): A list of contours.
#         contour_thickness (int): The thickness of the contour lines.
#
#     Returns:
#         np.ndarray: The image with contours filled with their dominant color drawn on it.
#     """
#     for contour in contours:
#         # area = cv2.contourArea(contour)
#         # if area < 100:  # Skip contours with area less than 100
#         #     continue
#
#         # Create a mask for the current contour
#         mask = np.zeros_like(image)
#         cv2.drawContours(mask, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)
#
#         # Extract pixels within the contour using the mask
#         masked_image = cv2.bitwise_and(image, mask)
#
#         # Calculate the dominant color within the contour
#         # Here you can implement your own logic to find the dominant color
#         # For example, you can use k-means clustering or other color quantization techniques
#         # dominant_color = find_dominant_color(masked_image)
#
#         # Fill the contour with its dominant color
#         cv2.drawContours(image, [contour], -1, dominant_color, thickness=cv2.FILLED)
#
#         # Draw contours
#         cv2.drawContours(image, [contour], -1, (255, 0, 0), contour_thickness)

    # return image
#
# def find_dominant_color(image):
#     """
#     Finds the dominant color within an image.
#
#     Parameters:
#         image (np.ndarray): The input image.
#
#     Returns:
#         tuple: The dominant color in BGR format.
#     """
#     # Here you can implement your own logic to find the dominant color
#     # For example, you can use k-means clustering or other color quantization techniques
#     # For simplicity, let's calculate the mean color value
#     mean_color = np.mean(image, axis=(0, 1))
#     mean_color_int = tuple(map(int, mean_color))
#     # return tuple(mean_color.astype(int))
#     return mean_color


def find_contours(image, threshold):
    """
        Finds contours in an image.

        Parameters:
            threshold:
            image (np.ndarray): The input image.

        Returns:
            tuple: A tuple containing a list of contours and their hierarchy.
    """

    # cv2.imshow('image', image)
    # cv2.waitKey(0)
    # Convert the image to grayscale if it's not already
    if len(image.shape) == 3:  # Color image
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image  # Assuming the image is already in grayscale

    # Apply Gaussian blur
    blured_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    # cv2.imshow("Blured Image", blured_image)
    # cv2.waitKey(0)

    # Apply binary inversion thresholding
    _, binary = cv2.threshold(blured_image, threshold, 255, cv2.THRESH_OTSU)
    # cv2.imshow("binary", binary)
    # cv2.waitKey(0)

    # inverted_binary = ~binary
    # cv2.imshow("inverted_binary", inverted_binary)
    # cv2.waitKey(0)

    # Find contours
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
    # contours, hierarchy = cv2.findContours(inverted_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return contours, hierarchy


def rotate_contour(contour, angle, pivot):
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
    angle_rad = np.deg2rad(angle)

    # Calculate the cosine and sine of the rotation angle
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)

    # Create a new contour to hold the rotated points
    rotated_contour = np.zeros_like(contour)

    # Perform the rotation
    for i in range(contour.shape[0]):
        x, y = contour[i, 0, :]
        x_rotated = cos_angle * (x - pivot[0]) - sin_angle * (y - pivot[1]) + pivot[0]
        y_rotated = sin_angle * (x - pivot[0]) + cos_angle * (y - pivot[1]) + pivot[1]
        rotated_contour[i, 0, :] = [int(round(x_rotated)), int(round(y_rotated))]

    return rotated_contour


def rotate_contour_and_children(contours, hierarchy, angle):
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
    def rotate_recursive(contour_idx, pivot):
        if contour_idx == -1:
            return

        contour = contours[contour_idx]

        # Rotate contour
        rotated_contour = rotate_contour(contour, angle, pivot)
        contours[contour_idx] = rotated_contour

        # Rotate child contours
        child_idx = hierarchy[0][contour_idx][2]  # First child
        while child_idx != -1:
            rotate_recursive(child_idx, pivot)  # Children are rotated around the same pivot
            child_idx = hierarchy[0][child_idx][0]  # Next sibling

    # Iterate over each top-level contour and rotate
    for i in range(len(contours)):
        if hierarchy[0][i][3] == -1:  # If contour has no parent
            # Calculate centroid of the top-level contour to use as a pivot
            M = cv2.moments(contours[i])
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                pivot = (cx, cy)
                rotate_recursive(i, pivot)
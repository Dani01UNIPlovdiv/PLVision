"""
* File: arucoModule.py
* Author: IlV
* Comments:
* Revision history:
* Date       Author      Description
* -----------------------------------------------------------------
** 140524     IlV         Initial release
* -----------------------------------------------------------------
*
"""

import cv2
from enum import Enum


class ArucoDictionary(Enum):
    """
    An enumeration class to represent the ArUco dictionaries available in OpenCV.
    """

    DICT_4X4_50 = cv2.aruco.DICT_4X4_50
    DICT_4X4_100 = cv2.aruco.DICT_4X4_100
    DICT_4X4_250 = cv2.aruco.DICT_4X4_250
    DICT_4X4_1000 = cv2.aruco.DICT_4X4_1000
    DICT_5X5_50 = cv2.aruco.DICT_5X5_50
    DICT_5X5_100 = cv2.aruco.DICT_5X5_100
    DICT_5X5_250 = cv2.aruco.DICT_5X5_250
    DICT_5X5_1000 = cv2.aruco.DICT_5X5_1000
    DICT_6X6_50 = cv2.aruco.DICT_6X6_50
    DICT_6X6_100 = cv2.aruco.DICT_6X6_100
    DICT_6X6_250 = cv2.aruco.DICT_6X6_250
    DICT_6X6_1000 = cv2.aruco.DICT_6X6_1000
    DICT_7X7_50 = cv2.aruco.DICT_7X7_50
    DICT_7X7_100 = cv2.aruco.DICT_7X7_100
    DICT_7X7_250 = cv2.aruco.DICT_7X7_250
    DICT_7X7_1000 = cv2.aruco.DICT_7X7_1000


class ArucoDetector:
    """
    A class used to detect ArUco markers in an image using a specified ArUco dictionary and detection parameters.

    Attributes
    ----------
    arucoDict : cv2.aruco_Dictionary
        The ArUco dictionary to be used for marker detection.
    parameters : cv2.aruco_DetectorParameters
        The parameters for the ArUco marker detection process.
    arucoIds : list
        The IDs of the ArUco markers to be detected.

    Methods
    -------
    detect(image)
        Detects the specified ArUco markers in the given image.
    """

    def __init__(self, arucoDict=ArucoDictionary.DICT_6X6_250, parameters=cv2.aruco.DetectorParameters()):
        """
        Constructs all the necessary attributes for the ArucoDetector object.

        Parameters:
        arucoDict : cv2.aruco_Dictionary
            The ArUco dictionary to be used for marker detection.
        parameters : cv2.aruco_DetectorParameters
            The parameters for the ArUco marker detection process.
        arucoIds : list
            The IDs of the ArUco markers to be detected.
        """
        self.arucoDict = arucoDict
        self.parameters = parameters

    def detect_all(self, image):
        """
        Detects the specified ArUco markers in the given image.

        Parameters:
        image (np.ndarray): The image in which to detect ArUco markers.

        Returns:
        list, list:
            A list of corners of the detected ArUco markers, and a list of their IDs.
            Returns (None, None) if no markers are detected.
        """
        # Get the predefined dictionary for the ArUco markers
        aruco_dict = cv2.aruco.getPredefinedDictionary(self.arucoDict.value)

        # Detect the markers in the image
        corners, ids, _ = cv2.aruco.detectMarkers(image, aruco_dict, parameters=self.parameters)

        # If no markers are detected, return None, None
        if ids is None:
            return None, None

        # Return the corners and IDs of the detected markers
        return corners, ids

    def detect_area_corners(self, image, arucoIds, maxAttempts=10):
        """
        Detects the specified ArUco markers in the given image.

        Parameters:
        image (np.ndarray): The image in which to detect ArUco markers.

        Returns:
        list, list:
            A list of corners of the detected ArUco markers, and a list of their IDs.
            Returns (None, None) if no markers are detected.
        """

        if len(arucoIds) != 4:
            print("Error: The number of ArUco markers must be 4.")
            return None, None

        # Get the predefined dictionary for the ArUco markers
        aruco_dict = cv2.aruco.getPredefinedDictionary(self.arucoDict.value)

        # Initialize the number of found markers to 0
        found_markers = 0

        # Initialize the IDs and bounding boxes of the detected markers to None
        ids = None
        bboxes = None

        # Initialize the corners of the detected markers to None
        corners = [None, None, None, None]

        # Start the detection loop
        while True:
            # Try to detect the markers until the maximum number of attempts is reached
            # or all the specified markers are found
            while maxAttempts > 0 and found_markers < len(arucoIds):
                # Detect the markers in the image
                detection_results = cv2.aruco.detectMarkers(image, aruco_dict, parameters=self.parameters)
                bboxes, ids, _ = detection_results

                # If no markers are detected, continue to the next attempt
                if ids is None:
                    continue

                # If some markers are detected, check if they are the specified ones
                for idx in ids:
                    if idx in arucoIds:
                        # If a specified marker is found, increment the number of found markers
                        found_markers += 1
                        print("Marker found", idx)

                # Decrement the number of remaining attempts
                maxAttempts -= 1

            # Check if all the specified markers are detected
            all_detected = True
            for idx in arucoIds:
                if idx not in ids:
                    all_detected = False

            # If all the specified markers are detected, return their corners and IDs
            if ids is not None and all_detected is True:
                # Match the detected markers with their respective IDs
                for bbox, marker_id in zip(bboxes, ids):
                    if marker_id[0] == arucoIds[0]:  # Top left marker ID
                        corners[0] = bbox[0][0]

                    elif marker_id[0] == arucoIds[1]:  # Top right marker ID
                        corners[1] = bbox[0][0]

                    elif marker_id[0] == arucoIds[2]:  # Bottom right marker ID
                        corners[2] = bbox[0][0]

                    elif marker_id[0] == arucoIds[3]:  # Bottom left marker ID
                        corners[3] = bbox[0][0]

                # Return the corners and IDs of the detected markers
                return [corners[0], corners[1], corners[2], corners[3]], ids
            else:
                # If not all the specified markers are detected, print a message and return None
                print("not detected")
                return None, None

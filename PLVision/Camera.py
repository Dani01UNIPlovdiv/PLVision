"""
* File: Camera.py
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
import numpy as np  # Import numpy
import time


class Camera:
    """
    A class to represent a camera capture object.

    Attributes:
        cameraIndex (int): The index of the camera.
        width (int): The width of the camera feed.
        height (int): The height of the camera feed.
        cap (cv2.VideoCapture): OpenCV video capture object.

    Methods:
        init_cap(camera_index, height, width): Initializes the camera capture object.
        set_camera_index(value): Sets the camera index after validation.
        set_width(value): Sets the width of the camera feed after validation.
        set_height(value): Sets the height of the camera feed after validation.
        get_frame_size(): Returns the frame size as a tuple.
    """

    def __init__(self, cameraIndex, width, height):
        """
        Constructs all the necessary attributes for the camera object.

        Parameters:
            cameraIndex (int): The index of the camera.
            width (int): The width of the camera feed.
            height (int): The height of the camera feed.
        """
        self.cameraIndex = cameraIndex
        self.width = width
        self.height = height

        self.cap = cv2.VideoCapture(self.cameraIndex)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*"MJPG"))

    def capture(self):
        """
           Captures a single frame from the camera feed.

           Returns:
               np.ndarray: The captured frame as a NumPy array, or None if the frame could not be captured.
           """
        ret, frame = self.cap.read()
        return frame

    def stopCapture(self):
        self.cap.release()

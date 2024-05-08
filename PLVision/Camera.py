"""
* File: Camera.py
* Author: AtD, IlV
* Comments:
* Revision history:
* Date       Author      Description
* -----------------------------------------------------------------
*
* -----------------------------------------------------------------
*
"""
import datetime  # Import datetime
import os
import threading
import tkinter  # Import tkinter
from tkinter import Tk  # Import tkinter
import cv2
import numpy as np  # Import numpy

class Camera:
    """
    A class to represent a camera capture object.

    Attributes:
        camera_index (int): The index of the camera.
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

    def __init__(self, camera_index, width, height):
        """
        Constructs all the necessary attributes for the camera object.

        Parameters:
            camera_index (int): The index of the camera.
            width (int): The width of the camera feed.
            height (int): The height of the camera feed.
        """
        self.set_camera_index(camera_index)
        self.set_width(width)
        self.set_height(height)

        self.cap = self.init_cap(camera_index, height, width)

    def init_cap(self, camera_index, height, width):
        """
        Initializes the camera capture object with specified index, width, and height.

        Parameters:
            camera_index (int): The index of the camera.
            height (int): The height to set for the camera feed.
            width (int): The width to set for the camera feed.

        Returns:
            cv2.VideoCapture: The initialized camera capture object.
        """
        cap = cv2.VideoCapture(camera_index)
        cap.set(3, width)  # cv2.CAP_PROP_FRAME_WIDTH
        cap.set(4, height)  # cv2.CAP_PROP_FRAME_HEIGHT
        return cap

    def set_camera_index(self, value):
        """
        Validates and sets the camera index.

        Parameters:
            value (int): The index of the camera to be set.
        """
        self.camera_index = value

    def set_width(self, value):
        """
        Validates and sets the width of the camera feed.

        Parameters:
            value (int): The width to be set for the camera feed.
        """
        self.width = value

    def set_height(self, value):
        """
        Validates and sets the height of the camera feed.

        Parameters:
            value (int): The height to be set for the camera feed.
        """
        self.height = value

    def get_frame_size(self):
        """
        Returns the current frame size.

        Returns:
            tuple: The width and height of the camera feed.
        """
        return self.width, self.height
    def capture(self):
        """
        Captures a frame from the camera feed.

        Returns:
            tuple: A tuple containing the return value and the frame.
        """
        ret,frame = self.cap.read()
        return frame

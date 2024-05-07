"""
* File: Calibration.py
* Author: AtD, IlV
* Comments: This file contains calibration functions for the camera.
* Revision history:
* Date       Author      Description
* -----------------------------------------------------------------
* 070524     AtD         Initial release
* -----------------------------------------------------------------
*
"""
import cv2  # Import OpenCV
import numpy as np  # Import numpy
import pickle  # Import pickle


class CameraCalibrator:
    """
        A class for calibrating a camera using a chessboard pattern.

        Attributes:
            chessboardWidth (int): The number of inner corners per chessboard row.
            chessboardHeight (int): The number of inner corners per chessboard column.
            chessboardSquaresSize (float): The size of each chessboard square in real-world units.
            criteria (tuple): The criteria for termination of the corner sub-pixel refinement algorithm.
            objp (np.ndarray): The object points in real-world space.
            objpoints (list): A list of object points in real-world space for all images.
            imgpoints (list): A list of corner points in image space for all images.
    """

    def __init__(self, chessboardWidth, chessboardHeight, chessboardSquaresSize):
        """
            Initializes the CameraCalibrator with chessboard parameters.

            Parameters:
                chessboardWidth (int): Number of inner corners in chessboard rows.
                chessboardHeight (int): Number of inner corners in chessboard columns.
                chessboardSquaresSize (float): Size of each chessboard square.
        """
        self.chessboardWidth = chessboardWidth
        self.chessboardHeight = chessboardHeight
        self.chessboardSquaresSize = chessboardSquaresSize
        self.criteria = self._init_criteria()
        self.objp = self._calculate_objp()
        self.objpoints = []
        self.imgpoints = []

    def _calculate_objp(self):
        """
            Calculates the object points for the chessboard corners in the real-world space.

            Returns:
                np.ndarray: A numpy array of shape (N, 3) where N is the number of corners,
                            containing the (x, y, z) coordinates of each corner in real-world units.
        """
        objp = np.zeros((self.chessboardWidth * self.chessboardHeight, 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.chessboardWidth, 0:self.chessboardHeight].T.reshape(-1,
                                                                                          2) * self.chessboardSquaresSize
        return objp

    def _init_criteria(self):
        """
            Initializes the criteria for corner refinement.

            Returns:
                tuple: The criteria (type, max_iter, epsilon) for termination of the corner sub-pixel refinement algorithm.
        """
        return (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    def find_corners(self, gray):
        """
            Finds the corners in a grayscale image of the chessboard.

            Parameters:
                gray (np.ndarray): A grayscale image of the chessboard.

            Returns:
                tuple: A tuple containing a boolean value indicating if corners were successfully found,
                       and an array of detected corners in the image.
        """
        return cv2.findChessboardCorners(gray, (self.chessboardWidth, self.chessboardHeight), None)

    def refine_corners(self, gray, corners):
        """
            Refines the detected corner locations in an image to sub-pixel accuracy.

            Parameters:
                gray (np.ndarray): A grayscale image of the chessboard.
                corners (np.ndarray): Initial coordinates of the detected corners in the image.

            Returns:
                np.ndarray: The refined corner locations in the image.
        """
        # Refine corner points
        return cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

    def drawCorners(self, frame, corners2, ret):
        """
            Draws the chessboard corners detected in an image.

            Parameters:
                frame (np.ndarray): The image on which to draw the corners.
                corners2 (np.ndarray): The corner points.
                ret (bool): Indicates whether the chessboard corners were found.
        """
        cv2.drawChessboardCorners(frame, (self.chessboardWidth, self.chessboardHeight), corners2, ret)

    def appendCorners(self, corners2):
        """
            Appends the detected corners and corresponding object points for camera calibration.

            Parameters:
                corners2 (np.ndarray): The corner points in the image.
        """
        self.imgpoints.append(corners2)
        self.objpoints.append(self.objp)

    def calculatePpm(self):
        """
        Calculates the pixels-per-metric ratio based on detected corners, considering both
        x and y distances between corners for a more accurate estimation.

        Returns:
            float: The average pixels-per-metric ratio for the detected squares.
        """
        squareSizesPx = []  # This will store the average square size for each image, in pixels.

        for i, points in enumerate(self.imgpoints):
            if points.shape[0] != self.chessboardWidth * self.chessboardHeight:
                continue  # Skip if not all corners are detected.

            # Reshape points for easier manipulation
            pointsReshaped = points.reshape((self.chessboardHeight, self.chessboardWidth, 2))
            # Calculate diffs along x and y directions for each square
            diffsX = np.diff(pointsReshaped, axis=1)[:, :-1, 0]  # Exclude the last diff which has no right neighbor
            diffsY = np.diff(pointsReshaped, axis=0)[:-1, :, 1]  # Exclude the last diff which has no bottom neighbor

            # Calculate the mean square size (in pixels) for this image
            meanSquareSizePx = np.mean([np.mean(diffsX), np.mean(diffsY)])
            squareSizesPx.append(meanSquareSizePx)

        # Calculate overall mean square size in pixels and convert to PPM
        squareSizePxMean = np.mean(squareSizesPx)
        ppm = squareSizePxMean / self.chessboardSquaresSize

        return ppm

    def calibrateCamera(self, image):
        """
           Performs camera calibration using the detected corners.

           Returns:
               tuple: The distortion coefficients, camera matrix, rotation vectors,
                      and translation vectors determined during calibration.
       """
        imageShape = image.shape[::-1]  # FIXME
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, imageShape, None, None)
        return dist, mtx, rvecs, tvecs

    def calculateMeanError(self, dist, mtx, rvecs, tvecs):
        """
           Calculates the mean error of the reprojected points against the original detected corners.

           Parameters:
               dist (np.ndarray): The distortion coefficients.
               mtx (np.ndarray): The camera matrix.
               rvecs (list): List of rotation vectors.
               tvecs (list): List of translation vectors.

           Returns:
               float: The mean error across all calibration images.
       """
        meanError = 0
        for i in range(len(self.objpoints)):
            imgpoints2, _ = cv2.projectPoints(self.objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(self.imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            meanError += error
        return meanError

    def saveCalibrationData(self, mtx, dist, ppm, path):
        """
            Saves the calibration data to a file.

            Parameters:
                path:
                mtx (np.ndarray): The camera matrix.
                dist (np.ndarray): The distortion coefficients.
                ppm (float): The pixels-per-metric ratio.
        """

        np.savez(path + "camera_calibration.npz", mtx=mtx, dist=dist, ppm=ppm)

    def performCameraCalibration(self, image, path):
        """
        Perform camera calibration using the provided image.

        Parameters:
            path:
            image (np.ndarray): The input image for calibration.

        Returns:
            tuple: A tuple containing a boolean indicating if calibration was successful,
                   the calibration data (distortion coefficients, camera matrix, rotation vectors,
                   and translation vectors), and the image.
        """
        # Convert image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Initialize CameraCalibrator instance

        # Find corners in the grayscale image
        ret, corners = self.find_corners(gray)

        if not ret:
            # Corner detection failed
            return False, None, None, None

        # Refine corners for sub-pixel accuracy
        cornersRefined = self.refine_corners(gray, corners)

        # Draw refined corners on the image
        self.drawCorners(image, cornersRefined, ret)

        # Append corners for calibration
        self.appendCorners(cornersRefined)

        # Calculate pixels-per-metric ratio
        ppm = self.calculatePpm()

        # Calibrate camera
        dist, mtx, rvecs, tvecs = self.calibrateCamera(gray)

        # Calculate mean error
        meanError = self.calculateMeanError(dist, mtx, rvecs, tvecs)

        # Save calibration data
        self.saveCalibrationData(mtx, dist, ppm, path)

        # Return calibration success, calibration data, and image
        return True, (dist, mtx, rvecs, tvecs, ppm, meanError), image, corners

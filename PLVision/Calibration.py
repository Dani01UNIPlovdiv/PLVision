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
            chess_board_width (int): The number of inner corners per chessboard row.
            chess_board_height (int): The number of inner corners per chessboard column.
            chess_board_squares_size (float): The size of each chessboard square in real-world units.
            criteria (tuple): The criteria for termination of the corner sub-pixel refinement algorithm.
            objp (np.ndarray): The object points in real-world space.
            objpoints (list): A list of object points in real-world space for all images.
            imgpoints (list): A list of corner points in image space for all images.
    """

    def __init__(self, chessboard_width, chessboard_height, chessboard_squares_size):
        """
            Initializes the CameraCalibrator with chessboard parameters.

            Parameters:
                chessboard_width (int): Number of inner corners in chessboard rows.
                chessboard_height (int): Number of inner corners in chessboard columns.
                chessboard_squares_size (float): Size of each chessboard square.
        """
        self.chess_board_width = chessboard_width
        self.chess_board_height = chessboard_height
        self.chess_board_squares_size = chessboard_squares_size
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
        objp = np.zeros((self.chess_board_width * self.chess_board_height, 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.chess_board_width, 0:self.chess_board_height].T.reshape(-1,
                                                                                              2) * self.chess_board_squares_size
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
        return cv2.findChessboardCorners(gray, (self.chess_board_width, self.chess_board_height), None)

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

    def draw_corners(self, frame, corners2, ret):
        """
            Draws the chessboard corners detected in an image.

            Parameters:
                frame (np.ndarray): The image on which to draw the corners.
                corners2 (np.ndarray): The corner points.
                ret (bool): Indicates whether the chessboard corners were found.
        """
        cv2.drawChessboardCorners(frame, (self.chess_board_width, self.chess_board_height), corners2, ret)

    def append_corners(self, corners2):
        """
            Appends the detected corners and corresponding object points for camera calibration.

            Parameters:
                corners2 (np.ndarray): The corner points in the image.
        """
        self.imgpoints.append(corners2)
        self.objpoints.append(self.objp)

    def calculate_ppm(self):
        """
        Calculates the pixels-per-metric ratio based on detected corners, considering both
        x and y distances between corners for a more accurate estimation.

        Returns:
            float: The average pixels-per-metric ratio for the detected squares.
        """
        square_sizes_px = []  # This will store the average square size for each image, in pixels.

        for i, points in enumerate(self.imgpoints):
            if points.shape[0] != self.chess_board_width * self.chess_board_height:
                print(f"Skipping image {i + 1} as not all corners are detected.")
                continue  # Skip if not all corners are detected.

            # Reshape points for easier manipulation
            points_reshaped = points.reshape((self.chess_board_height, self.chess_board_width, 2))
            print("points_reshaped", points_reshaped)
            # Calculate diffs along x and y directions for each square
            diffs_x = np.diff(points_reshaped, axis=1)[:, :-1, 0]  # Exclude the last diff which has no right neighbor
            diffs_y = np.diff(points_reshaped, axis=0)[:-1, :, 1]  # Exclude the last diff which has no bottom neighbor

            # Debugging prints
            print(f"Image {i + 1} - Diffs X: {diffs_x}")
            print(f"Image {i + 1} - Diffs Y: {diffs_y}")

            # Calculate the mean square size (in pixels) for this image
            mean_square_size_px = np.mean([np.mean(diffs_x), np.mean(diffs_y)])
            print(f"Image {i + 1} - Mean Square Size (pixels): {mean_square_size_px}")
            square_sizes_px.append(mean_square_size_px)

        # Debugging print
        print("Square Sizes (pixels) for all images:", square_sizes_px)

        # Calculate overall mean square size in pixels and convert to PPM
        square_size_px_mean = np.mean(square_sizes_px)
        ppm = square_size_px_mean / self.chess_board_squares_size

        # Debugging print
        print("Overall Mean Square Size (pixels):", square_size_px_mean)
        print("Pixels-per-metric (PPM):", ppm)

        return ppm

    def calibrate_camera(self, image):
        """
           Performs camera calibration using the detected corners.

           Returns:
               tuple: The distortion coefficients, camera matrix, rotation vectors,
                      and translation vectors determined during calibration.
       """
        image_shape = image.shape[::-1]  # FIXME
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, image_shape, None, None)
        return dist, mtx, rvecs, tvecs

    def calculate_mean_error(self, dist, mtx, rvecs, tvecs):
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
        mean_error = 0
        for i in range(len(self.objpoints)):
            imgpoints2, _ = cv2.projectPoints(self.objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(self.imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            mean_error += error
        return mean_error

    def save_calibration_data(self, mtx, dist, ppm, projector=False):
        """
            Saves the calibration data to a file.

            Parameters:
                mtx (np.ndarray): The camera matrix.
                dist (np.ndarray): The distortion coefficients.
                ppm (float): The pixels-per-metric ratio.
        """
        if projector is True:
            np.savez('storage/calibration/projector_calibration.npz', mtx=mtx, dist=dist, ppm=ppm)
        else:
            np.savez('storage/calibration/camera_calibration.npz', mtx=mtx, dist=dist, ppm=ppm)

    def perform_camera_calibration(self, image, projector=False):
        """
        Perform camera calibration using the provided image.

        Parameters:
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
            print("Detection failed")
            return False, None, None, None

        # Refine corners for sub-pixel accuracy
        corners_refined = self.refine_corners(gray, corners)

        # Draw refined corners on the image
        self.draw_corners(image, corners_refined, ret)

        # Append corners for calibration
        self.append_corners(corners_refined)

        # Calculate pixels-per-metric ratio
        ppm = self.calculate_ppm()

        # Calibrate camera
        dist, mtx, rvecs, tvecs = self.calibrate_camera(gray)

        # Calculate mean error
        mean_error = self.calculate_mean_error(dist, mtx, rvecs, tvecs)

        # Save calibration data
        self.save_calibration_data(mtx, dist, ppm, projector=projector)

        # Return calibration success, calibration data, and image
        return True, (dist, mtx, rvecs, tvecs, ppm, mean_error), image, corners
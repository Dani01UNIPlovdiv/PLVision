"""
* File: test_Calibration.py
* Author: IlV
* Comments:
* Revision history:
* Date       Author      Description
* -----------------------------------------------------------------
** 070524     IlV         Initial release
* -----------------------------------------------------------------
*
"""

import unittest

import numpy as np

from PLVision.Calibration import CameraCalibrator


class TestCameraCalibrator(unittest.TestCase):
    def setUp(self):
        """Set up the CameraCalibrator object for testing."""
        self.calibrator = CameraCalibrator(10, 15, 25)
        self.assertIsNotNone(self.calibrator)
        self.assertEqual(10, self.calibrator.chessboardWidth)
        self.assertEqual(15, self.calibrator.chessboardHeight)
        self.assertEqual(25, self.calibrator.chessboardSquaresSize)

    def test_calculateObjp(self):
        """Test if the _calculateObjp method returns a numpy array of the correct shape."""
        result = self.calibrator._calculateObjp()
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual((150, 3), result.shape)

    def test_findCorners(self):
        """Test if the findCorners method returns the correct types when given a grayscale image."""
        gray = np.zeros((100, 100), dtype=np.uint8)
        ret, corners = self.calibrator.findCorners(gray)
        self.assertIsInstance(ret, bool)
        self.assertIsNone(corners)
        self.assertFalse(ret)

    def test_refineCorners(self):
        """Test if the refineCorners method returns a numpy array of the same shape as the input corners."""
        gray = np.zeros((100, 100), dtype=np.uint8)
        corners = np.array([[[50, 50]]], dtype=np.float32)
        result = self.calibrator.refineCorners(gray, corners)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, corners.shape)

    def test_calculateMeanError(self):
        """Test if the calculateMeanError method returns a float when given valid input parameters."""
        dist = np.zeros((5, 1), dtype=np.float32)
        mtx = np.eye(3, dtype=np.float32)
        rvecs = [np.zeros((3, 1), dtype=np.float32)]
        tvecs = [np.zeros((3, 1), dtype=np.float32)]
        meanError = self.calibrator.calculateMeanError(dist, mtx, rvecs, tvecs)
        self.assertIsInstance(meanError, float)

    def test_saveCalibrationData(self):
        """Test if the saveCalibrationData method correctly creates a file in the specified directory."""
        import os
        import tempfile

        mtx = np.eye(3, dtype=np.float32)
        dist = np.zeros((5, 1), dtype=np.float32)
        ppm = 1.0
        with tempfile.TemporaryDirectory() as tmpdirname:
            self.calibrator.saveCalibrationData(mtx, dist, ppm, tmpdirname)
            self.assertTrue(os.path.exists(os.path.join(tmpdirname, "camera_calibration.npz")))

    # ... add more tests ...


if __name__ == '__main__':
    unittest.main()

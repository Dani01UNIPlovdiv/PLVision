"""
* File: FileSelector.py
* Author: AtD
* Comments: This file contains the main function of the project.
* Revision history:
* Date       Author      Description
* -----------------------------------------------------------------
* 070524     AtD/IlV         Initial release
* -----------------------------------------------------------------
*
"""
import glob  # Import glob
import os  # Import os
from datetime import datetime  # Import datetime
import cv2  # Import OpenCV
import json # Import json


def getFile(fileCount, filePath, fileFormat, all=False):
    files = glob.glob(f"{filePath}/**/*.{fileFormat}", recursive=True)

    # Check if the list of files is not empty
    if files:
        # Sort the files by creation time in descending order
        files.sort(key=os.path.getctime, reverse=True)
        # Get the most recent files
        latest_files = files[:fileCount]
    else:
        latest_files = []  # Return an empty list if the list of files is empty

    if all:
        return files
    elif fileCount == 1 and latest_files:
        return latest_files[0]  # Return the most recent file as a string
    else:
        return latest_files  # Return the most recent files as a list
def saveFile(image):
    # Get the current date and time
    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    # Create a new directory path
    dir_path = 'storage/' + current_time

    # Create the directory if it doesn't exist
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Create the filename
    filename = dir_path + '/' + current_time + '.jpg'

    # Save the image to the file
    cv2.imwrite(filename, image)
    return filename
def saveContours(contours, filename):
    # Change the extension of the filename to .json
    filename = filename.rsplit('.', 1)[0] + '.json'

    # Create a list to hold the contour data
    contours_list = []

    # Convert each contour to a list and add it to contours_list
    for contour in contours:
        contours_list.append(contour.tolist())

    # Open the file in write mode and dump the contours_list into it
    with open(filename, 'w') as f:
        json.dump(contours_list, f)
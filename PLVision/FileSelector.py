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


def getFile(fileCount, filePath, fileFormat, all=False):
    files = glob.glob(f"{filePath}/**/*.{fileFormat}", recursive=True)

    # Check if the list of files is not empty
    if files:
        latest_file = max(files, key=os.path.getctime)  # Get the latest file
    else:
        latest_file = None  # Return None or a default value if the list is empty

    if all:
        return files
    else:
        return latest_file
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
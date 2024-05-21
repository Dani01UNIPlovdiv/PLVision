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


def getFile(fileCount, filePath, fileFormat, all=False):
    files = glob.glob(f"{filePath}/*.{fileFormat}")

    # Check if the list of files is not empty
    if files:
        latest_file = max(files, key=os.path.getctime)  # Get the latest file
    else:
        latest_file = None  # Return None or a default value if the list is empty

    if all:
        return files
    else:
        return latest_file
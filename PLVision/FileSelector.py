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
    # Use glob to get all files in the directory that match the file format
    files = glob.glob(f"{filePath}/*.{fileFormat}")
    latest_file = max(files, key=os.path.getctime)  # Get the latest file
    if all:
        return files
    else:
        return latest_file
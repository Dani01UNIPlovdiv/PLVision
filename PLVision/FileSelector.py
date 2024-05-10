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


def getFile(fileCount, filePath, fileFormat, all=True):  # This function returns the most recent file
    list_of_files = glob.glob(filePath + fileFormat)  # Get a list of files
    latest_file = max(list_of_files, key=os.path.getctime)  # Get the latest file
    latest_file = latest_file.replace("storage\\", "storage/")  # Remove storage\\ from the string
    return latest_file  # Return the latest file

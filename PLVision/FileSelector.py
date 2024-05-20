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
    # Get a list of all files
    list_of_files = glob.glob(filePath + fileFormat)
    list_of_files = [file.replace("storage\\", "storage/") for file in list_of_files]

    # Sort the list of files based on creation time in descending order
    list_of_files.sort(key=os.path.getctime, reverse=True)

    if all:
        # If all is True, return all files
        return list_of_files
    else:
        # If all is False, return the first fileCount files
        return list_of_files[:fileCount]

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

    # Sort the files based on their modification time
    sorted_files = sorted(files, key=os.path.getmtime, reverse=True)

    # If all is True, return all files
    if all:
        return sorted_files

    # Otherwise, return the specified number of most recent files
    return sorted_files[:fileCount]
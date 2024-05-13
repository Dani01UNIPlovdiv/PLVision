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
    list_of_files = glob.glob(filePath + fileFormat)  # Get a list of files
    list_of_files.sort(key=os.path.getctime, reverse=True)  # Sort files by creation time in descending order

    if all:
        return [file.replace("storage\\", "storage/") for file in list_of_files]  # Return all files

    latest_files = []
    for i in range(min(fileCount, len(list_of_files))):  # Loop through the number of files specified by fileCount
        latest_files.append(list_of_files[i].replace("storage\\", "storage/"))  # Append the file to latest_files

    return latest_files  # Return the latest files
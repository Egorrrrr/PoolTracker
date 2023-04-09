import cv2
import numpy as np


def detect_objects(img):
    # Convert the image to the HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define a range of blue color in the HSV color space
    lower_blue = np.array([80, 100, 100])
    upper_blue = np.array([140, 230, 255])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    # Initialize start_row and end_row
    start_row = end_row = None

    # Scan the image horizontally for the first line
    for i in range(mask.shape[0]):
        # Get the number of white pixels in the current row
        white_pixels = np.sum(mask[i, :] == 255)
        # Calculate the percentage of white pixels
        white_percentage = white_pixels / mask.shape[1]
        # If the percentage of white pixels is greater than or equal to 0.75, set start_row to the current row and break
        if white_percentage >= 0.75:
            start_row = i
            break

    # Scan the image horizontally for the last line
    for i in range(mask.shape[0] - 1, 0, -1):
        # Get the number of white pixels in the current row
        white_pixels = np.sum(mask[i, :] == 255)
        # Calculate the percentage of white pixels
        white_percentage = white_pixels / mask.shape[1]
        # If the percentage of white pixels is greater than or equal to 0.75, set end_row to the current row and break
        if white_percentage >= 0.75:
            end_row = i
            break

    lines = []
    i = start_row
    while i < end_row:
        # Get the number of white pixels in the current row
        white_pixels = np.sum(mask[i, :] == 0)
        # Calculate the percentage of white pixels
        white_percentage = white_pixels / mask.shape[1]
        # If the percentage of white pixels is less than 0.75, add the current row to lines
        if white_percentage > 0.95:
            lines.append(i)
            i += 20
            continue
        i += 1

    # Draw lines within the rectangle
    #
    return lines, start_row, end_row

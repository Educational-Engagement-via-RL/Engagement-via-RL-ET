"""
script for evaluating engagement
"""

import numpy as np
from Fixpos2Densemap import Fixpos2Densemap
from GazepointAPI import data_queue
import threading

data_store = []

def continuous_data_reader():
    global data_store 
    while True:
        data = data_queue.get()
        # print(data)
        data_store.append(data)

def start_data_reader():
    thread = threading.Thread(target=continuous_data_reader)
    thread.daemon = True  
    thread.start()

# Placeholder dimensions for the UI and areas of interest
UI_WIDTH, UI_HEIGHT = 1920, 1080

def parse_gaze_data(gaze_data_dic):
    """
    Parse the gaze data dictionary and convert it into a NumPy array.
    """
    # Initialize an empty list to store the fixation data
    fixation_data_list = []

    # Loop through each data point in the list
    for data in gaze_data_dic:
        x = float(data['FPOGX']) * UI_WIDTH
        y = float(data['FPOGY']) * UI_HEIGHT
        fixation_duration = float(data['FPOGD'])
        # Append the data as a new row in the fixation data list
        fixation_data_list.append([x, y, fixation_duration])

    # Convert the list to a NumPy array
    fixation_data = np.array(fixation_data_list)

    return fixation_data

def calculate_engagement(fixation_data, width, height):
    """
    Calculate the aggregate engagement level based on fixation data.
    """
    # Normalize and scale the fixation data
    fixation_data -= fixation_data.min(axis=0)
    fixation_data /= fixation_data.max(axis=0)
    fixation_data[:, 0] *= width
    fixation_data[:, 1] *= height

    # Generate the heatmap for the entire UI
    heatmap = Fixpos2Densemap(fixation_data, width, height, None)

    # The engagement score might be the sum of the heatmap values or a more complex function
    engagement_score = np.sum(heatmap)

    return engagement_score


def get_current_engagement_score():
    """
    Calculate and return the current engagement score based on the latest fixation data.
    """

    if not data_store:
        return None  # Return None or some default value if no data is available

    fixation_data = parse_gaze_data(data_store)

    # Filter the fixation data to exclude dictionaries where both FPOGX and FPOGY are zeros
    nonzero_data = [data_point for data_point in fixation_data if data_point['FPOGX'] != 0 or data_point['FPOGY'] != 0]

    if len(nonzero_data) > 0:
        # At least one dictionary has nonzero FPOGX or FPOGY
        real_time_engagement_score = calculate_engagement(nonzero_data, UI_WIDTH, UI_HEIGHT)
        log_engagement_score = np.log10(real_time_engagement_score)
    else:
        log_engagement_score = None  # Set to None if both FPOGX and FPOGY are zeros in all dictionaries


    data_store.clear()

    return log_engagement_score
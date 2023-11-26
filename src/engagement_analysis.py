"""
script for evaluating engagement
"""

import numpy as np
import json
from Fixpos2Densemap import Fixpos2Densemap, GaussianMask  # Import the required functions
#from mock_eye_tracker import run as mock_run
from GazepointAPI import data_queue
import threading

def continuous_data_reader():
    while True:
        data = data_queue.get()
        print(data)
        data_store.append(data)

def start_data_reader():
    thread = threading.Thread(target=continuous_data_reader)
    thread.daemon = True  
    thread.start()

# Placeholder dimensions for the UI and areas of interest
UI_WIDTH, UI_HEIGHT = 1021, 768
AREA_COUNTRY_FLAG = (0, 0, UI_WIDTH // 2, UI_HEIGHT // 2)
AREA_LOCATION = (UI_WIDTH // 2, 0, UI_WIDTH, UI_HEIGHT // 2)
AREA_DETAILS_COLORS = (0, UI_HEIGHT // 2, UI_WIDTH, UI_HEIGHT)

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

def engagement_area(fixation_data, area, width, height):
    """
    Calculate the engagement score for a specific area using the Fixpos2Densemap function.
    """
    start_x, start_y, end_x, end_y = area
    # Filter fixation data for the current area of interest
    area_data = fixation_data[(fixation_data[:, 0] >= start_x) & (fixation_data[:, 0] <= end_x) &
                              (fixation_data[:, 1] >= start_y) & (fixation_data[:, 1] <= end_y)]

    # Generate the heatmap for the specific area
    heatmap = Fixpos2Densemap(area_data, width, height, None)

    # The engagement score might be the sum of the heatmap values or a more complex function
    engagement_score = np.sum(heatmap)

    return engagement_score

def calculate_engagement(fixation_data):
    """
    Calculate the aggregate engagement level based on fixation data.
    """
    # Normalize and scale the fixation data
    fixation_data -= fixation_data.min(axis=0)
    fixation_data /= fixation_data.max(axis=0)
    fixation_data[:, 0] *= UI_WIDTH
    fixation_data[:, 1] *= UI_HEIGHT

    # Calculate the engagement score for each defined area
    engagement_country_flag = engagement_area(fixation_data, AREA_COUNTRY_FLAG, UI_WIDTH, UI_HEIGHT)
    engagement_location = engagement_area(fixation_data, AREA_LOCATION, UI_WIDTH, UI_HEIGHT)
    engagement_details_colors = engagement_area(fixation_data, AREA_DETAILS_COLORS, UI_WIDTH, UI_HEIGHT)

    # Aggregate the scores from each area
    aggregate_engagement = (engagement_country_flag + engagement_location + engagement_details_colors) / 3
    return aggregate_engagement

if __name__ == "__main__":
    # Initialize the main application for eye tracking
    # main_app = Main()
    data_store = []

    # Assuming 'Main.run()' yields real-time JSON formatted gaze data
    # Parse the real-time JSON formatted gaze data
    fixation_data = parse_gaze_data(data_store)

    # Process the real-time fixation data to calculate engagement
    real_time_engagement_score = calculate_engagement(fixation_data)
    data_store.clear()

    # Output the real-time engagement score
    print(f"Real-time Engagement Score: {real_time_engagement_score}")
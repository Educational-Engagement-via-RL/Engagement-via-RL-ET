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

def parse_gaze_data(gaze_data_json):
    """
    Parse the JSON gaze data and convert it into a NumPy array.
    """
    """
    modification needed, right now, the data is a list of dictionaries, in the form of
    data_store = [
    {"FPOGX": 0.3456, "FPOGY": 0.5678, "FPOGD": 0.1234, "FPOGID": 1, "LPUPILD": 2.3456, "RPUPILD": 2.4567},
    {"FPOGX": 0.3460, "FPOGY": 0.5680, "FPOGD": 0.1200, "FPOGID": 2, "LPUPILD": 2.3500, "RPUPILD": 2.4600},
    {"FPOGX": 0.3465, "FPOGY": 0.5685, "FPOGD": 0.1250, "FPOGID": 3, "LPUPILD": 2.3550, "RPUPILD": 2.4650},
    {"FPOGX": 0.3470, "FPOGY": 0.5690, "FPOGD": 0.1300, "FPOGID": 4, "LPUPILD": 2.3600, "RPUPILD": 2.4700},
    {"FPOGX": 0.3475, "FPOGY": 0.5695, "FPOGD": 0.1350, "FPOGID": 5, "LPUPILD": 2.3650, "RPUPILD": 2.4750},
    ...
    ]
    But feel free to modify the data structure in the script GazepointAPI.py
    """
    # Parse the JSON data
    data = json.loads(gaze_data_json)

    # Extract the relevant data and scale the X, Y coordinates to the screen dimensions
    x = float(data['FPOGX']) * UI_WIDTH
    y = float(data['FPOGY']) * UI_HEIGHT
    fixation_duration = float(data['FPOGD'])

    # Create an array of the data
    fixation_data = np.array([[x, y, fixation_duration]])

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
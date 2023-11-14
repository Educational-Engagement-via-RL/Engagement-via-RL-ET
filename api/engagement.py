######################################################################################
# Processes eye-tracking data to determine and provide user engagement levels
# Created on 2023-10-29 by Zoe Zhou
######################################################################################

from GazepointAPI import data_queue
import threading

data_store = []

def continuous_data_reader():
    while True:
        data = data_queue.get()
        print(data)
        data_store.append(data)

def start_data_reader():
    thread = threading.Thread(target=continuous_data_reader)
    thread.daemon = True  
    thread.start()

def determine_engagement():
    engagement_level = len(data_store)  # Placeholder
    data_store.clear()
    return engagement_level

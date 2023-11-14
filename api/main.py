######################################################################################
# main script
# Created on 2023-10-29 by Zoe Zhou
######################################################################################

import time
from engagement import determine_engagement, start_data_reader


start_data_reader()

while True:
    engagement_level = determine_engagement()
    print("Engagement Level:", engagement_level)
    time.sleep(5)



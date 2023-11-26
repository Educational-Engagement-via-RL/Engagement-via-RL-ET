import itertools
import pandas as pd
import numpy as np
import random
import os
import glob
from engagement_analysis import determine_engagement, start_data_reader

# similarity with scale -1 to 1
sim_high = 0.3
sim_low = -0.7
# engagement with scale 0 to 1
egm_high = 0.65
egm_low = 0.35
sim_directory = os.path.join(os.path.dirname(__file__), '..', 'data/SimilarityCsv') # __file__ represents the current file path
fam_path = os.path.join(os.path.dirname(__file__), '..', 'data/familiarity.csv')

def initialize():
    ''' Create state space with dimensions = ['Engagement', 'Familiarity', 'Similarity', 'Boring level']
        and action space with a_dimensions = ['Familiarity', 'Similarity']
        integer index
    '''
    values = ['high', 'median', 'low']
    integer_range = range(0, 21) #possible page numbers
    # Generate all possible combinations for the 4 dimensions
    state_space = list(itertools.product(values, repeat=3))
    state_space = [combination + (i,) for combination in state_space for i in integer_range] 
    a_values = ['high', 'median', 'low']
    action_space = list(itertools.product(a_values, repeat=2))
    # Create dictionaries to map between index and state space or action space
    state_to_index = {state: index for index, state in enumerate(state_space)}
    action_to_index = {action: index for index, action in enumerate(action_space)}
    index_to_state = {index: state for index, state in enumerate(state_space)}
    index_to_action = {index: action for index, action in enumerate(action_space)}

    return state_space, action_space, state_to_index, action_to_index, index_to_state, index_to_action

def familiar(flag):
    '''Read familiarity table and return familiarity level
    input param, flag: string, national flag file name
    return, familiarity level: string
    '''
    df_fam =pd.read_csv(fam_path)
    return df_fam.loc[flag, 'Familiarity']

def similar(current_flag, next_flag):
    '''Read similarity table and return similarity level
    input param
        current_flag: string, current national flag file name
        next_flag: string, next national flag file name\
    return, similarity level: string, categorical created by if-else statement
    '''
    csv_path = os.path.join(sim_directory, current_flag, '.csv')
    df_sim = pd.read_csv(csv_path)
    sim = df_sim.loc[next_flag, 'Similarity']

    if sim >= sim_high:
        return 'high'
    elif sim >= sim_low and sim < sim_high:
        return 'median'
    else:
        return 'low'

def create_flag_list():
    '''Create flag list by reading all flag files
    return, csv_file_names: list
    '''
    # Path to the directory containing CSV files, relative to the location of rl_algo.py
    csv_pattern = os.path.join(sim_directory, '*.csv') 
    # List all CSV files
    csv_files = glob.glob(csv_pattern)
    # Extract just the file names without the '.csv' extension
    csv_file_names = [os.path.splitext(os.path.basename(f))[0] for f in csv_files]
    # os.path.splitext() is used to split the file name into a name and extension tuple.
    # os.path.basename() extracts the file name from the full path.
    return csv_file_names

def decide_flag(current_flag, action, flags):
    '''Decide flag by action given
       Filter the familiarity level first, then similarity level next
    input param
        current_flag: string, current flag name
        action: tuple, (familiarity level, similarity level)
    return
        flag_chosen: string, next flag name
    '''
    (familiarity, similarity) = action

    # sort out (un)familiar flags
    df_fam = pd.read_csv(fam_path)
    df_fam = df_fam[df_fam['Image'].isin(flags)]
    df_fam = df_fam[df_fam['Familiarity'] == familiarity]
    familiar_flags = set(df_fam['Image'])
    # sort out (dis)similar flags
    sim_pattern = os.path.join(sim_directory, current_flag, '.csv')
    df_sim = pd.read_csv(sim_pattern)

    if similarity == 'high':
        flag_chosen = df_sim[df_sim['Image'].isin(familiar_flags)].sort_values(by=['Similarity']).iloc[0]['Image']
    elif similarity == 'median':
        df_sim_filter1 = df_sim[df_sim['Image'].isin(familiar_flags)].sort_values(by=['Similarity'])
        flag_chosen = df_sim_filter1.iloc[int(len(df_sim_filter1)/2)+1]['Image']
    else:
        flag_chosen = df_sim[df_sim['Image'].isin(familiar_flags)].sort_values(by=['Similarity']).iloc[-1]['Image']

    return flag_chosen

def engagement_level(engagement):
    '''Create categorical variable engagement_level by float variable engagement
    input param, engagement: float
    return, engagement_level: string
    '''
    if engagement > egm_high:
        engagement_level = 'high'
    elif engagement > egm_low and engagement <= egm_high:
        engagement_level = 'median'
    else:
        engagement_level = 'low'

    return engagement_level

# Global variables to maintain state
current_flag = None
Q = None
current_state = None
total_steps = 0
flags = None

def initialize_learning():
    global current_flag, Q, current_state, total_steps, flags
    state_space, action_space, state_to_index, action_to_index, index_to_state, index_to_action = initialize()
    Q = np.zeros([len(state_space), len(action_space)])
    flags = create_flag_list()
    current_flag = random.choice(flags)
    engagement = determine_engagement(current_flag)  # Assuming you have this function
    current_state = (engagement_level(engagement), familiar(current_flag), random.choice(state_space)[2], 0)
    total_steps = 0

def run_one_step():
    global current_flag, Q, current_state, total_steps, flags
    state_space, action_space, state_to_index, action_to_index, index_to_state, index_to_action = initialize()
    gamma = 0.95
    learnRate = 0.8
    epsilon = np.exp(-total_steps / 35.0)

    s = state_to_index.get(current_state)
    if np.random.rand() < epsilon:
        a_content = random.choice(action_space)
        a = action_to_index.get(a_content)
    else:
        a = np.argmax(Q[s, :])
        a_content = index_to_action.get(a)

    next_flag = decide_flag(current_flag, a_content, flags)
    r = determine_engagement(next_flag)
    s1_content = (engagement_level(r), familiar(next_flag), similar(current_flag, next_flag), total_steps + 1)
    s1 = state_to_index.get(s1_content)

    Q[s, a] = (1 - learnRate) * Q[s, a] + learnRate * (r + gamma * np.max(Q[s1, :]))

    current_flag = next_flag
    current_state = s1_content
    total_steps += 1

    return next_flag


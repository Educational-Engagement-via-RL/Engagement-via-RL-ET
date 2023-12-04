import itertools
import pandas as pd
import numpy as np
import random
import os
import glob
from engagement_analysis import get_current_engagement_score
import pickle

# similarity with scale -1 to 1
sim_low = -0.13 # X value for the first line (1/3 of total): -0.13180964986483257
sim_high = 0.10 # X value for the second line (2/3 of total): 0.10018077492713928

# engagement with scale 0 to 1
egm_high = 3
egm_low = 1

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sim_directory = os.path.join(parent_dir, 'data/similarity') 
fam_path = os.path.join(parent_dir, 'data/flag_familiarity.csv')
intr_path = os.path.join(parent_dir, 'data/intrinsic_scores.csv') 
df_intr = pd.read_csv(intr_path)
q_table_path = os.path.join(script_dir, 'q_table.pkl')
scores_record = []

def save_q_table(q_table, filename=q_table_path):
    """
    saving Q-table
    """
    try:
        with open(filename, 'wb') as file:
            pickle.dump(q_table, file)
    except Exception as e:
        print(f"Error saving Q-table: {e}")

def load_q_table(filename=q_table_path):
    """
    read Q-Table
    """
    if not os.path.exists(filename):
        return None
    try:
        with open(filename, 'rb') as file:
            return pickle.load(file)
    except Exception as e:
        print(f"Error loading Q-table: {e}")
        return None

def initialize():
    ''' Create state space with dimensions = ['Engagement', 'Familiarity', 'Similarity', 'Boring level']
        and action space with dimensions = ['Familiarity', 'Similarity']
    '''
    values = [1, 2, 3]  # First dimension values
    familiarity_values = [1, 2]  # Second dimension values (Familiarity)
    similarity_values = [1, 2,3]  # Third dimension values (Similarity)

    # Create the state space
    state_space = list(itertools.product(values, familiarity_values, similarity_values))

    a_values = [2, 1]  # Updated to reflect new familiarity levels
    action_space = list(itertools.product(a_values, repeat=2))

    # Create dictionaries to map between index and state space or action space
    state_to_index = {state: index for index, state in enumerate(state_space)}
    action_to_index = {action: index for index, action in enumerate(action_space)}
    index_to_state = {index: state for index, state in enumerate(state_space)}
    index_to_action = {index: action for index, action in enumerate(action_space)}

    return state_space, action_space, state_to_index, action_to_index, index_to_state, index_to_action

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

total_steps = 0
flags = create_flag_list()
state_space, action_space, state_to_index, action_to_index, index_to_state, index_to_action = initialize()
Q = np.zeros([len(state_space), len(action_space)])

def familiar(flag):
    '''Read familiarity table and return familiarity level
    input param, flag: string, national flag file name
    return, familiarity level: string
    '''
    df_fam = pd.read_csv(fam_path)
    if flag in df_fam[df_fam['Familiarity Level'] == 'Familiar']['Flag Name'].values:
        return 2
    else:
        return 1

def similar(current_flag, next_flag):
    '''Read similarity table and return similarity level
    input param
        current_flag: string, current national flag file name
        next_flag: string, next national flag file name
    return, similarity level: string, categorical created by if-else statement
    '''
    csv_path = os.path.join(sim_directory, current_flag+'.csv')
    df_sim = pd.read_csv(csv_path)
    sim = float(df_sim[df_sim["Image"] == next_flag]["Similarity"])

    if sim >= sim_high:
        return 3
    elif sim >= sim_low and sim < sim_high:
        return 2
    else:
        return 1

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
    df_fam = df_fam[df_fam['Flag Name'].isin(flags)]
    df_fam = df_fam[df_fam['Familiarity Level'] == familiarity]
    familiar_flags = set(df_fam['Flag Name'])
    # sort out (dis)similar flags
    sim_pattern = os.path.join(sim_directory, current_flag +'.csv')
    df_sim = pd.read_csv(sim_pattern)

    df_sim_filter1 = df_sim[df_sim['Image'].isin(familiar_flags)].sort_values(by=['Similarity'])
    
    if not df_sim_filter1.empty:
        if similarity == 3:
            flag_chosen = df_sim_filter1.iloc[0]['Image']
        elif similarity == 2:
            median_index = len(df_sim_filter1) // 2
            flag_chosen = df_sim_filter1.iloc[median_index]['Image']
        else:
            flag_chosen = df_sim_filter1.iloc[-1]['Image']
    else:
        print("oopsssss")
        flag_chosen = random.choice([item for item in flags if item != current_flag])

    return flag_chosen

def engagement_level(engagement):
    '''Create categorical variable engagement_level by float variable engagement
    input param, engagement: float
    return, engagement_level: string
    '''
    scores_record.append(engagement)
    if engagement > egm_high:
        engagement_level = 3
    elif engagement > egm_low and engagement <= egm_high:
        engagement_level = 2
    else:
        engagement_level = 1

    return engagement_level

# Global variables to maintain state
# current_flag = None
# current_state = None

def initialize_learning():
    global Q, current_state, total_steps, flags, current_flag
    loaded_Q = load_q_table(filename = 'q_table.pkl')
    if loaded_Q is not None:
        Q = loaded_Q
    current_flag = random.choice(flags)
    intr_norm = df_intr[df_intr['Code'] == current_flag]["Score"]
    engagement = 1 # for debugging purposes
    # comment for debugging purposes
    # engagement = get_current_engagement_score(current_flag) - intr_norm
    # write this in csv
    current_state = (engagement_level(engagement), familiar(current_flag), random.choice(state_space)[2])
    total_steps = 0
    return current_state

def run_one_step():
    global current_flag, Q, current_state, total_steps, flags
    
    gamma = 0.95
    learnRate = 0.8
    epsilon = np.exp(-total_steps / 35.0)

    # for debugging purpose:
    s = state_to_index.get(current_state)
    if s is None:
        print(f"Error: State {current_state} not found in state_to_index.")
        return None
    if np.random.rand() < epsilon:
        a_content = random.choice(action_space)
        a = action_to_index.get(a_content)
    else:
        a = np.argmax(Q[s, :])
        a_content = index_to_action.get(a)

    next_flag = decide_flag(current_flag, a_content, flags)
    # comment for debugging purposes - intr
    # r = get_current_engagement_score()
    r = 1 # for debugging
    s1_content = (engagement_level(r), familiar(next_flag), similar(current_flag, next_flag))
    s1 = state_to_index.get(s1_content)

    Q[s, a] = (1 - learnRate) * Q[s, a] + learnRate * (r + gamma * np.max(Q[s1, :]))

    current_flag = next_flag
    current_state = s1_content
    total_steps += 1

    save_q_table(Q, filename = 'q_table.pkl')

    return next_flag, current_state, r


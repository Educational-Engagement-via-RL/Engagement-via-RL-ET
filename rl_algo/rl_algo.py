#initialize episodic structure
import itertools
import pandas as pd
import numpy as np
import random
import os
import glob

high_threshold = 0.5
low_threshold = -0.5
sim_directory = os.path.join(os.path.dirname(__file__), '..', 'data/SimilarityCsv') # __file__ represents the current file path
fam_path = os.path.join(os.path.dirname(__file__), '..', 'data/familiarity.csv')

def familiar(flag):
    df_fam =pd.read_csv(fam_path)
    return df_fam.loc[flag, 'Familiarity']

def similar(current_flag, next_flag):
    csv_path = os.path.join(sim_directory, current_flag, '.csv')
    df_sim = pd.read_csv(csv_path)
    sim = df_sim.loc[next_flag, 'Similarity']

    if sim >= high_threshold:
        return 'high'
    elif sim >= low_threshold and sim < high_threshold:
        return 'median'
    else:
        return 'low'

def create_flag_list():
    # Path to the directory containing CSV files, relative to the location of algo.py
    csv_pattern = os.path.join(sim_directory, '*.csv') 
    # List all CSV files
    csv_files = glob.glob(csv_pattern)
    # Extract just the file names without the '.csv' extension
    csv_file_names = [os.path.splitext(os.path.basename(f))[0] for f in csv_files]
    # os.path.splitext() is used to split the file name into a name and extension tuple.
    # os.path.basename() extracts the file name from the full path.
    return csv_file_names

def decide_flag(current_flag, action):
    (familiarity, similarity) = action

    # sort out (dis)similar flags
    sim_pattern = os.path.join(sim_directory, current_flag, '.csv')
    df_sim = pd.read_csv(sim_pattern)

    if similarity == 'high':
        df_sim = df_sim[df_sim['Similarity'] >= high_threshold] 
    elif similarity == 'median':
        df_sim = df_sim[(df_sim['Similarity'] >= low_threshold) & (df_sim['Similarity']) < high_threshold]
    else:
        df_sim = df_sim[df_sim['Similarity'] < low_threshold]

    similar_flags = set(df_sim['Image'].head(5))

    # sort out (un)familiar flags
    df_fam = pd.read_csv(fam_path)

    df_fam = df_fam[df_fam['Familiarity'] == familiarity]
    familiar_flags = set(df_fam['Image'].head(5))

    flags_chosen = similar_flags.intersection(familiar_flags)
    if flags_chosen:
        return random.choice(flags_chosen)
    else: # to-do: what shall we do when empty set
        None

def run():
    # Define the labels for the dimensions
    # dimensions = ['Engagement', 'Familiarity', 'Similarity']
    # Possible values for each dimension
    values = ['high', 'median', 'low']
    # Generate all possible combinations for the 3 dimensions
    state_space = list(itertools.product(values, repeat=3))
    # [(engagement, familiarity, similarity), ...]

    # a_dimensions = ['Familiarity', 'Similarity']
    a_values = ['high', 'median', 'low']
    action_space = list(itertools.product(a_values, repeat=2))
    # [(familiarity, similarity), ...]

    rs = random.randint(0,len(state_space))
    ra = random.randint(0,len(action_space))
    num_episodes=1000
    gamma=0.95
    learnRate=0.8
    epsilon = 0.8
    min_explore = 0.01

    # initialization
    Q=np.zeros([state_space[rs], action_space[ra]]) #Q(s,a). The Q-values from this array will be used to evaluate your policy.
    flags = create_flag_list()
    current_flag = random.choice(flags)
    s = (engagement(current_flag), familiar(current_flag), random.choice(state_space)[2]) # call engagement function

    #execute in episodes
    for i in range(num_episodes):

        #reset the environment at the beginning of an episode       
        done = False #not done

        if np.random.rand() < epsilon:
            a = random.choice(action_space)
        else:
            a = np.argmax(Q[s, :])
        print(a) # e.g. ('high', 'low')
        next_flag = decide_flag(a)
        r = engagement_level(next_flag)
        s1 = (engagement(next_flag), familiar(next_flag), similar(current_flag, next_flag))
        # done = ?
        #     
        Q[s, a] = (1 - learnRate) * Q[s, a] + learnRate * (r + gamma * np.max(Q[s1, :]))

        #break if done, reached terminal state
        if done:
            break

        s=s1
        current_flag = next_flag

    epsilon = max(epsilon*0.999, min_explore)

if __name__ == "__main__":
    run()
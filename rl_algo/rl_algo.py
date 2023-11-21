#initialize episodic structure
import itertools
import pandas as pd
import numpy as np
import random
import os
import glob

high_threshold = 0.5
low_threshold = -0.5
egm_high = 0.65
egm_low = 0.35
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

    similar_flags = set(df_sim['Image'].head(5))

    # sort out (un)familiar flags
    df_fam = pd.read_csv(fam_path)

    df_fam = df_fam[df_fam['Familiarity'] == familiarity]
    familiar_flags = set(df_fam['Image'])

    if similarity == 'high':
        flag_chosen = df_sim[df_sim['Image'].isin(familiar_flags)].sort_values(by=['Similarity']).iloc[0]['Image']
    elif similarity == 'median':
        df_sim_filter1 = df_sim[df_sim['Image'].isin(familiar_flags)].sort_values(by=['Similarity'])
        flag_chosen = df_sim_filter1.iloc[int(len(df_sim_filter1)/2)]['Image']
    else:
        flag_chosen = df_sim[df_sim['Image'].isin(familiar_flags)].sort_values(by=['Similarity']).iloc[-1]['Image']

    return flag_chosen

def decide_engagement_level(engagement):
    if engagement > egm_high:
        engagement_level = 'high'
    elif engagement > egm_low and engagement <= egm_high:
        engagement_level = 'median'
    else:
        engagement_level = 'low'

    return engagement_level

def run():
    # Define the labels for the dimensions
    # dimensions = ['Engagement', 'Familiarity', 'Similarity', 'Boring level']
    # Possible values for each dimension
    values = ['high', 'median', 'low']
    integer_range = range(0, 21)
    # Generate all possible combinations for the 4 dimensions
    state_space = list(itertools.product(values, repeat=3))
    state_space = [combination + (i,) for combination in state_space for i in integer_range] 
    # [(engagement, familiarity, similarity, pages), ...]

    # a_dimensions = ['Familiarity', 'Similarity']
    a_values = ['high', 'median', 'low']
    action_space = list(itertools.product(a_values, repeat=2))
    # [(familiarity, similarity), ...]
    reward_sum = 0

    rs = random.randint(0,len(state_space))
    ra = random.randint(0,len(action_space))
    num_episodes=4 # number of test users
    gamma=0.95
    learnRate=0.8
    epsilon=0.8
    min_explore=0.01

    # initialization
    Q=np.array([state_space[rs], action_space[ra]]) #Q(s,a). The Q-values from this array will be used to evaluate your policy.
    flags = create_flag_list()

    #each user session is one episode
    for i in range(num_episodes):
        #reset the environment at the beginning of an episode
        current_flag = random.choice(flags)
        pages = 0
        engagement = calculate_engagement(current_flag)
        s = (decide_engagement_level(engagement), familiar(current_flag), random.choice(state_space)[2], pages)
        done = False #not done

        while not done:
            if np.random.rand() < epsilon:
                a = random.choice(action_space)
            else:
                a = np.argmax(Q[s, :])
            print(a) # e.g. ('high', 'low')
            next_flag = decide_flag(a)
            r = calculate_engagement(next_flag)
            reward_sum += r
            pages += 1
            s1 = (decide_engagement_level(r), familiar(next_flag), similar(current_flag, next_flag), pages)
        
            Q[s, a] = (1 - learnRate) * Q[s, a] + learnRate * (r + gamma * np.max(Q[s1, :]))

            #break if done, reached terminal state
            if pages == 20:
                done = True
            if done:
                break

            s=s1
            current_flag = next_flag

            epsilon = max(epsilon*0.999, min_explore)
    return reward_sum

if __name__ == "__main__":
    run()
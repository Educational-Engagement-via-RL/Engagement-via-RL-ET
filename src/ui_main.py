"""
main script
"""

import os
import random
import csv

from flask import Flask, render_template, request, redirect, url_for
import pandas as pd

from rl_algo import initialize_learning, run_one_step
from engagement_analysis import get_current_engagement_score, start_data_reader

app = Flask(__name__)

script_dir = os.path.dirname(os.path.abspath(__file__)) 
rating_dir = os.path.join(script_dir, 'static', 'rating')
parent_dir = os.path.dirname(script_dir)
intr_path = os.path.join(parent_dir, 'data/intrinsic_scores.csv') 
df_intr = pd.read_csv(intr_path)
page_num_max = 20

flags = [{"id": idx + 1, 
          "name": os.path.splitext(filename)[0],  
          "image": os.path.join('/static/rating', filename), 
          "info": f"Info about Country{idx + 1}"} 
         for idx, filename in enumerate(os.listdir(rating_dir))]

scores_record_random = []
selected_group = 'default'

def append_scores_to_csv(group_number, scores, file_path):
    # Prepare the row to be appended
    row = [group_number] + scores
    with open(file_path, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row)

@app.route('/')
def index():
    """
    selecting control / test group
    """
    return render_template('opening.html')

@app.route('/submit_group', methods=['POST'])
def submit_group():
    """
    submitting control / test group
    """
    global selected_group
    selected_group = request.form.get('group')
    return redirect(url_for('rate_flags'))

@app.route('/rate_flags')
def rate_flags():
    """
    rating familarity level
    """
    global selected_group
    return render_template('rate_flags.html', group=selected_group, flags=flags)

@app.route('/submit_ratings', methods=['POST'])
def submit_ratings():
    """
    submitting familiarity level
    """
    global selected_group, current_state
    
    flag_id_to_name = {str(flag['id']): flag['name'] for flag in flags}

    # Get the IDs of flags marked as familiar
    familiar_flag_ids = request.form.getlist('familiar')

    familiarity_data = []
    for flag in flags:
        flag_id_str = str(flag['id'])
        if flag_id_str in familiar_flag_ids:
            # Familiar flags
            familiarity_data.append([flag_id_to_name[flag_id_str], 2])
        else:
            # Unfamiliar flags
            familiarity_data.append([flag_id_to_name[flag_id_str], 1])

    df = pd.DataFrame(familiarity_data, columns=['Flag Name', 'Familiarity Level'])

    parent_dir = os.path.dirname(script_dir)
    csv_path = os.path.join(parent_dir, 'data', 'flag_familiarity.csv')
    df.to_csv(csv_path, index=False)

    start_data_reader()

    current_state = initialize_learning()

    return redirect(url_for('view_flag', page_num=1))



@app.route('/random_image')
def random_image():
    """
    control group
    """
    images_dir = os.path.join(app.static_folder, 'learningMaterial') 
    images = os.listdir(images_dir) 
    random_image = random.choice(images)  
    image_path = 'learningMaterial/' + random_image 
    return [url_for('static', filename=image_path), random_image]


@app.route('/start_learning')
def start_learning():
    """
    test group - starting the algorithm
    """
    global current_state 
    current_state = initialize_learning()
    return redirect(url_for('view_flag'))

@app.route('/view_flag')
def view_flag():
    global selected_group, current_state, scores_record_random
    page_num = int(request.args.get('page_num', 1))

    if page_num > page_num_max:
        output_dir = os.path.join(parent_dir, 'output')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        file_path = os.path.join(output_dir, 'cumulative_scores.csv')
        group_number = "control" if selected_group == 'group1' else "test"
        append_scores_to_csv(group_number, scores_record_random, file_path)
        
        scores_record_random = []  # Reset scores for the next turn
        
        return redirect(url_for('congrats'))

    elif selected_group == 'group1':  # control group
        image_url, current_flag = random_image()  # Changed to unpack a tuple returned by random_image()
        print(df_intr, current_flag.replace(".jpg",""))
        intr_norm = df_intr[df_intr['Code'] == current_flag.replace(".jpg","")]["Score"]
        engagement_score_ori = get_current_engagement_score() 
        engagement_score = float(engagement_score_ori - intr_norm)
        print("engagement score: ",engagement_score, type(engagement_score))
        scores_record_random.append(engagement_score)
    else:
        # q_table = load_q_table()
        image_name, current_state, engagement_score = run_one_step()  # test group
        image_url = url_for('static', filename='learningMaterial/' + image_name + '.jpg')
        scores_record_random.append(engagement_score)

    next_page_num = page_num + 1
    return render_template('view_flag.html', flag_image_url=image_url, page_num=next_page_num, selected_group=selected_group)


@app.route('/congrats')
def congrats():
    """
    final page
    """
    # Determine group number based on selected_group
       
    return render_template('congrats.html')

if __name__ == '__main__':
    app.run(debug=True)



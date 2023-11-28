"""
main script
"""

import os
import random

from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd

from rl_algo import initialize_learning, run_one_step
from engagement_analysis import get_current_engagement_score, start_data_reader

app = Flask(__name__)
app.secret_key = "secret_key"

script_dir = os.path.dirname(os.path.abspath(__file__)) 
rating_dir = os.path.join(script_dir, 'static', 'rating')
parent_dir = os.path.dirname(script_dir)
intr_path = os.path.join(parent_dir, 'data/intrinsic_scores.csv') 
df_intr = pd.read_csv(intr_path)

flags = [{"id": idx + 1, 
          "name": os.path.splitext(filename)[0],  
          "image": os.path.join('/static/rating', filename), 
          "info": f"Info about Country{idx + 1}"} 
         for idx, filename in enumerate(os.listdir(rating_dir))]

scores_record_random = []

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
    selected_group = request.form.get('group')
    session['selected_group'] = selected_group 
    return redirect(url_for('rate_flags'))

@app.route('/rate_flags')
def rate_flags():
    """
    rating familarity level
    """
    group = session.get('selected_group', 'default') 
    return render_template('rate_flags.html', group=group, flags=flags)

@app.route('/submit_ratings', methods=['POST'])
def submit_ratings():
    """
    submitting familarity level
    """
    flag_id_to_name = {str(flag['id']): flag['name'] for flag in flags}

    unfamiliar_flags = list(set(request.form.getlist('unfamiliar')) - set(request.form.getlist('not_sure')))
    not_sure_flags = list(set(request.form.getlist('not_sure')) - set(request.form.getlist('familiar')))
    familiar_flags = request.form.getlist('familiar')

    familiarity_data = []
    for flag_id in unfamiliar_flags:
        familiarity_data.append([flag_id_to_name[flag_id], 3])
    for flag_id in not_sure_flags:
        familiarity_data.append([flag_id_to_name[flag_id], 2])
    for flag_id in familiar_flags:
        familiarity_data.append([flag_id_to_name[flag_id], 1])

    df = pd.DataFrame(familiarity_data, columns=['Flag Name', 'Familiarity Level'])

    parent_dir = os.path.dirname(script_dir)
    csv_path = os.path.join(parent_dir, 'data', 'flag_familiarity.csv')
    df.to_csv(csv_path, index=False)

    start_data_reader()

    return redirect(url_for('view_flag', flag_id=1))

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
    initialize_learning()
    return redirect(url_for('view_flag'))

@app.route('/view_flag')
def view_flag():
    """
    viewing learning material
    """
    session['flag_count'] = session.get('flag_count', 0) + 1

    if session.get('selected_group') == 'group1':
        image_url = random_image()[0]
        current_flag = random_image()[1]
        intr_norm = df_intr[df_intr['Code'] == current_flag]["Score"]
        engagement_score = get_current_engagement_score() - intr_norm
        print('engagement score:', engagement_score)
        scores_record_random.append(engagement_score)
    else:
        image_name = run_one_step()  
        image_url = url_for('static', filename=os.path.join('learningMaterial', image_name))

    return render_template('view_flag.html', flag_image_url=image_url)


@app.route('/congrats')
def congrats():
    """
    final page
    """
    return render_template('congrats.html')

if __name__ == '__main__':
    app.run(debug=True)



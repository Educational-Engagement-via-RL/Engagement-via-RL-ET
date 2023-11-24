from flask import Flask, render_template, request, redirect, url_for, session
import os
import random

app = Flask(__name__)
app.secret_key = "secret_key"

# Get the absolute path to the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the 'static/rating' directory
rating_dir = os.path.join(script_dir, 'static', 'rating')

# Now list the files in the 'static/rating' directory
flags = [{"id": idx + 1, "name": f"Country{idx + 1}", 
          "image": os.path.join('/static/rating', filename), 
          "info": f"Info about Country{idx + 1}"} 
         for idx, filename in enumerate(os.listdir(rating_dir))]


@app.route('/congrats')
def congrats():
    return render_template('congrats.html')

@app.route('/')
def index():
    # Starting page where users rate their familiarity with flags
    return render_template('rate_flags.html', flags=flags)

@app.route('/view_flag')
def view_flag():
    # Use an absolute path for the directory
    base_dir = os.path.abspath(os.path.dirname(__file__))  # This will give you the absolute path of the current file
    image_path = os.path.join(base_dir, 'static', 'learningMaterial')  # Correctly join paths
    images = os.listdir(image_path)
    random_image = random.choice(images)
    
    # Render the view_flag.html template with the random image
    return render_template('view_flag.html', random_image_url=url_for('static', filename=f'learningMaterial/{random_image}'))


@app.route('/submit_ratings', methods=['POST'])
def submit_ratings():

    unfamiliar_flags = list(set(request.form.getlist('unfamiliar')) - set(request.form.getlist('not_sure')))
    not_sure_flags = list(set(request.form.getlist('not_sure')) - set(request.form.getlist('familiar')))
    familiar_flags = request.form.getlist('familiar')
    
    print("Familiar Flags: ", familiar_flags)
    print("Not Sure Flags: ", not_sure_flags)
    print("Unfamiliar Flags: ", unfamiliar_flags)

    return redirect(url_for('view_flag', flag_id=1))


@app.route('/random_image')
def random_image():
    images = os.listdir('static/learningMaterial')  # List all files in the directory
    random_image = random.choice(images)  # Randomly select an image
    return url_for('static', filename=f'learningMaterial/{random_image}')


if __name__ == '__main__':
    app.run(debug=True)

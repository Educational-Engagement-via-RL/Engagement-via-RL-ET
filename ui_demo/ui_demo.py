from flask import Flask, render_template, request, redirect, url_for, session

app = Flask(__name__)
app.secret_key = "secret_key"

# Mockup data for flags
flags = [
    {"id": 1, "name": "Country1", "image": "/static/flag1.png", "info": "Info about flag1"},
    {"id": 2, "name": "Country2", "image": "/static/flag2.png", "info": "Info about flag2"},
    {"id": 3, "name": "Country3", "image": "/static/flag3.png", "info": "Info about flag3"}
]


@app.route('/congrats')
def congrats():
    return render_template('congrats.html')

@app.route('/')
def index():
    # Starting page where users rate their familiarity with flags
    return render_template('rate_flags.html', flags=flags)

@app.route('/view_flag/<int:flag_id>', methods=['GET', 'POST'])
def view_flag(flag_id):
    # Show a flag along with its details (name, image, info)
    flag = next((f for f in flags if f['id'] == flag_id), None)
    return render_template('view_flag.html', flag=flag)


@app.route('/submit_ratings', methods=['POST'])
def submit_ratings():

    unfamiliar_flags = list(set(request.form.getlist('unfamiliar')) - set(request.form.getlist('not_sure')))
    not_sure_flags = list(set(request.form.getlist('not_sure')) - set(request.form.getlist('familiar')))
    familiar_flags = request.form.getlist('familiar')
    # ... (process the data as needed)
    print(familiar_flags,not_sure_flags,unfamiliar_flags)

    return redirect(url_for('view_flag', flag_id=1))

if __name__ == '__main__':
    app.run(debug=True)

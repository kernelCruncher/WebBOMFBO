 # Import flask and datetime module for showing date and time
import sys
from flask import Flask
import datetime
import os
from flask import Flask, render_template, request, redirect, url_for, abort
from werkzeug.utils import secure_filename

x = datetime.datetime.now()
 
# Initializing flask app
app = Flask(__name__)

app.config["UPLOAD_PATH"]="uploads"
app.config['UPLOAD_EXTENSIONS']=['.csv', '.txt']

# Route for seeing a data
@app.route('/data')
def get_time():
    # Returning an api for showing in  reactjs
    return {
        'Name':"geek", 
        "Age":"22",
        "Date":x, 
        "programming":"python"
        }
 
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    uploaded_file = request.files['file']
    filename = secure_filename(uploaded_file.filename)
    if filename != '':
        file_ext = os.path.splitext(filename)[1]
        if file_ext not in app.config['UPLOAD_EXTENSIONS']:
            abort(400)
        uploaded_file.save(os.path.join(app.config['UPLOAD_PATH']), filename)
    return redirect(url_for('index'))

# Running app
if __name__ == '__main__':
    app.run(debug=True)

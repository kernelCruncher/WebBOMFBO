from flask import Flask
import os
from flask import Flask, render_template, request, redirect, url_for, abort, send_from_directory
from werkzeug.utils import secure_filename
 
# Initializing flask app
app = Flask(__name__)

app.config["UPLOAD_PATH"]="uploads"
app.config['UPLOAD_EXTENSIONS']=['.csv', '.txt']

@app.route('/uploads')
def get_uploads():
    files = os.listdir(app.config["UPLOAD_PATH"])
    return files
 
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
        # print('EJ', workingDirectory, flush=True)
        os.makedirs(app.config["UPLOAD_PATH"], mode=0o777, exist_ok=True)
        filePath = os.path.join(app.config["UPLOAD_PATH"], filename)
        filePathNumbered = uniquify(filePath)
        uploaded_file.save(filePathNumbered)
    return redirect(url_for('index'))

@app.route('/optimize', methods=['POST'])
def optimize():
    fileName =request.form['file']
    budget=request.form['budget']
    acquisitionFunction =request.form['acquisition']
    # MFBO code should go here.
    return [0]

def uniquify(path):
    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = filename + "(" + str(counter) + ")" + extension
        counter += 1

    return path

# Running app
if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, render_template
from flask import redirect, url_for, request,flash
from werkzeug.utils import secure_filename


import os


app = Flask(__name__)

app.secret_key = 'mysecretkey'
app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'static/midi/')

@app.route('/')
def home():
    # return "hello Hải"
    return render_template('home.html')

@app.route('/sangtac.html')
def sangtac():

    return render_template('sangtac.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    if file:
        if file.filename[-3:] not in ['mid', 'xml', 'krn']:
            flash('Tệp không thuộc định dạng cho phép. Tải lên không thành công!')
            return redirect(url_for('sangtac'))
        else:
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename)))
            flash('Tải tệp lên thành công')
            return redirect(url_for('sangtac'))

    else:
        flash('Không có tệp được chọn')
        return redirect(url_for('sangtac'))
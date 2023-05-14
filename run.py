#####################################
from flask import Flask, render_template
from flask import redirect, url_for, request,flash
from werkzeug.utils import secure_filename

print("đang cài đặt và tải các thư viện .... !")
import tensorflow as tf
from app.convert_goc import midi2idxenc,idxenc2stream,create_dataset
from app.transformer import MusicGenerator
from  app.vocab import  MusicVocab
from app.music_transform import MusicItem

import os
########################################
# load model và predict
print("Tải model và tokens input .... ")

def load_model(fp):
    return tf.saved_model.load(fp)


def get_song_tokens(vocab,fp_midi):
    return midi2idxenc(fp_midi, vocab, add_bos=True, add_eos=False)


def get_middle_c_song(vocab):
    return [vocab.stoi['n100'], vocab.stoi['d10'], vocab.stoi['n109'], vocab.stoi['d10'], vocab.stoi['n112'],
            vocab.stoi['d10']]







#######################################
NAME_MIDI = ''
# tạo app flask
print("tạo app flask")
app = Flask(__name__)
# Thiết lập PYTHONPATH cho Flask
path = '.\\app\\'
if 'PYTHONPATH' in os.environ:
    os.environ['PYTHONPATH'] += os.pathsep + path
else:
    os.environ['PYTHONPATH'] = path

app.secret_key = 'mysecretkey'
app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'static/midi/')

@app.route('/')
def home():
    # return "hello Hải"
    return render_template('home.html')

@app.route('/sangtac.html')
def sangtac():
    midi_files = os.listdir('static/midi')
    midi_new_files = os.listdir('static/midi_new/')
    return render_template('sangtac.html', midi_files=midi_files,midi_new_files =midi_new_files)

@app.route('/process_select', methods=['POST'])
def process_select():
    global NAME_MIDI 
    NAME_MIDI = request.data.decode('utf-8')
    print("NAME_MIDI=",NAME_MIDI)
    # Làm gì đó với biến name_midi ở đây
    flash(f'Đã chọn tệp midi: {NAME_MIDI}')
    print(f"chọn tệp midi: {NAME_MIDI}")
    return redirect(url_for('sangtac'))

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

@app.route('/show_result', methods=['POST'])
def show_result():
    content_type = request.headers.get('Content-Type')
    if (content_type == 'application/json'):
        start = int(request.json.get('startValue'))
        end = int(request.json.get('endValue'))
        genlen = 256
        # genlen = int(request.json.get('inputGenLen')) 
        print( "Content type is supported.")
        print(f"start và end = {start} - {end}")
    

        print("load model và input")
        loaded_model = tf.saved_model.load('static\\model\\decoder_only_smaller_512_mega_ds\\')
        generator = MusicGenerator(loaded_model)
        created_vocab = MusicVocab.create()

        # Load midi và tạo tokens input
        inp_path = 'static/midi/'+NAME_MIDI
        print("transform midi to idxenc: ",inp_path)
        tokens_original = midi2idxenc(inp_path, created_vocab, add_bos=False, add_eos=False)
        len_tokens_original = len(tokens_original)
        start = round(start*len_tokens_original/100)
        if start%2==1:
            start -= 1 
        end = round(end*len_tokens_original/100)
        if end%2==1:
            end += 1 

        print(f"len tokens_original = {len_tokens_original}, cắt token input [{start}:{end}]")
        # print(f"start = {start}")
        # print(f"end = {end}")
        i = tokens_original[start:end]
        o = tokens_original[end:-1]
        ii = idxenc2stream(i, vocab=created_vocab)
        ii.write('midi','static/midi_new/input.mid')
        oo = idxenc2stream(o, vocab=created_vocab)
        oo.write('midi','static/midi_new/target.mid')


        print("Đợi 1 chút, đang sáng tác ...")
        op4 = generator.extend_sequence(i, max_generate_len=genlen)
        print("sáng tác xong !!! ...")

        op4_stream = idxenc2stream(op4.numpy(), vocab=created_vocab)
        op4_stream.write('midi','static/midi_new/predict.mid')
        print("lưu") 
        flash('sáng tác xong')
        return redirect(url_for('sangtac'))
    else:
        print( "Content type is not supported.")
        # return redirect(url_for('sangtac'))
            



#################################################
if __name__ == '__main__':


    app.run(debug=True)

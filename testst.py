import tensorflow as tf
from app.convert_goc import midi2idxenc,idxenc2stream,create_dataset
from app.transformer import MusicGenerator
from  app.vocab import  MusicVocab
# from app.music_transform import MusicItem

import os

print("Tải model và tokens input .... ")

def load_model(fp):
    return tf.saved_model.load(fp)


def get_song_tokens(vocab,fp_midi):
    return midi2idxenc(fp_midi, vocab, add_bos=True, add_eos=False)


def get_middle_c_song(vocab):
    return [vocab.stoi['n100'], vocab.stoi['d10'], vocab.stoi['n109'], vocab.stoi['d10'], vocab.stoi['n112'],
            vocab.stoi['d10']]

loaded_model = tf.saved_model.load('static\\model\\decoder_only_smaller_512_mega_ds\\')
generator = MusicGenerator(loaded_model)
created_vocab = MusicVocab.create()

# Load midi và tạo tokens input
inp_path = 'static/midi/themeoflove2.mid'
tokens_original = midi2idxenc(inp_path, created_vocab, add_bos=False, add_eos=False)
i = tokens_original[:512]
o = tokens_original[512:512+512]
# ii = idxenc2stream(i, vocab=created_vocab)
# ii.write('midi','static/midi_new/input.mid')
# oo = idxenc2stream(o, vocab=created_vocab)
# oo.write('midi','static/midi_new/target.mid')


print("Đợi 1 chút, đang sáng tác ...")
op4 = generator.extend_sequence(i, max_generate_len=512)

# string_note = created_vocab.textify(op4)

print(op4)  

op4_stream = idxenc2stream(op4.numpy(), vocab=created_vocab)
op4_stream.write('midi','.\\static\\midi_new\\predcit.mid')
print("sáng tác xong !!! ...")
print("lưu") 
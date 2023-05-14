"""
Convert midi files in directory to a TF Dataset for training
helper functions: https://github.com/bearpelican/musicautobot/blob/master/musicautobot/music_transformer/transform.py
"""

import os
import pathlib
import random
import numpy as np
import tensorflow as tf
from app.vocab import *


def create_dataset(fp, vocab, seq_len, batch_size, transpose, shuffle=True,kieu_chop_idxenc = 1, type_file = '*.mid|*.krn|*.xml'):
    ds = files2ds(fp, type_file, vocab, seq_len, transpose,kieu_chop_idxenc).cache()
    if shuffle:
        ds = ds.shuffle(buffer_size=1024, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)
    ds = ds.map(lambda x: (x[:, 0, :], x[:, 1, :]))
    return ds.prefetch(tf.data.AUTOTUNE)


def files2ds(root, pattern, vocab, seq_len, transpose,kieu_chop_idxenc):
    datasets = []
    root = pathlib.Path(root)
    filenames = []
    patterns = pattern.split('|')
    for pat in patterns:
      for filepath in root.rglob(pat) :
          filenames.append(filepath.resolve())
      
    # random.shuffle(filena mes)
    print(f"tìm thấy {len(filenames)} file ")
    
    if len(filenames)==0: 
        raise Exception(f"không tim thấy file {pattern} nào !!!")
        # filenames = filenames[:15]
    # filenames = filenames[:3]
    for i,filepath in enumerate(filenames):
        try:
            datasets.append(file2ds(filepath, vocab, seq_len, transpose, kieu_chop_idxenc,index_fp = i))
        except Exception as e:
            # os.remove(filepath)
            print(f'{i} could not parse and deleted {filepath}')
    ds = datasets[0]
    for i in range(1, len(datasets)):
        ds = ds.concatenate(datasets[i])
    return ds


def file2ds(fp, vocab, seq_len, transpose, kieu_chop_idxenc, index_fp = None):
    path_txt = pathlib.Path(str(fp)+'.txt')
    
    if not path_txt.exists() or str(fp)[-4:] == '.krn':
      idxencs = midi2idxenc(fp, vocab, transpose=transpose, add_bos=True, add_eos=True, index_fp = index_fp)
      if str(fp)[-4] != '.krn':
        np.savetxt(str(fp)+'.txt' , np.array(idxencs))
    else:
      with open(path_txt, 'r') as f:
        idxencs = np.loadtxt(f, dtype=np.int32, delimiter=' ')
        print(f"{index_fp}đã load lại: {path_txt}")

    for i, idx in enumerate(idxencs[0]):
        if idx < 0 or idx >= vocab.dur_range[1]:
            print(f'{index_fp}:{fp}: elem at {i} is {idx}')
            raise Exception("lỗi và xóa file")
            break
            
    if len(idxencs[0]) < seq_len and kieu_chop_idxenc !=11:
        print(f'WARNING: The encoded version of {fp.name} is only {len(idxencs[0])} tokens compared to step({seq_len})')
        raise Exception("lỗi và xóa file")
    data = []
    for idxenc in idxencs:
        if  kieu_chop_idxenc == 0:
            data.extend(chop_idxenc(idxenc, seq_len))
        elif kieu_chop_idxenc == 1:
            data.extend(chop_idxenc_1(idxenc, seq_len))
        elif  kieu_chop_idxenc == 11:
            data.extend(chop_idxenc_11(idxenc, seq_len))

    ds = tf.data.Dataset.from_tensor_slices(data)
    return ds


def chop_idxenc(idxenc, seq_len):
    data = []
    i = 0
    idxenc = pad_seq(idxenc, (len(idxenc) // seq_len + 1) * seq_len)
    while i + seq_len *2 <= len(idxenc):
        inp = idxenc[i: i + seq_len]
        target = idxenc[i + seq_len: i + seq_len *2]
        data.append([inp, target])
        # i += seq_len // 2
        i += seq_len
    return data

def chop_idxenc_1(idxenc, seq_len):
    data = []
    i = 0
    idxenc = pad_seq(idxenc, (len(idxenc) // seq_len + 1) * seq_len + 1)
    while i + seq_len + 1 <= len(idxenc):
        inp = idxenc[i: i + seq_len]
        target = idxenc[i+1 : i + seq_len +1]
        data.append([inp, target])
        i += seq_len // 2
        # i += 1
    return data

def chop_idxenc_11(idxenc, seq_len):
    data = []
    i = 0
    idxenc = np.pad(idxenc, (seq_len), mode='constant', constant_values=0)
    while i + seq_len + 1 <= len(idxenc):
        inp = idxenc[i: i + seq_len]
        target = idxenc[i+1 : i + seq_len +1]
        data.append([inp, target])
        i += 1
    return data

def pad_seq(seq, seq_len):
    pad_len = max(seq_len - seq.shape[0], 0)
    return np.pad(seq, (0, pad_len), 'constant', constant_values=0)[:seq_len]


def transpose_npenc(npenc_transposed, offset):
    npenc_transposed = npenc_transposed.copy()
    for note in npenc_transposed:
        if note[0] >= 0:
            note[0] += offset
    return npenc_transposed


def midi2idxenc(midi_file, vocab, transpose=-1, add_bos=False, add_eos=False, index_fp = None):
    "Converts midi file to index encoding for training, applies transposition"
    npenc = midi2npenc(midi_file)
    idxencs = [npenc2idxenc(npenc, vocab, add_bos=add_bos, add_eos=add_eos)]
    print(f'{index_fp}: converted {midi_file} to index encoding ({len(idxencs[0])} tokens)')
    if transpose == -1:
        return idxencs[0]
    if transpose == 0:
        return idxencs
    max_pitch = -1
    min_pitch = 1000000
    for n in npenc:
        max_pitch = max(n[0], max_pitch)
        if n[0] >= 0:
            min_pitch = min(min_pitch, n[0])
    transpose_down = min(0, PIANO_RANGE[0] - min_pitch)
    transpose_up = max(0, PIANO_RANGE[1] - max_pitch)
    if transpose_up == 0 and transpose_up == 0:
        return idxencs
    transpose_range = [max(-transpose, transpose_down), min(transpose, transpose_up)]
    print(f'transposing in range {transpose_range}')
    idxencs = []
    for i in range(transpose_range[0], transpose_range[1] + 1):
        if i == 0:
            continue
        idxencs.append(npenc2idxenc(transpose_npenc(npenc, i), vocab, add_bos=add_bos, add_eos=add_eos))
    # idxencs = [npenc2idxenc(transpose_npenc(npenc, transpose_range[0]), vocab, add_bos=add_bos, add_eos=add_eos),
    #            npenc2idxenc(transpose_npenc(npenc, transpose_range[0] / 3), vocab, add_bos=add_bos, add_eos=add_eos),
    #            npenc2idxenc(transpose_npenc(npenc, transpose_range[-1] / 3), vocab, add_bos=add_bos, add_eos=add_eos),
    #            npenc2idxenc(transpose_npenc(npenc, transpose_range[-1]), vocab, add_bos=add_bos, add_eos=add_eos)]
    return idxencs


def idxenc2stream(arr, vocab, bpm=120):
    "Converts index encoding to music21 stream"
    npenc = idxenc2npenc(arr, vocab)
    return npenc2stream(npenc, bpm=bpm)


# single stream instead of note,dur
def npenc2idxenc(t, vocab, add_bos=False, add_eos=False):
    "Transforms numpy array from 2 column (note, duration) matrix to a single column"
    "[[n1, d1], [n2, d2], ...] -> [n1, d1, n2, d2]"
    t = t.copy()

    t[:, 0] = t[:, 0] + vocab.note_range[0]
    t[:, 1] = t[:, 1] + vocab.dur_range[0]

    prefix = np.array([vocab.stoi[BOS], vocab.pad_idx]) if add_bos else np.empty(0, dtype=int)
    suffix = np.array([vocab.stoi[EOS]]) if add_eos else np.empty(0, dtype=int)
    return np.concatenate([prefix, t.reshape(-1), suffix]).astype('int32')


def idxenc2npenc(t, vocab, validate=True):
    if validate:
        t = to_valid_idxenc(t, vocab.npenc_range)
    t = t.copy().reshape(-1, 2)
    if t.shape[0] == 0: return t

    t[:, 0] = t[:, 0] - vocab.note_range[0]
    t[:, 1] = t[:, 1] - vocab.dur_range[0]

    if validate: return to_valid_npenc(t)
    return t


def to_valid_idxenc(t, valid_range):
    r = valid_range
    t = t[np.where((t >= r[0]) & (t < r[1]))]
    if t.shape[-1] % 2 == 1: t = t[..., :-1]
    return t


def to_valid_npenc(t):
    is_note = (t[:, 0] < VALTSEP) | (t[:, 0] >= NOTE_SIZE)
    invalid_note_idx = is_note.argmax()
    invalid_dur_idx = (t[:, 1] < 0).argmax()

    invalid_idx = max(invalid_dur_idx, invalid_note_idx)
    if invalid_idx > 0:
        if invalid_note_idx > 0 and invalid_dur_idx > 0: invalid_idx = min(invalid_dur_idx, invalid_note_idx)
        print('Non midi note detected. Only returning valid portion. Index, seed', invalid_idx, t.shape)
        return t[:invalid_idx]
    return t
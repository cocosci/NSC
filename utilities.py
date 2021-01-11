from loss_terms_and_measures import *
import numpy as np
from pystoi.stoi import stoi
import soundfile as sf


def hann_process(utterance_seg, seg_ind, seg_amount):
    hop_size = frame_length - overlap_each_side # followed bloomberg paper

    the_window = np.append(np.append(np.hanning(overlap_each_side * 2 - 1)[:overlap_each_side],
                                     np.array([1] * (frame_length - overlap_each_side * 2))),
                           np.hanning(overlap_each_side * 2 - 1)[overlap_each_side - 1:])

    first_window = np.append(np.append(np.array([1] * (overlap_each_side)),  np.array([1] * (frame_length - overlap_each_side*2))), np.hanning(overlap_each_side*2)[(overlap_each_side):])
    last_window = np.append(np.append(np.hanning(overlap_each_side*2)[:(overlap_each_side)],  np.array([1] * (frame_length - overlap_each_side*2))), np.array([1]*(overlap_each_side)))
    if seg_ind == 0:
        utterance_seg = utterance_seg * first_window
    elif seg_ind == seg_amount - 1:
        utterance_seg = utterance_seg * last_window
    else:
        utterance_seg = utterance_seg * the_window
    return utterance_seg


def utterance_to_segment(utterance, post_window=False):
    ret = np.empty((len(range(0, len(utterance) - frame_length, frame_length - overlap_each_side)), frame_length))
    ind = 0
    the_window = np.append(np.append(np.hanning(overlap_each_side * 2 - 1)[:overlap_each_side],
                                     np.array([1] * (frame_length - overlap_each_side * 2))),
                           np.hanning(overlap_each_side * 2 - 1)[overlap_each_side - 1:])
    if post_window:
        for i in range(0, len(utterance) - frame_length, frame_length - overlap_each_side):
            ret[ind, :] = utterance[i: (i + frame_length)] * 1
            ind += 1
    else:
        for i in range(0, len(utterance) - frame_length, frame_length - overlap_each_side):
            ret[ind, :] = utterance[i: (i + frame_length)] * the_window
            ind += 1
    return ret


def eval_metrics(ori_sig, dec_sig, _rand_model_id):
    _min_len, _snr, _ori_sig, _dec_sig = snr(ori_sig, dec_sig)
    _si_snr = si_snr(_dec_sig, _ori_sig)
    the_stoi = stoi(_ori_sig, _dec_sig, 16000, extended=False)
    sf.write('ori_sig_' + _rand_model_id + '.wav', ori_sig, 16000, 'PCM_16')
    sf.write('dec_sig_' + _rand_model_id + '.wav', dec_sig, 16000, 'PCM_16')
    the_pesq = pesq('ori_sig_' + _rand_model_id + '.wav',
                    'dec_sig_' + _rand_model_id + '.wav', 16000)
    return _min_len, _si_snr, _snr, float(the_stoi), float(the_pesq), np.corrcoef(_ori_sig, _dec_sig)[0][1]
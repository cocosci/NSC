from audiolazy import *
from spectrum import poly2lsf, lsf2poly
import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
from constants import *

empha_filter = ZFilter(np.array([1, empha_filter_coeff]).tolist())

highpass_filter = ZFilter(np.array([0.989502, -1.979004, 0.989592]).tolist()) / ZFilter(
    np.array([1, -1.978882, 0.979126]).tolist())


def lpc_analysis_at_train(raw_data_one_batch, order):  # already windowized by 512
    raw_data_one_batch = raw_data_one_batch[:, :, 0]
    how_many = raw_data_one_batch.shape[0]  # which is the batch size
    all_lpc_coeff_segments = np.empty((how_many, order))

    for i in range(how_many):
        preprocessed_frame = empha_filter(highpass_filter(raw_data_one_batch[i, :]))
        preprocessed_frame_listed = list(preprocessed_frame)
        # preprocessed_frame_listed = list(raw_data_one_batch[i, :])
        lpc_analysis = lpc(preprocessed_frame_listed, order)
        all_lpc_coeff_segments[i, :] = poly2lsf(ZFilter(np.array(lpc_analysis.numlist).tolist()).numlist)
    return all_lpc_coeff_segments  # after poly2lsf, the "1" is removed


def lsf2poly_after_quan(lpc_in_lsf, order):
    how_many = lpc_in_lsf.shape[0]  # which is the batch size
    all_lpc_coeff_poly = np.empty((how_many, order + 1))
    for i in range(how_many):
        all_lpc_coeff_poly[i, :] = lsf2poly(lpc_in_lsf[i, :]).tolist()
    return all_lpc_coeff_poly.astype(np.float32)


def lpc_analysis_get_residual(raw_data_one_batch, quan_lpc_coeff):
    how_many = raw_data_one_batch.shape[0]  # which is the batch size
    all_lpc_res_segments = np.zeros((how_many, 512))
    for i in range(how_many):
        lpc_analysis = ZFilter(quan_lpc_coeff[i, :].tolist())
        # without sub-frame
        # all_lpc_res_segments[i, :] = np.array(list(lpc_analysis(raw_data_one_batch[i, :].flatten())))
        # with sub-frame
        all_lpc_res_segments[i, :128] += np.array(
            list(lpc_analysis(raw_data_one_batch[i, :128].flatten()))) * np.append(np.array([1] * 64),
                                                                                   np.hanning(64 * 2)[(64):])
        all_lpc_res_segments[i, 64: 192] += np.array(
            list(lpc_analysis(raw_data_one_batch[i, 64: 192].flatten()))) * np.hanning(64 * 2)
        all_lpc_res_segments[i, 128:256] += np.array(
            list(lpc_analysis(raw_data_one_batch[i, 128:256].flatten()))) * np.hanning(64 * 2)
        all_lpc_res_segments[i, 192:320] += np.array(
            list(lpc_analysis(raw_data_one_batch[i, 192:320].flatten()))) * np.hanning(64 * 2)
        all_lpc_res_segments[i, 256:384] += np.array(
            list(lpc_analysis(raw_data_one_batch[i, 256:384].flatten()))) * np.hanning(64 * 2)
        all_lpc_res_segments[i, 320:448] += np.array(
            list(lpc_analysis(raw_data_one_batch[i, 320:448].flatten()))) * np.hanning(64 * 2)
        all_lpc_res_segments[i, 384:512] += np.array(
            list(lpc_analysis(raw_data_one_batch[i, 384:512].flatten()))) * np.append(np.hanning(64 * 2)[:(64)],
                                                                                      np.array([1] * 64))

    return all_lpc_res_segments.astype(np.float32)


# def sub_frame_hann_process(utterance_seg, seg_ind, seg_amount):
# 	hop_size = frame_length - overlap_each_side # followed bloomberg paper
#
#
# 	the_window = np.append(np.append(np.hanning(overlap_each_side*2)[:(overlap_each_side)],  np.array([1] * (frame_length - overlap_each_side*2))), np.hanning(overlap_each_side*2)[(overlap_each_side):])
# 	first_window= np.append(np.append(np.array([1]*(overlap_each_side)),  np.array([1] * (frame_length - overlap_each_side*2))), np.hanning(overlap_each_side*2)[(overlap_each_side):])
# 	last_window = first_window= np.append(np.append(np.hanning(overlap_each_side*2)[:(overlap_each_side)],  np.array([1] * (frame_length - overlap_each_side*2))), np.array([1]*(overlap_each_side)))
# 	if seg_ind == 0:
# 		utterance_seg = utterance_seg*first_window
# 	elif seg_ind == seg_amount - 1:
# 		utterance_seg = utterance_seg*last_window
# 	else:
# 		utterance_seg = utterance_seg*the_window
# 	return utterance_seg
def lpc_analysis_at_test(raw_data, order):
    # all_lpc_res_segments = np.empty((how_many, 512))
    #
    # print(raw_data.shape, 'raw_data')
    raw_data = raw_data[:, :].flatten()

    ret = np.empty((len(range(0, len(raw_data) - frame_length * 2, frame_length * 2 - frame_length)), frame_length * 2))
    ind = 0
    for i in range(0, len(raw_data) - frame_length * 2, frame_length * 2 - frame_length):
        ret[ind, :] = raw_data[i:i + frame_length * 2] * 1
        ind += 1

    raw_data = ret
    # print(raw_data.shape, 'raw_data')

    how_many = raw_data.shape[0]
    all_lpc_coeff_segments = np.empty((how_many, order))

    for i in range(how_many):
        # no window
        # preprocessed_frame = (raw_data[i,:])
        # preprocessed_frame_listed = list(preprocessed_frame)
        # lpc_analysis = lpc(preprocessed_frame_listed, order)
        # all_lpc_coeff_segments[i,:] = poly2lsf(ZFilter(np.array(lpc_analysis.numlist).tolist()).numlist)
        # with window
        # preprocessed_frame = ((raw_data[i, :]))*np.hanning(512*2)
        preprocessed_frame = ((raw_data[i, :])) * np.append(np.append(np.hanning(256 * 2)[:256], np.array([1] * 512)),
                                                            np.hanning(256 * 2)[256:])  # np.hanning(512*2)
        preprocessed_frame_listed = list(preprocessed_frame)
        lpc_analysis = lpc(preprocessed_frame_listed, order)
        all_lpc_coeff_segments[i, :] = poly2lsf(ZFilter(np.array(lpc_analysis.numlist).tolist()).numlist)
    return all_lpc_coeff_segments


# all_lpc_coeff_segments[i,:] = np.array(lpc_analysis.numlist)[1:]
# all_lpc_res_segments[i, :] = np.array(list(lpc_analysis(preprocessed_frame_listed)))

# lpc_analysis = lpc(raw_data[i,:], order)
# all_lpc_coeff_segments[i,:] = np.array(lpc_analysis.numlist)[1:]
# all_lpc_res_segments[i, :] = np.array(list(lpc_analysis(raw_data[i,:])))
# return all_lpc_coeff_segments, all_lpc_res_segments


@tf.custom_gradient
def lpc_synthesizer_tr(lpc_coeff, lpc_res):
    how_many = lpc_res.shape[0]
    synthesized = np.empty((how_many, 512))
    for i in range(how_many):
        # lpc_coeff_analysis = ZFilter(lpc_coeff[i, :].tolist())
        # synth_filt = 1 / lpc_coeff_analysis
        # synthesized[i, :] = np.array(list(synth_filt(lpc_res[i, :].flatten())))

        lpc_coeff_analysis = ZFilter((np.array(lpc_coeff[i, :]) / 1).tolist())
        synth_filt = 1 / lpc_coeff_analysis
        # synthesized[i, :] = np.array(list(((1/empha_filter)(synth_filt(lpc_res[i, :].flatten())))))
        synthesized[i, :] = np.array(list(((synth_filt(lpc_res[i, :].flatten())))))

    # synthesized[i, :] = np.array(list(((synth_filt(lpc_res[i, :].flatten())))))
    # print(tf.gradients( synthesized, lpc_res, '====3========'))
    def grad(dy):
        return dy

    return synthesized.astype(np.float32), grad
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import os
import mdct
from constants import *


def vec_l2norm_tf(x):
    return tf.sqrt(tf.reduce_sum(input_tensor=tf.square(x), axis=-1))


def si_snr_loss(x, s):
    """
    TensorFlow version of Si-SNR. The results for the same x and s from this function
    is the same with the one from si_snr function.
    :param x: decoded sample
    :param s: source (ground truth)
    :return: si-snr
    """
        # return tf.norm(x, axis=-1)
        # return tf.reduce_sum(input_tensor=tf.square(x), axis=-1) + 1e-07
    # x_zm = tf.compat.v1.subtract(x, tf.reduce_mean(input_tensor=x, axis=-1,  keepdims=True))
    # s_zm = tf.compat.v1.subtract(s, tf.reduce_mean(input_tensor=s, axis=-1,  keepdims=True))
    x_zm = x
    s_zm = s
    t = tf.matmul(tf.compat.v1.linalg.tensor_diag(1 / (vec_l2norm_tf(s_zm)**2)),
                  tf.matmul(tf.compat.v1.linalg.tensor_diag(tf.reduce_sum(tf.multiply(x_zm, s_zm), -1)), s_zm))
    n = x_zm - t
    return vec_l2norm_tf(n)
    # return 10 * tflog10(vec_l2norm_tf(n))
    # return 10 * tflog10(vec_l2norm_tf(t) / vec_l2norm_tf(n))



def si_snr(x, s):
    """
    Python version of Si-SNR.
    :param x: decoded sample
    :param s: source (ground truth)
    :return: si-snr
    """
    def vec_l2norm(x):
        return np.linalg.norm(x, 2)
    x_zm = x - np.mean(x)
    s_zm = s - np.mean(s)
    t = (np.inner(x_zm, s_zm) / vec_l2norm(s_zm)**2) * s_zm
    n = x_zm - t
    return 20 * np.log10(vec_l2norm(t) / vec_l2norm(n))


# def mu_law_compressing(input_x, mu=255):
#     return np.sign(input_x) * (np.log(1 + mu * np.abs(input_x)) / np.log(1 + mu))

def mu_law_mapping(input_x, mu=255):
    return tf.sign(input_x) * (tf.compat.v1.math.log(1.0 + mu * tf.abs(input_x)) / tf.compat.v1.math.log(1.0 + mu))


def inverse_mu_law_mapping(input_x, mu=255):
    return tf.sign(input_x) * (1 / mu) * ((1 + mu) ** tf.abs(input_x) - 1)


def entropy_to_bitrate(total_entropy, the_strides):
    # print(code_len_val)
    code_len_val = 128 if the_strides == 4 else 256
    bitrate = ((sample_rate / 1024.0) / (frame_length - overlap_each_side)) * code_len_val * total_entropy
    return bitrate


def bitrate_to_entropy(bitrate, the_strides):
    PRE_ENTROPY_RATE = (frame_length / the_strides) * (float(frame_length / the_strides) / frame_length)
    entropy = (bitrate / PRE_ENTROPY_RATE * sample_rate)
    entropy *= (frame_length - overlap_each_side / float(frame_length))
    return entropy


def mse_loss(decoded_sig, original_sig, kai_re_mat=1):
    mse = tf.reduce_mean(input_tensor=tf.square(tf.subtract(decoded_sig, original_sig)), axis=-1)
    return tf.sqrt(mse + 1e-07)


def mse_loss_v1(decoded_sig, original_sig, kai_re_mat=1):
    mse = tf.reduce_mean(input_tensor=tf.square(tf.subtract(decoded_sig, original_sig)), axis=-1)
    return tf.sqrt(mse + 1e-07)

# @tf.function
# def mdct_transform(sig):
#     bar = 256
#     for i in range(128):  # batch size
#         mdct_spec_full = tf.signal.mdct(sig[i, :], frame_length=512)
#         mask = np.ones((mdct_spec_full.shape[0], mdct_spec_full.shape[1]))
#         mask[bar:, :] = 0
#         mdct_spec_full = tf.multiply(mdct_spec_full, mask)
#         sig[i, :] = tf.signal.inverse_mdct(mdct_spec_full)
#     return sig

def mdct_transform(sig):
    bar = 128
    mdct_spec_full = tf.signal.mdct(sig, frame_length=512)
    # print(mdct_spec_full.shape)
    mask = np.ones((128, 1, mdct_spec_full.shape[1]))
    mask[:, :, bar:] = 0
    mdct_spec_full = tf.multiply(mdct_spec_full, mask)
    sig = tf.signal.inverse_mdct(mdct_spec_full)
    print(sig.shape)
    return sig


#def mse_loss(decoded_sig, original_sig, kai_re_mat=1):
#    decoded_sig, original_sig = mdct_transform(decoded_sig), mdct_transform(original_sig)
#    mse = tf.reduce_mean(input_tensor=tf.square(tf.subtract(decoded_sig, original_sig)), axis=-1)
#    return tf.sqrt(mse + 1e-07)


def tp_mse_loss(decoded_sig, original_sig, kai_re_mat=1):
    eps = 1e-05
    # mse = tf.reduce_mean(input_tensor=tf.square(tf.subtract(decoded_sig, original_sig)), axis=-1)
    tp_mse = tf.reduce_mean(tf.square((tf.subtract(original_sig, decoded_sig) / (original_sig + eps) * tf.expand_dims(tf.sqrt(tf.reduce_mean(original_sig + eps, axis=-1)), -1))), axis=-1)
    corr = tf.square(1 - tfp.stats.correlation(decoded_sig, original_sig, sample_axis=1, event_axis=None))

    return corr


def tflog10(x):
    numerator = tf.compat.v1.log(x)
    denominator = tf.compat.v1.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


def mfcc_transform(the_stft, the_spectrum, is_finetuning=False):    # Warp the linear scale spectrograms into the mel-scale.
    num_spectrogram_bins = the_stft.shape[-1]
    sample_rate, lower_edge_hertz, upper_edge_hertz = 16000, 0.0, 8000.0
    selected_ind = [8, 16, 32, 128]
    # selected_ind = [8, 16, 32]
    # selected_ind = [128]
    MEL_FILTERBANKS = []
    for num_mel_bins in selected_ind:
        linear_to_mel_weight_matrix = tf.compat.v2.signal.linear_to_mel_weight_matrix(
            num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz, upper_edge_hertz)
        MEL_FILTERBANKS.append(linear_to_mel_weight_matrix)
        # MEL_FILTERBANKS.shape = (257*num_mel_bins)
    transform = []
    for filter_bank in MEL_FILTERBANKS:
        mel_spectrograms = tf.matmul(the_spectrum, filter_bank) # axis = 1 means it's just mat mul.
        log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-7)
        transform.append(log_mel_spectrograms)
    # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
    return transform


def mfcc_loss(decoded_sig, original_sig, is_finetuning=False):
    # calculate stft spectrum
    the_frame_length = frame_length
    dec_stfts, dec_spectrograms = tf_stft(decoded_sig)
    ori_stfts, ori_spectrograms = tf_stft(original_sig)
    # calculate stft psd
    ori_spectrograms = ori_spectrograms ** 2
    ori_spectrograms = 1.0 / frame_length * ori_spectrograms
    dec_spectrograms = dec_spectrograms ** 2
    dec_spectrograms = 1.0 / frame_length * dec_spectrograms
    ###########

    pvec_pred = mfcc_transform(dec_stfts, dec_spectrograms, is_finetuning)
    pvec_true = mfcc_transform(ori_stfts, ori_spectrograms, is_finetuning)

    distances = []
    for i in range(0, len(pvec_true)):
        # For the highest resolution, focuse on low frequencies.
        if pvec_pred[i].shape[0] == 128:
            pvec_pred[i][64:] = pvec_true[i][64:]
        error = tf.expand_dims(mse_loss_v1(pvec_pred[i], pvec_true[i]), axis=-1)
        distances.append(error)
    distances = tf.concat(distances, axis=-1)
    mfcc_loss = tf.reduce_mean(input_tensor=distances, axis=-1)
    return mfcc_loss


def tf_stft(sig, the_frame_length=frame_length):
    dec_stfts = tf.compat.v2.signal.stft(tf.reshape(sig, [-1, frame_length]), frame_length=the_frame_length,
                                         frame_step=int(the_frame_length), fft_length=the_frame_length, window_fn=None)
    dec_stfts = tf.reshape(dec_stfts, (-1, int(the_frame_length / 2) + 1))
    dec_spectrograms = tf.sqrt(tf.square(tf.math.real(dec_stfts)) + tf.square(tf.math.imag(dec_stfts)) + 1e-7)
    return dec_stfts, dec_spectrograms


def tf_log10(x):
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


def tf_psd(sig, the_frame_length=frame_length):
    # dec_stfts = tf.compat.v2.signal.stft(tf.reshape(sig, [-1, frame_length]), frame_length = the_frame_length, frame_step=int(the_frame_length), fft_length=the_frame_length, window_fn=None)
    # dec_stfts = tf.reshape(dec_stfts, (-1, int(the_frame_length/2)+1))
    spec = tf.compat.v2.signal.rfft(sig)
    # abs_spec = tf.abs(spec)
    abs_spec = tf.sqrt(tf.square(tf.math.real(spec)) + tf.square(tf.math.imag(spec)) + 1e-7)
    # P = tf.clip_by_value(20 * tf_log10(abs_spec / 512), -200, 10000)
    P = (20 * tf_log10(abs_spec / 512))
    # Delta = 96 - np.max(P)
    Delta = 96 - tf.reduce_max(P)
    P = P + Delta
    return P, spec


# The higher the SMR is, the less masking we have.
# We use SMR to prioritize the the loss minimization in frequency domain.
def SMR(decoded_sig, original_sig, GMS, the_frame_length=frame_length):
    dec_psd, dec_spectrograms = tf_psd(decoded_sig)
    ori_psd, ori_spectrograms = tf_psd(original_sig)
    # Usually you wouldn't spend any bits on the parts of the signal where
    # you have a negative SMR, because that would imply that that part of the signal is inaudible anyway.
    SMR = tf.nn.relu(ori_psd - GMS)
    mse = tf.reduce_mean(input_tensor=tf.multiply(tf.square(tf.subtract(ori_psd, dec_psd)), SMR), axis=-1)
    return tf.sqrt(mse + 1e-07)


# This is to maximize the average MNR
def MNR(decoded_sig, original_sig, GMS, the_frame_length=frame_length):
    diff_psd, diff_spectrograms = tf_psd(decoded_sig - original_sig)
    # return tf.reduce_mean((diff_psd - GMS) + 50, axis=-1)
    return tf.reduce_mean((diff_psd) + 50, axis=-1)

def tp_psd_corr(decoded_sig, original_sig):
    ori_psd, _ = tf_psd(original_sig)
    dec_psd, _ = tf_psd(decoded_sig)
    return tf.square(1 - tfp.stats.correlation(dec_psd, ori_psd, sample_axis=1, event_axis=None))

# This is to maximize the lower bound of MNR
def MNR_reduce_min(decoded_sig, original_sig, GMS, the_frame_length=frame_length):
    diff_psd, diff_spectrograms = tf_psd(decoded_sig - original_sig)
    return -tf.reduce_min((GMS - diff_psd) + 50, axis=-1)


def psd_loss(decoded_sig, original_sig, mat):
    ori_psd, ori_spectrograms = tf_psd(decoded_sig)
    dec_psd, dec_spectrograms = tf_psd(original_sig)
    mse = tf.reduce_mean(input_tensor=tf.multiply(tf.square(tf.subtract(ori_psd, dec_psd)), mat), axis=-1)
    return tf.sqrt(mse + 1e-07)


#def stft_loss(decoded_sig, original_sig, mat):
#    ori_stft, dec_spectrograms = tf_stft(decoded_sig)
#    dec_stft, ori_spectrograms = tf_stft(original_sig)
#    mse = tf.reduce_mean(input_tensor=tf.multiply(tf.square(tf.subtract(ori_spectrograms, dec_spectrograms)), mat),
#                         axis=-1)
#    return tf.sqrt(mse + 1e-07)

def stft_loss(decoded_sig, original_sig):
    ori_stft, dec_spectrograms = tf_stft(decoded_sig)
    dec_stft, ori_spectrograms = tf_stft(original_sig)
    mse = tf.reduce_mean(input_tensor=tf.square(tf.subtract(tf.math.real(ori_stft), tf.math.real(dec_stft))), axis=-1)
    # mse = tf.reduce_mean(input_tensor=tf.square(tf.subtract(ori_spectrograms, dec_spectrograms)), axis=-1)
    return tf.sqrt(mse + 1e-07)


def quan_loss(softmax_assignment):
    # return tf.reduce_mean(input_tensor=(tf.reduce_sum(input_tensor=tf.sqrt(softmax_assignment + 1e-20), axis=-1) - 1.0), axis=-1)
    return tf.reduce_mean(input_tensor=(tf.reduce_sum(input_tensor=tf.sqrt(softmax_assignment + 1e-20), axis=-1)), axis=-1)


def entropy_coding_loss(soft_assignment):
    soft_assignment = tf.reshape(soft_assignment, (-1, soft_assignment.shape[2]))
    onehot_hist = tf.reduce_sum(input_tensor=soft_assignment, axis=0)
    onehot_hist /= tf.reduce_sum(input_tensor=onehot_hist)
    ent_loss = -tf.reduce_sum(input_tensor=onehot_hist * tf.math.log(onehot_hist + 1e-7) / tf.math.log(2.0))
    return ent_loss


def snr(ori_sig, dec_sig):
    min_len = min(len(ori_sig), len(dec_sig))
    ori_sig, dec_sig = ori_sig[:min_len], dec_sig[:min_len]
    nom = np.sum(np.power(ori_sig, 2))
    denom = np.sum(np.power(np.subtract(ori_sig, dec_sig), 2))
    eps = 1e-20
    snr = 10 * np.log10(nom / (denom + eps) + eps)
    return min_len, snr, ori_sig, dec_sig


def pesq(reference, degraded, sample_rate=None, program='pesq'):
    """ Return PESQ quality estimation (two values: PESQ MOS and MOS LQO) based
    on reference and degraded speech samples comparison.
    Sample rate must be 8000 or 16000 (or can be defined reading reference file
    header).
    PESQ utility must be installed.
    """
    if not os.path.isfile(reference) or not os.path.isfile(degraded):
        raise ValueError('reference or degraded file does not exist')
    if not sample_rate:
        import wave
        w = wave.open(reference, 'r')
        sample_rate = w.getframerate()
        w.close()
    if sample_rate not in (8000, 16000):
        raise ValueError('sample rate must be 8000 or 16000')

    import subprocess
    args = [program, reference, degraded, '+%d' % sample_rate, '+wb']
    pipe = subprocess.Popen(args, stdout=subprocess.PIPE)
    out, _ = pipe.communicate()
    out = out.decode("utf-8")
    last_line = out.split('\n')[-2]
    pesq_wb = float(last_line.split()[-1:][0])
    return pesq_wb

from os import listdir
import random
import sys
import scipy.io.wavfile
import librosa
import argparse
from tensorflow.python.ops import math_ops
import math
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import random_ops
import os.path
import time
import collections
import itertools
from loss_terms_and_measures import *
from lpc_utilities import *
from nn_core_operator import *
from constants import *
from utilities import *


"""
The python module for one neural speech codec
"""
class neuralSpeechCodingModule(object):
    def __init__(self, arg):
        self._root_path = '/N/u/zhenk/BNN/libsdae-autoencoder-tensorflow/'
        self._learning_rate_tanh = arg.learning_rate_tanh
        self._coeff_term = list(map(lambda x: float(x), arg.coeff_term.split()))
        self._pretrain_step = arg.pretrain_step
        self._target_entropy = arg.target_entropy
        self._the_strides = list(map(lambda x: int(x), arg.the_strides.split()))

        self._res_scalar = arg.res_scalar
        self._save_unique_mark = arg.save_unique_mark
        self._num_bins_for_follower = list(map(lambda x: int(x), arg.num_bins_for_follower.split()))
        self._tr_data_size = training_data_size
        self._max_amp = max_amp_tr

        if is_pure_time_domain:
            tr_data = np.load(self._root_path + '/ac_stft_data/tr_time_journal.npy')[:int(self._tr_data_size), :]
            self._tr_data = tr_data
            print('tr_data', tr_data.shape)
        else:
            tr_lpc_res = np.load(self._root_path + 'cmrl_v2/residuals_with_quantized_lpc_coeff_256bins_signal_processed_order_16_final_flat.npy')[
                         :int(self._tr_data_size), :]
            tr_data = np.load(self._root_path + 'cmrl_v2/flat_raw_data.npy')[:int(self._tr_data_size), :]  # * self._lpc_res_scaler
            tr_lpc_coeff = np.load(self._root_path + 'cmrl_v2/end2end_lpc_coeff_in_lsf_16_signal_based_final_flat.npy')[:int(self._tr_data_size), :]
            self._lpc_order = 16 # tr_lpc_coeff.shape[1]
            self._is_cq = int(arg.is_cq)

            print('expanded tr_data: ', tr_data.shape, tr_lpc_coeff.shape)
            self._tr_data = np.concatenate([tr_data, tr_lpc_coeff, tr_lpc_res], 1)

        print(' tr_data: ', tr_data.shape)
        self._epoch_tanh = arg.epoch_tanh
        self._epoch_greedy_followers = list(map(lambda x: int(x), arg.epoch_greedy_followers.split()))
        self._batch_size = arg.batch_size
        self._training_mode = int(arg.training_mode)
        self._sep_val = [self._root_path + 'timit_val/' + f for f in listdir(self._root_path + 'timit_val/') if
                         f.endswith('.wav')]
        self._sep_test = [self._root_path + 'timit_test/' + f for f in listdir(self._root_path + 'timit_test/')
                                   if f.endswith('.wav')]

        self._rand_model_id = str(np.random.randint(1000000, 2000000))
        print('self._rand_model_id:', self._rand_model_id)
        self._base_model_id = arg.base_model_id
        self._suffix = arg.suffix
        self._bottleneck_kernel_and_dilation = list(map(lambda x: int(x), arg.bottleneck_kernel_and_dilation.split()))
        self._window_size = arg.window_size
        self._write_to_file_and_update_to_display(str(arg) + '\n\n')

    def _write_to_file_and_update_to_display(self, the_string):
        """
        Write model training and test outputs to a doc.
        """
        self._file_handler = open('./doc/' + self._rand_model_id + self._suffix + self._save_unique_mark + '_journal.txt', 'a')
        self._file_handler.write(the_string)
        self._file_handler.close()

    def _load_sig(self, the_wav_file):

        """
        Read one test file, normalize it, divided it by the max.
        """
        s, sr = librosa.load(the_wav_file, sr=None)  # saving redundantly many speech sources (easier to handle)
        the_scaler = np.std(s) * self._max_amp
        # s_1 = s/np.std(s)
        # s_2 = s/np.max(np.abs(s_1))
        # the_scaler = np.max(s_2) / np.max(s)

        # print('before norm', np.max(s))
        # print(np.std(s), self._max_amp, the_scaler)
        s /= the_scaler
        # print('after norm', np.max(s))
        return s, the_scaler

    def _load_sig_lpc(self, the_wav_file):

        """
        Read one test file, normalize it, divided it by the max.
        """
        s, sr = librosa.load(the_wav_file, sr=None)  # saving redundantly many speech sources (easier to handle)


        # resample
        if sr != sample_rate:
            print('resampled to', sample_rate)
            s = librosa.resample(s, sr, sample_rate)

        the_scaler = np.std(s)  # * self._max_amp
        s /= the_scaler
        return s, the_scaler

    def _generate_one_epoch_end2end(self, x, y, batchsize):
        the_list = list(range(0, x.shape[0] - self._batch_size, self._batch_size))
        random.shuffle(the_list)
        for i in the_list[:2500]:
        # for i in the_list[:500]:
            ret = np.reshape(self._tr_data[i:(i + batchsize), :], (batchsize, frame_length, 1))
            yield ret, ret

    def _generate_one_epoch_end2end_lpc(self, x, y, batchsize):
        the_list = list(range(0, x.shape[0] - self._batch_size, self._batch_size))
        random.shuffle(the_list)
        for i in the_list[:]:
            ret = np.reshape(x[i:i + batchsize, :frame_length], (batchsize, frame_length, 1))
            ret_lpc_coeff = np.reshape(x[i:i + batchsize, frame_length:frame_length + self._lpc_order],
                                       (batchsize, self._lpc_order, 1))
            yield ret, ret, ret_lpc_coeff

    def _generate_one_epoch_end2end_lpc_fast(self, x, y, batchsize):
        the_list = list(range(0, self._tr_data.shape[0] - self._batch_size, self._batch_size))
        random.shuffle(the_list)
        # print(len(the_list))  # 2310
        for i in the_list[:]:
            ret = np.reshape(self._tr_data[i:i + batchsize, :frame_length], (batchsize, frame_length, 1))
            ret_lpc_coeff = np.reshape(self._tr_data[i:i + batchsize, frame_length:frame_length + self._lpc_order],
                                       (batchsize, self._lpc_order, 1))
            b_res_x = np.reshape(self._tr_data[i:i + batchsize, frame_length + self._lpc_order:],
                                 (batchsize, frame_length, 1))
            yield ret, ret, ret_lpc_coeff, b_res_x

    def init_placeholder_end_to_end(self):
        x = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, frame_length, 1), name='x')
        x_ = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, frame_length, 1), name='x_')
        lr = tf.compat.v1.placeholder(dtype=tf.float32, shape=None, name='lr')
        the_share = tf.compat.v1.placeholder(dtype=tf.bool, shape=None, name='the_share')
        # the_share = tf.compat.v1.placeholder(dtype=tf.float32, shape=None, name='the_share')
        return x, x_, lr, the_share

    def _down_sampling_mod(self, the_input, the_stride=2):
        the_output = conv1d(the_input, self._bottleneck_kernel_and_dilation[2], filter_size=9, padding='SAME', 
                            dilation_rate=1, strides=the_stride, activation=None)
        the_output = activation_func(the_output)
        return the_output

    def _up_sampling_mod_helper(self, the_input, the_stride=2):
        """
        Conduct dimension permutation for upsampling
        Paper: Real-time single image and video super-resolution using an efficient 
               sub-pixel convolutional neural network
        """
        r = tf.reshape(the_input, (-1, the_input.shape[1], the_input.shape[2] // the_stride, the_stride))
        r = tf.keras.backend.permute_dimensions(r, (0, 1, 3, 2))
        r = tf.reshape(r, (-1, the_input.shape[1] * the_stride, the_input.shape[2] // the_stride))
        return r

    def _up_sampling_mod(self, the_input, the_stride=2):
        if resnet_type=='bottleneck':
            the_output = conv1d(the_input, the_input.shape[-1], filter_size=9, padding='SAME', dilation_rate=1,
                                strides=1,
                                activation=None)
        else:
            the_output = conv1d_depth(the_input, the_input.shape[-1], filter_size=9, padding='SAME', dilation_rate=1,
                                      strides=1,
                                      activation=None)

        the_output = activation_func(the_output)
        the_output = self._up_sampling_mod_helper(the_output, the_stride=the_stride)
        return the_output

    def _stack_bottleneck_blocks(self, compressed_bit, strides=1, is_post_up_samling=True, the_share=False, is_enc=True):
        """
        Stack pre-defined dilated bottleneck residual blocks
        """
        assert self._bottleneck_kernel_and_dilation[2] % strides == 0

        if compressed_bit.shape[-1] == 1:  # this basically is for the channel expansion.
            wide_layer = self._bottleneck_kernel_and_dilation[2]
        else:
            wide_layer = int(compressed_bit.shape[-1] / strides) if is_post_up_samling else compressed_bit.shape[-1]

        for i in range(len(
                self._bottleneck_kernel_and_dilation) - 4):  # the first one is the kernel # second is the wide layer
            flag = i == (len(self._bottleneck_kernel_and_dilation) - 5)
            if resnet_type == 'bottleneck':
                print(resnet_type)
                compressed_bit = the_bottleneck(compressed_bit,
                                                non_dilated_neck_kernel_size=self._bottleneck_kernel_and_dilation[1],
                                                dilated_neck_kernel_size=self._bottleneck_kernel_and_dilation[0],
                                                wide_layer=wide_layer,
                                                narrow_layer=self._bottleneck_kernel_and_dilation[3],
                                                dilation_rate=self._bottleneck_kernel_and_dilation[i + 4],
                                                is_last_flat=flag)
            else:
                print(resnet_type)
                compressed_bit = gated_bottleneck(compressed_bit,
                                                  non_dilated_neck_kernel_size=self._bottleneck_kernel_and_dilation[1],
                                                  dilated_neck_kernel_size=self._bottleneck_kernel_and_dilation[0],
                                                  wide_layer=wide_layer,
                                                  narrow_layer=self._bottleneck_kernel_and_dilation[3],
                                                  dilation_rate=self._bottleneck_kernel_and_dilation[i + 4],
                                                  is_last_flat=flag,
                                                  the_share=the_share)

        return compressed_bit

    def _the_encoder_in_each_module(self, the_input, the_stride, the_share):
        """
        The definition of neural encoder.
        """
        compressed_bit = change_channel(the_input, the_channel=self._bottleneck_kernel_and_dilation[2],
                                        # kernel_size=self._bottleneck_kernel_and_dilation[0],  #55,  # self._bottleneck_kernel_and_dilation[0],
                                        kernel_size=55,  # self._bottleneck_kernel_and_dilation[0],
                                        activation=None)
        compressed_bit = activation_func(compressed_bit)

        for i in the_stride:
            compressed_bit = self._stack_bottleneck_blocks(compressed_bit, is_post_up_samling=False, the_share=the_share)
            compressed_bit = self._down_sampling_mod(compressed_bit, the_stride=i)

        post_down_sampling_hidden = compressed_bit
        compressed_bit = self._stack_bottleneck_blocks(compressed_bit, is_post_up_samling=False, the_share=the_share)
        # compressed_bit = change_channel(compressed_bit, the_channel=1, kernel_size=9, activation=tf.nn.tanh)
        compressed_bit = change_channel(compressed_bit, the_channel=1, kernel_size=55, activation=tf.nn.tanh)
        return post_down_sampling_hidden, compressed_bit

    def _the_decoder_in_each_module(self, the_code, the_stride, the_share):
        """
        The definition of neural decoder.
        """
        compressed_bit = the_code
        # compressed_bit = change_channel(compressed_bit, the_channel=self._bottleneck_kernel_and_dilation[2],
        #                                 # kernel_size=self._bottleneck_kernel_and_dilation[0],  #55,  # self._bottleneck_kernel_and_dilation[0],
        #                                 kernel_size=55, # self._bottleneck_kernel_and_dilation[0],
        #                                 activation=None)
        # compressed_bit = activation_func(compressed_bit)
        for i in the_stride:
            compressed_bit = self._stack_bottleneck_blocks(compressed_bit, is_post_up_samling=False, the_share=the_share, is_enc=False)
            pre_up_sampling_hidden = compressed_bit
            compressed_bit = self._up_sampling_mod(compressed_bit, the_stride=i)

        compressed_bit = self._stack_bottleneck_blocks(compressed_bit, is_post_up_samling=False, the_share=the_share, is_enc=False)
        expand_back = change_channel(compressed_bit,
                                     the_channel=1,
                                     # kernel_size=self._bottleneck_kernel_and_dilation[0],
                                     kernel_size=55,
                                     activation=None)
        return pre_up_sampling_hidden, expand_back

    def computational_graph_end2end_quan_on(self, encoded, the_share, is_quan_on, number_bins, the_scope, the_strides):
        """
        The definition of the whole neural codec. 
        Softmax quantizer is called in between encoding and decoding. 
        """
        with tf.compat.v1.variable_scope(the_scope):
            alpha = tf.Variable(init_alpha, dtype=tf.float32, name='alpha')
            bins = tf.Variable(np.linspace(-1, 1, number_bins), dtype=tf.float32, name='bins')
            # bins = tf.Variable(mu_law_bins, dtype = tf.float32, name = 'bins')
            # encoded = mu_law_mapping(encoded, mu=conv_mu)
            hidden_1, floating_code = self._the_encoder_in_each_module(encoded, the_strides, the_share)
            print('floating_code', floating_code.shape)
            # floating_code = differential_coding_subtract(floating_code)
            if mu_law_transform:
                floating_code = mu_law_mapping(floating_code, mu=conv_mu)
            soft_assignment_3d, the_final_code = scalar_softmax_quantization(floating_code,
                                                                             alpha,
                                                                             bins,
                                                                             is_quan_on,
                                                                             the_share,
                                                                             frame_length // (2**len(the_strides)),
                                                                             number_bins)
            if mu_law_transform:
                the_final_code = inverse_mu_law_mapping(the_final_code, mu=conv_mu)

            hidden_2, expand_back = self._the_decoder_in_each_module(the_final_code, the_strides, the_share)
            print('model parameters:',
                  np.sum([np.prod(v.get_shape().as_list()) for v in tf.compat.v1.trainable_variables()]))
            # print('model parameters:',
            #       [v.shape for v in tf.compat.v1.trainable_variables()])
            print('end2end output shape:', expand_back.shape)
            # decode = inverse_mu_law_mapping(expand_back[:, :, 0], mu=conv_mu)
            return soft_assignment_3d, -1, -1, the_final_code[0, :, 0], expand_back[:, :, 0], alpha, bins, soft_assignment_3d
            # return soft_assignment_3d, -1, -1, bit_code_hard[0, :, 0], decode, alpha, bins, hard_assignment

    def computational_graph_end2end_quan_on_lpc(self,
                                                encoded,  # the input
                                                quan_lpc_coeff,  # the quantized LPC coefficients
                                                the_share,  # to set whether it's soft or hard.
                                                is_quan_on,  # to set whether the pretrain mode is on.
                                                number_bins,  # number of bins in the quantizer
                                                the_scope,  # scope of the variable
                                                the_strides):  # downsampling strides, represented as a list
        with tf.compat.v1.variable_scope(the_scope):
            alpha = tf.Variable(init_alpha, dtype=tf.float32, name='alpha')
            # alpha = tf.constant(init_alpha, dtype=tf.float32, name='alpha')
            bins = tf.Variable(np.linspace(-beta_boundary, beta_boundary, number_bins), dtype=tf.float32, name='bins')
            _, floating_code = self._the_encoder_in_each_module(encoded, the_strides, the_share)
            print('floating_code############', floating_code.shape)

            if mu_law_transform:
                floating_code = mu_law_mapping(floating_code, mu=conv_mu)
            soft_assignment_3d, the_final_code = scalar_softmax_quantization(floating_code,
                                                                             alpha,
                                                                             bins,
                                                                             is_quan_on,
                                                                             the_share,
                                                                             frame_length // (2**len(the_strides)),
                                                                             # frame_length // the_strides,
                                                                             number_bins)
            if mu_law_transform:
                the_final_code = inverse_mu_law_mapping(the_final_code, mu=conv_mu)

            _, expand_back = self._the_decoder_in_each_module(the_final_code, the_strides, the_share)
            print('model parameters:',
                  np.sum([np.prod(v.get_shape().as_list()) for v in tf.compat.v1.trainable_variables()]))
            print('end2end output shape:', expand_back.shape)
            # print('model parameters:', [np.prod(v.get_shape().as_list()) for v in tf.compat.v1.trainable_variables()])
            print('model parameters:', [v.get_shape() for v in tf.compat.v1.trainable_variables()])
            #synthesized_batch = tf.compat.v1.py_func(lpc_synthesizer_tr,
            #                                         [quan_lpc_coeff, expand_back[:, :, 0] / self._res_scalar],
            #                                         [tf.float32])[0]

            return soft_assignment_3d, -1, -1, the_final_code[0, :, 0], expand_back[:, :, 0], alpha, bins

    def computational_graph_end2end_quan_on_lpc_vq(self,
                                                   encoded,  # the input
                                                   quan_lpc_coeff,  # the quantized LPC coefficients
                                                   the_share,  # to set whether it's soft or hard.
                                                   is_quan_on,  # to set whether the pretrain mode is on.
                                                   number_bins,  # number of bins in the quantizer
                                                   the_scope,  # scope of the variable
                                                   the_strides):  # downsampling strides, represented as a list
        with tf.compat.v1.variable_scope(the_scope):
            alpha = tf.Variable(init_alpha, dtype=tf.float32, name='alpha')
            code_len = code_len_val
            a, b = np.linspace(-0.1, 0.1, 32), np.linspace(-0.1, 0.1, 32)
            # a, b = np.linspace(-1, 1, 32), np.linspace(-1, 1, 32)
            two_dim_bins = []
            for r in itertools.product(a, b):
                two_dim_bins.append([r[0], r[1]])

            bins = tf.Variable(two_dim_bins, dtype=tf.float32, name='bins')

            _, floating_code = self._the_encoder_in_each_module(encoded, the_strides, the_share)
            print('floating_code############', floating_code.shape)

            soft_assignment_3d, the_final_code = vector_softmax_quantization(floating_code,
                                                                             alpha,
                                                                             bins,
                                                                             is_quan_on,
                                                                             the_share,
                                                                             0,
                                                                             code_len)
            _, expand_back = self._the_decoder_in_each_module(the_final_code, the_strides, the_share)
            print('model parameters:',
                  np.sum([np.prod(v.get_shape().as_list()) for v in tf.compat.v1.trainable_variables()]))
            print('end2end output shape:', expand_back.shape)

            return soft_assignment_3d, -1, -1, the_final_code[0, :, 0], expand_back[:, :, 0], alpha, bins

    def computational_graph_end2end_quan_on_vq(self, encoded, the_share, is_quan_on, number_bins, the_scope, the_strides):
        """
        The definition of the whole neural codec. 
        Softmax quantizer is called in between encoding and decoding. 
        """
        with tf.compat.v1.variable_scope(the_scope):
            alpha = tf.Variable(init_alpha, dtype=tf.float32, name='alpha')
            # alpha = tf.constant(init_alpha, dtype=tf.float32, name='alpha')
            code_len = code_len_val
            # code_book_size = code_book_size_val
            # the_whole_codebook = np.reshape(np.load('pretrained_codebook.npy'), (-1, bottle_neck_size // code_len))
            # the_list = np.random.choice(the_whole_codebook.shape[0], code_book_size, replace=False)
            # bins = tf.Variable(np.reshape(self._tr_data[the_list, :], (-1, 256)), dtype=tf.float32, name='bins')

            a, b = np.linspace(-0.1, 0.1, 32), np.linspace(-0.1, 0.1, 32)
            # a, b = np.linspace(-1, 1, 32), np.linspace(-1, 1, 32)
            two_dim_bins = []
            for r in itertools.product(a, b):
                two_dim_bins.append([r[0], r[1]])

            bins = tf.Variable(two_dim_bins, dtype=tf.float32, name='bins')

            # bins = tf.Variable(the_whole_codebook[the_list, :], dtype=tf.float32, name='bins')
            #bins = tf.constant(np.reshape(np.linspace(-1, 1, code_book_size_val), (code_book_size_val, 1)),
            #                   dtype=tf.float32, name='bins')
            #bins = tf.Variable(np.reshape(np.linspace(-1, 1, code_book_size_val), (code_book_size_val, 1)),
            #                   dtype=tf.float32, name='bins')
            # bins = tf.constant(np.load('pretrained_codebook_lpc.npy')[the_list, :], dtype=tf.float32, name='bins')
            # bins = tf.constant(np.reshape(self._tr_data[the_list, :], (-1, 256)), dtype=tf.float32, name='bins')
            hidden_1, floating_code = self._the_encoder_in_each_module(encoded, the_strides, the_share)
            print('floating_code', floating_code.shape)

            # the_input_of_decoder = vector_softmax_quantization(floating_code, alpha, bins, the_share, top_k=1)
            # vector_softmax_quantization(floating_code, alpha, bins, is_pretrain, top_k, code_len):
            # the_output_of_encoder,
            soft_assignment, the_input_of_decoder = vector_softmax_quantization(floating_code,
                                                                                alpha,
                                                                                bins,
                                                                                is_quan_on,
                                                                                the_share,
                                                                                0,
                                                                                code_len)
            hidden_2, expand_back = self._the_decoder_in_each_module(the_input_of_decoder, the_strides, the_share)
            print('model parameters:',
                  np.sum([np.prod(v.get_shape().as_list()) for v in tf.compat.v1.trainable_variables()]))
            # print('model parameters:',
            #       [v for v in tf.compat.v1.trainable_variables()])
            return soft_assignment, -1, floating_code[:, :, 0], the_input_of_decoder[:, :, 0], expand_back[:, :, 0], \
                   alpha, bins, expand_back[:, :, 0]


    def model_training(self, sess, x, x_, lr, tau, the_share, is_quan_on, encoded, loss1, mfcc_loss, quan_loss,
                       ent_loss, trainop2_list, decoded, alpha, bins, saver, the_learning_rate, epoch, flag,
                       interested_var=None, save_id='', the_tau_val=1.0):
        print('model_training is called.', flag)
        ave_snr, ave_stoi, ave_pesq = 0, 0, 0
        init_tau = the_tau_val
        init_tau_1 = the_tau_val
        init_tau_2 = the_tau_val

        for i in range(epoch):
            print('-----------------------')
            if flag == 'pretrain':
                is_quan_on_val = 0.0 if i < self._pretrain_step else 1.0
                # the_share_val = True
                if i < self._pretrain_step:
                    trainop2 = trainop2_list[0]
                    print('no quan op is used')
                else:
                    trainop2 = trainop2_list[1]
                    print('quan op', init_tau)
            else:
                is_quan_on_val = 1.0
                # the_share_val = False
                trainop2 = trainop2_list[1]
                print(flag + ' is working.', init_tau)

            print('Epoch ----------------------- ', i)

            start = time.perf_counter()

            for b_x, b_x_ in self._generate_one_epoch_end2end(self._tr_data, self._tr_data, self._batch_size):
                sess.run(trainop2, feed_dict={x: b_x, x_: b_x_, lr: the_learning_rate, the_share: True,
                                              # tau: init_tau,
                                              tau: np.reshape([init_tau_1, init_tau_2], (1, 2)),
                                              is_quan_on: is_quan_on_val})
            elapsed = time.perf_counter() - start
            np.random.shuffle(self._tr_data)

            # For every a few epoch, get validation results.
            if i % 1 == 0:
                if type(decoded) == list:
                    decoded = np.sum(decoded, axis=0)
                ave_snr, ave_si_snr, ave_stoi, ave_pesq, ave_linearity, _quan_loss, ent_codec_1, ent_codec_2, \
                fully_entropy = self.end2end_eval(sess, x, x_, lr, the_share, is_quan_on, flag, loss1,
                                                  encoded, decoded, alpha, bins, 100, i,
                                                  interested_var)
                print(
                    'Epoch %3d: SNR: %7.5f dB  Si-SNR: %7.5f dB  STOI: %6.5f    PESQ: %6.5f   linearity: %6.5f  modelid: %s  '
                    '_quan_loss: %6.5f,  fully_entropy: %6.5f , time: %.3f, tau: %.3f, tau_1: %.3f, tau_2: %.3f' % (
                    i, ave_snr, ave_si_snr, ave_stoi, ave_pesq,
                    ave_linearity, self._rand_model_id,
                    _quan_loss, fully_entropy, elapsed,
                    init_tau, init_tau_1, init_tau_2))

                # ave_snr, ave_si_snr, ave_stoi, ave_pesq, ave_linearity, _quan_loss, fully_snr, fully_pesq, fully_entropy = self.end2end_eval_not_pretrain(
                #     sess, x, x_, lr, the_share, is_quan_on, flag, loss1, encoded, decoded, alpha, bins, 100, i,
                #     interested_var)
                # print(
                #     'Epoch %3d: SNR: %7.5f dB  Si-SNR: %7.5f dB  STOI: %6.5f    PESQ: %6.5f   linearity: %6.5f  modelid: %s  '
                #     '_quan_loss: %6.5f,  fully_entropy: %6.5f , time: %.3f, tau: %.3f' % (
                #     i, ave_snr, ave_si_snr, ave_stoi, ave_pesq,
                #     ave_linearity, self._rand_model_id,
                #     _quan_loss, fully_entropy, elapsed,
                #     init_tau))

                # print('the_mu: ', interested_var[-4].eval())
                #'_quan_loss: %6.5f,  fully_entropy: %6.5f , time: %.3f, tau: %.3f' % (
                #i, np.mean(ave_snr), np.mean(ave_si_snr), np.mean(ave_stoi), np.mean(ave_pesq), np.mean(ave_linearity), self._rand_model_id, _quan_loss,
                #                                                                                     np.mean(fully_entropy), elapsed, init_tau))

            ent_change = 0.015
            if flag == 'finetune':
                ent_change *= 1
                target_1 = 1.5
                target_2 = 2.5
                # target_1 = 2.0
                # target_2 = 2.0
                # target_1 = 2.5
                # target_2 = 1.5
            # ent_change = 0.0
                print(ent_codec_1, ent_codec_2)
                if ent_codec_1 > target_1:
                    init_tau_1 += ent_change
                if ent_codec_1 < target_1:
                    init_tau_1 -= ent_change
                if ent_codec_2 > target_2:
                    init_tau_2 += ent_change
                if ent_codec_2 < target_2:
                    init_tau_2 -= ent_change
            else:
                if fully_entropy > self._target_entropy:
                    init_tau += ent_change
                elif fully_entropy < self._target_entropy:
                    init_tau -= ent_change
            print('Tau: %7.5f, Tau_1: %7.5f, Tau_2: %7.5f'% (init_tau, init_tau_1, init_tau_2))

            self._write_to_file_and_update_to_display('Epoch %3d: SNR: %7.5f dB Si-SNR: %7.5f dB STOI: '
                                                      '%6.5f PESQ: %6.5f _quan_loss: %6.5f'
                                                      'tau: %6.5f   '
                                                      'fully_entropy: %6.5f \n' % (i,
                                                                                   ave_snr,#np.mean(ave_snr),
                                                                                   ave_si_snr,#np.mean(ave_si_snr),
                                                                                   ave_stoi,#np.mean(ave_stoi),
                                                                                   ave_pesq,#np.mean(ave_pesq),
                                                                                   _quan_loss,#np.mean(_quan_loss),
                                                                                   init_tau,#init_tau,
                                                                                   fully_entropy))#np.mean(fully_entropy)))

        # pretrained_codebook = np.random.rand(code_book_size_val, bottle_neck_size)
        # the_list = np.random.choice(self._tr_data.shape[0], code_book_size_val)
        # selected_tr_data = self._tr_data[the_list, :]
        # for j in range(selected_tr_data.shape[0]):
        #     if j%1000==0:
        #         print(j)
        #     feed_x = np.reshape(self._tr_data[j, :], (1, frame_length, 1))
        #     _encoded = sess.run(
        #         [encoded], feed_dict=
        #         {x: feed_x,
        #          x_: feed_x,
        #          lr: 0.0,
        #          the_share: False,
        #          is_quan_on: 1.0})
        #     pretrained_codebook[j, :] = np.array(_encoded)#.flatten()
        # np.save('pretrained_codebook.npy', pretrained_codebook)
        saver.save(sess, "./check/model_bnn_ac_" + self._rand_model_id + '_' + save_id + ".ckpt")
        print('Model saved!')

    def model_training_lpc(self, sess, x, x_, lpc_x, lr, synthesized, res_x, tau, the_share, is_quan_on, encoded, loss1,
                           mfcc_loss, quan_loss, ent_loss, trainop2_list, decoded, alpha, bins, saver,
                           the_learning_rate, epoch,  flag, interested_var=None, save_id='', the_tau_val=1.0):
        ave_snr, ave_stoi, ave_pesq = 0, 0, 0
        init_tau = the_tau_val
        for i in range(epoch):
            # print(i, '===========iter')
            # #update w and b.
            print('-----------model_training_lpc-------------')
            if flag == 'pretrain':
                is_quan_on_val = 0.0 if i < self._pretrain_step else 1.0
                if i < self._pretrain_step:
                    trainop2 = trainop2_list[0]
                    print('no quan op is used')

                else:
                    trainop2 = trainop2_list[1]  # normal ent
                    print('quan op', init_tau)

            else:
                is_quan_on_val = 1.0
                trainop2 = trainop2_list[1]
                print('greedy_follower is working.', init_tau)

            print('-----------------------')

            start = time.perf_counter()
            if self._is_cq and (i % 30 == 0 and i != 0 or i == epoch - 3):
            # if self._is_cq and (i % 3 == 0 and i != 0):
                print('=============Recalculating the residuals...=============')
                # time.sleep(10.0)
                self._update_lpc_residual()
                save_path = saver.save(sess,"./check/model_bnn_ac_" + self._rand_model_id + '_' + save_id + ".ckpt")
                print('Model saved!')
            else:
                for b_x, b_x_, b_lpc_coeff, b_res_x in self._generate_one_epoch_end2end_lpc_fast(self._tr_data, self._tr_data,
                                                                                                 self._batch_size):
                    sess.run(trainop2, feed_dict={x: b_x,
                                                  x_: b_x_,
                                                  lpc_x: b_lpc_coeff,
                                                  res_x: b_res_x,
                                                  lr: the_learning_rate,
                                                  the_share: True,
                                                  tau:init_tau,
                                                  is_quan_on: is_quan_on_val})


            elapsed = time.perf_counter() - start
            if i % 1 == 0:
                if type(decoded) == list:
                    decoded = np.sum(decoded, axis=0)
                ave_snr, ave_stoi, ave_pesq, ave_linearity, _quan_loss, fully_snr, fully_pesq, fully_entropy = \
                    self.end2end_eval_lpc(sess,
                                          x=x,
                                          x_=x_,
                                          lr=lr,
                                          the_share=the_share,
                                          lpc_x=lpc_x,
                                          synthesized=synthesized,
                                          res_x=res_x,
                                          is_quan_on=is_quan_on,
                                          flag=flag,
                                          loss1=loss1,
                                          encoded=encoded,
                                          decoded=decoded,
                                          alpha=alpha,
                                          bins=bins,
                                          how_many=100,
                                          the_epoch=i,
                                          interested_var=interested_var)

            print(
                'Epoch %3d: SNR: %7.5f dB    STOI: %6.5f    PESQ: %6.5f   linearity: %6.5f  modelid: %s  _quan_loss: %6.5f,  fully_entropy: %6.5f , time: %.3f, tau: %.3f' % (
                i, ave_snr, ave_stoi, ave_pesq, ave_linearity, self._rand_model_id, _quan_loss, fully_entropy, elapsed,
                init_tau))
            self._write_to_file_and_update_to_display(
                'Epoch %3d: SNR: %7.5f dB    STOI: %6.5f   PESQ: %6.5f   _quan_loss: %6.5f  fully_snr: %6.5f   fully_pesq: %6.5f  fully_entropy: %6.5f \n' % (
                i, ave_snr, ave_stoi, ave_pesq, _quan_loss, fully_snr, fully_pesq, fully_entropy))

            ent_change = 0.015
            # ent_change = 0.5
            # if flag == 'pretrain':
            # if flag == 'ent_control_on':
            if is_quan_on_val == 1.0:
                if fully_entropy > self._target_entropy + 0.05:
                    init_tau += ent_change
                elif fully_entropy < self._target_entropy:
                    init_tau -= ent_change * 3
                print('tau:', init_tau)

            # if flag == 'pretrain' and fully_pesq > 4.4 and fully_entropy <= self._target_entropy:
            #   break

            if flag == 'finetuning_follower_all' and fully_entropy < self._target_entropy:  # LPC
                break
                # break
        if flag == 'quan' and epoch == 0:
            pass
            # ave_snr, ave_stoi, ave_pesq, ave_linearity, _ = self.end2end_eval(sess, x, x_, lr, the_share, loss1, encoded, decoded, alpha, bins, 100, 30, interested_var)
            # print('Epoch %3d: SNR: %7.5f dB    STOI: %6.5f    PESQ: %6.5f   linearity: %6.5f modelid: %s' %(0, ave_snr, ave_stoi, ave_pesq, ave_linearity, self._rand_model_id))
        elif epoch != 0:
            save_path = saver.save(sess, "./check/model_bnn_ac_" + self._rand_model_id + '_' + save_id + ".ckpt")
            print('Model saved!')
        else:
            print('Model not saved.')

    def end2end_eval(self, sess, x, x_, lr, the_share, is_quan_on, flag, loss1, encoded, decoded, alpha, bins,
                     how_many, the_epoch, interested_var=None):
        """
        Conduct validation during model training. 
        """
        how_many = 10  # 10 if the_epoch > 100 else 1  # len(self._sep_val)
        min_len, snr_list, si_snr_list = np.array([0] * how_many), np.array([0.0] * how_many), np.array([0.0] * how_many)
        the_stoi, the_pesqs = np.array([0.0] * how_many), np.array([0.0] * how_many)
        fully_snr_list, fully_the_pesqs, fully_the_entropy = np.array([0.0] * how_many), np.array(
            [0.0] * how_many), np.array([0.0] * how_many)
        entropy_per_codec = np.array([])
        _quan_loss_arr = np.array([0.0] * how_many)
        the_linearitys = np.array([0.0] * how_many)
        all_loss = [0.0] * how_many
        _alpha, _bins = 0, 0

        for i in range(how_many):
            per_sig, the_std = self._load_sig(self._sep_val[i])

            segments_per_utterance = utterance_to_segment(per_sig, True)
            _decoded_sig = np.array(
                [0.0] * (frame_length + (frame_length - overlap_each_side) * (segments_per_utterance.shape[0] - 1)))
            # code_segment_len = int(frame_length / (2**len(self._the_strides)))
            code_segment_len = bottle_neck_size
            _encoded_sig = np.array(
                [0.0] * (bottle_neck_size + (bottle_neck_size) * (segments_per_utterance.shape[0] - 1))).astype(
                np.float32)
            _quan_loss_arr_each = np.array([0.0] * segments_per_utterance.shape[0])
            entropy_per_segment = np.array([0.0] * segments_per_utterance.shape[0])
            lpc_entropy = np.array([0.0] * segments_per_utterance.shape[0])
            each_entropy = np.array([])


            for j in range(segments_per_utterance.shape[0]):
                feed_x = np.reshape(segments_per_utterance[j], (1, frame_length, 1))
                _interested_var, the_loss, _encoded, _decoded, _alpha, _bins = sess.run(
                    [interested_var, loss1, encoded, decoded, alpha, bins], feed_dict=
                    {x: feed_x,
                     x_: feed_x,
                     lr: 0.0,
                     the_share: False,
                     is_quan_on: 1.0})  # share is False at test time


                # the_loss, _encoded, _decoded, _alpha, _bins = sess.run(
                #     [loss1, encoded, decoded, alpha, bins], feed_dict=
                #     {x: feed_x,
                #      x_: feed_x,
                #      lr: 0.0,
                #      the_share: False,
                #      is_quan_on: 1.0})  # share is False at test time

                # mu
                # _decoded = inverse_mu_law_mapping(_decoded)
                _decoded_sig[j * (frame_length - overlap_each_side): j * (
                frame_length - overlap_each_side) + frame_length] += hann_process(_decoded.flatten(), j,
                                                                                  segments_per_utterance.shape[0])
                # _encoded_sig[j * bottle_neck_size: j * bottle_neck_size + bottle_neck_size] += _encoded
                _quan_loss_arr_each[j] = _interested_var[2]

                if len(_interested_var[4]) <= 3:
                    each_entropy = np.append(each_entropy, np.array(_interested_var[4]))
                if isinstance(_interested_var[5], np.ndarray):
                    entropy_per_segment[j] = np.mean(_interested_var[5])
                else:
                    entropy_per_segment[j] = _interested_var[5]
            if flag == 'finetune':
                ent_list = np.array([256]) if len(np.array(_interested_var[4])) == 1 else np.array([256, 256])
                each_entropy = np.reshape(each_entropy, (-1, len(np.array(_interested_var[4]))))
                # each_entropy = np.mean(np.sum(ent_list / np.sum(ent_list) * np.mean(each_entropy, axis=0)))
                each_entropy = np.mean(each_entropy, axis=0)
                entropy_per_codec = np.append(entropy_per_codec, each_entropy)

            per_sig *= the_std
            _decoded_sig *= the_std
            # np.save('the_code_' + self._rand_model_id + '.npy', _encoded_sig)
            fully_the_entropy[i] = np.mean(entropy_per_segment)
            # lpc_entropy[i] = _interested_var[-2]
            min_len[i], si_snr_list[i], snr_list[i], the_stoi[i], the_pesqs[i], the_linearitys[i] = \
                eval_metrics(per_sig, _decoded_sig, self._rand_model_id)
            # print('signal ', i, 'snr: ', snr_list[i])
            _quan_loss_arr[i] = np.mean(_quan_loss_arr_each)

        np.save('bins' + self._rand_model_id + str(the_epoch) + '.npy', _interested_var[4])
        sdr_return_it = np.sum(min_len * snr_list / np.sum(min_len))
        si_snr_return_it = np.sum(min_len * si_snr_list / np.sum(min_len))
        stoi_return_it = np.sum(min_len * the_stoi / np.sum(min_len))
        pesq_return_it = np.sum(min_len * the_pesqs / np.sum(min_len))
        print(si_snr_list, si_snr_return_it)
        print(the_pesqs, pesq_return_it)
        if flag == 'finetune':
            print(entropy_per_codec[::2])
            print(entropy_per_codec[1::2])
            self._write_to_file_and_update_to_display('codec-1: ' + str(entropy_per_codec[::2]))
            self._write_to_file_and_update_to_display('codec-2: ' + str(entropy_per_codec[1::2]))
        linearity_return_it = np.sum(min_len * the_linearitys / np.sum(min_len))
        quan_return_it = np.sum(min_len * _quan_loss_arr / np.sum(min_len))
        # fully_sdr_return_it = np.sum(min_len * fully_snr_list / np.sum(min_len))
        ent_codec_1 = np.sum(min_len * entropy_per_codec[::2] / np.sum(min_len))
        ent_codec_2 = np.sum(min_len * entropy_per_codec[1::2] / np.sum(min_len))
        fully_entropy_return_it = np.sum(min_len * fully_the_entropy / np.sum(min_len))

        return sdr_return_it, si_snr_return_it, stoi_return_it, pesq_return_it, linearity_return_it, quan_return_it, \
               ent_codec_1, ent_codec_2, fully_entropy_return_it

    def end2end_eval_lpc(self, sess, x, x_, lr, the_share, lpc_x, synthesized, res_x, is_quan_on, flag, loss1, encoded,
                         decoded, alpha, bins, how_many, the_epoch, interested_var=None):
        # if the_epoch < 95:
        # 	how_many = 20
        how_many = 1  # 10#10
        min_len, snr_list = np.array([0] * how_many), np.array([0.0] * how_many)
        the_stoi, the_pesqs = np.array([0.0] * how_many), np.array([0.0] * how_many)

        fully_snr_list, fully_the_pesqs, fully_the_entropy = np.array([0.0] * how_many), np.array(
            [0.0] * how_many), np.array([0.0] * how_many)

        _quan_loss_arr = np.array([0.0] * how_many)
        the_linearitys = np.array([0.0] * how_many)
        all_loss = [0.0] * how_many
        
        _alpha, _bins = 0, 0

        # selected_test_set = np.random.choice(1500, how_many, replace=False)
        for i in range(how_many):
            per_sig, the_std = self._load_sig_lpc(self._sep_val[i])
            # print(np.max(per_sig), 'max')
            # high pass and pre-emphasis on signals
            per_sig_highpass_empha_filtered = np.array(list(empha_filter(highpass_filter(per_sig))))
            segments_per_utterance = utterance_to_segment(per_sig_highpass_empha_filtered, True)

            _res_x_sig = np.array(
                [0.0] * (frame_length + (frame_length - overlap_each_side) * (segments_per_utterance.shape[0] - 1)))
            _decoded_sig = np.array(
                [0.0] * (frame_length + (frame_length - overlap_each_side) * (segments_per_utterance.shape[0] - 1)))
            _synthesized_sig = np.array(
                [0.0] * (frame_length + (frame_length - overlap_each_side) * (segments_per_utterance.shape[0] - 1)))

            # print(len(per_sig), len(per_sig_highpass_empha_filtered), len(_synthesized_sig), len(_res_x_sig))

            per_utterance_res = np.array(
                [0.0] * (frame_length + (frame_length - overlap_each_side) * (segments_per_utterance.shape[0] - 1)))

            # code_segment_len = int(frame_length / self._the_strides[0])
            # code_segment_len = frame_length // (2 ** len(self._the_strides))
            the_stride_ind = 1 if flag == 'the_follower' else 0
            code_segment_len = 128 if self._the_strides[the_stride_ind] == 4 else 256
            # code_segment_len = 64

            _encoded_sig = np.array(
                [0.0] * (code_segment_len + (code_segment_len) * (segments_per_utterance.shape[0] - 1))).astype(
                np.float32)
            _quan_loss_arr_each = np.array([0.0] * segments_per_utterance.shape[0])
            all_entropy_fully = np.array([0.0] * segments_per_utterance.shape[0])
            each_entropy = np.array([])

            # just LPC , no high pass and pre-emphasis.
            segments_per_utterance_coeff = lpc_analysis_at_test(segments_per_utterance, self._lpc_order)
            segments_per_utterance = utterance_to_segment(per_sig_highpass_empha_filtered[256:], True)
            print(segments_per_utterance.shape)
            for j in range(segments_per_utterance.shape[0] - 2):
                feed_x = np.reshape(segments_per_utterance[j], (1, frame_length, 1))
                feed_lpc_coeff = np.reshape(segments_per_utterance_coeff[j], (1, self._lpc_order, 1))

                _interested_var, the_loss, _encoded, _decoded, _synthesized, _res_x, quan_lpc_coeff, _alpha, _bins = \
                    sess.run([interested_var, loss1, encoded, decoded, synthesized, res_x, lpc_x, alpha, bins],
                             feed_dict={x: feed_x,
                                        x_: feed_x,
                                        lpc_x: feed_lpc_coeff,  # the lpc coefficients are preprocessed, unquantized, in lsf.
                                        lr: 0.0,
                                        is_quan_on: 1.0,
                                        the_share: False})  # share is 1 means it's soft, 0 means it's hard

                _decoded_sig[j * (frame_length - overlap_each_side): j * (
                frame_length - overlap_each_side) + frame_length] += hann_process(_decoded.flatten(), j,
                                                                                  segments_per_utterance.shape[0])
                _res_x_sig[j * (frame_length - overlap_each_side): j * (
                frame_length - overlap_each_side) + frame_length] += hann_process(_res_x.flatten(), j,
                                                                                  segments_per_utterance.shape[0])
                _synthesized_sig[j * (frame_length - overlap_each_side): j * (
                frame_length - overlap_each_side) + frame_length] += hann_process(_synthesized[0, :], j,
                                                                                  segments_per_utterance.shape[0])

                np.save('lpc_coeff_lsf_bins_updated_' + self._rand_model_id + '.npy', _interested_var[3])
                _encoded_sig[j * (code_segment_len): j * (code_segment_len) + code_segment_len] += _encoded  # [0,:]
                _quan_loss_arr_each[j] = _interested_var[2]
                #


                # print(_interested_var[6], np.sum(ent_list/np.sum(ent_list) * _interested_var[6]))
                # each_entropy.append(np.sum(ent_list/np.sum(ent_list) * _interested_var[6]))
                # each_entropy.append(np.array(_interested_var[6]))
                if len(_interested_var[6]) <= 3:
                    each_entropy = np.append(each_entropy, np.array(_interested_var[6]))

                if isinstance(_interested_var[5], np.ndarray):
                    all_entropy_fully[j] = np.mean(_interested_var[5])
                else:
                    all_entropy_fully[j] = _interested_var[5]
            ent_list = np.array([16.0, 256]) if len(np.array(_interested_var[6])) == 2 else np.array([16.0, 256, 256])
            each_entropy = np.reshape(each_entropy, (-1, len(np.array(_interested_var[6]))))
            average_entropy_each_mod = np.mean(each_entropy, axis=0)
            print(average_entropy_each_mod)

            self._write_to_file_and_update_to_display(str(average_entropy_each_mod) + ', ')
            each_entropy = np.mean(np.sum(ent_list / np.sum(ent_list) * np.mean(each_entropy, axis=0)))
            # print(each_entropy)
            # print(np.max(_res_x_sig), 'max')
            per_sig *= the_std

            _synthesized_sig = np.array(list((1 / empha_filter)(_synthesized_sig)))

            _synthesized_sig *= the_std

            fully_the_entropy[i] = each_entropy  # np.mean(all_entropy_fully)

            # min_len[i], snr_list[i], the_stoi[i], the_pesqs[i], the_linearitys[i] = self.end2end_final_eval(per_sig, _synthesized_sig)
            min_len[i], _, snr_list[i], the_stoi[i], the_pesqs[i], the_linearitys[i] = \
                eval_metrics(per_sig[256:], _synthesized_sig, self._rand_model_id)
            _quan_loss_arr[i] = np.mean(_quan_loss_arr_each)
            print(snr(_res_x_sig, _decoded_sig)[1], snr_list[i])
            self._write_to_file_and_update_to_display(str(snr(_res_x_sig, _decoded_sig)[1]) + ',' + str(snr_list[i]) + ', ' )

        sdr_return_it = np.sum(min_len * snr_list / np.sum(min_len))
        stoi_return_it = np.sum(min_len * the_stoi / np.sum(min_len))
        pesq_return_it = np.sum(min_len * the_pesqs / np.sum(min_len))
        linearity_return_it = np.sum(min_len * the_linearitys / np.sum(min_len))
        quan_return_it = np.sum(min_len * _quan_loss_arr / np.sum(min_len))

        fully_sdr_return_it = np.sum(min_len * fully_snr_list / np.sum(min_len))
        fully_pesq_return_it = np.sum(min_len * fully_the_pesqs / np.sum(min_len))
        fully_entropy_return_it = np.sum(min_len * fully_the_entropy / np.sum(min_len))
        return sdr_return_it, stoi_return_it, pesq_return_it, linearity_return_it, quan_return_it, fully_sdr_return_it, \
               fully_pesq_return_it, fully_entropy_return_it

    def one_ae(self):
        """
        Train one neural codec. 
        """
        with tf.Graph().as_default():
            x, x_, lr, the_share = self.init_placeholder_end_to_end()
            tau = tf.compat.v1.placeholder(dtype=tf.float32, shape=None, name='tau')
            is_quan_on = tf.compat.v1.placeholder(dtype=tf.float32, shape=None, name='is_quan_on')
            _softmax_assignment, weight, decoded_fully, encoded, decoded, alpha, bins, _hard_assignment = \
                self.computational_graph_end2end_quan_on(
                    x,
                    the_share,
                    is_quan_on,
                    self._num_bins_for_follower[0],
                    'scope_1',
                    self._the_strides)

            time_loss = mse_loss(decoded, x_[:, :, 0])
            # the_mu = tf.Variable(15, dtype=tf.float32, name='the_mu')

            freq_loss = mfcc_loss(decoded, x_[:, :, 0])  # + tp_psd_corr(x_[:, :, 0] - decoded, x_[:, :, 0])
            quantization_loss = quan_loss(_softmax_assignment)
            ent_loss = entropy_coding_loss(_softmax_assignment)
            loss_no_quan = self._coeff_term[0] * time_loss + self._coeff_term[1] * freq_loss
            loss_quan_init = self._coeff_term[0] * time_loss + \
                             self._coeff_term[1] * freq_loss + \
                             self._coeff_term[2] * quantization_loss + \
                             tau * ent_loss  # self._coeff_term[3] * ent_loss  # tau * ent_loss  # self._coeff_term[3] * ent_loss  #

            # loss_no_quan = self._coeff_term[0] * time_loss
            # loss_quan_init = self._coeff_term[0] * time_loss
            trainop2_no_quan = tf.compat.v1.train.AdamOptimizer(lr, beta2=0.999).\
                minimize(loss_no_quan, var_list=tf.compat.v1.trainable_variables())
            trainop2_quan_init = tf.compat.v1.train.AdamOptimizer(lr, beta2=0.999).\
                minimize(loss_quan_init, var_list=tf.compat.v1.trainable_variables())
            trainop2_list = [trainop2_no_quan, trainop2_quan_init]

            saver = tf.compat.v1.train.Saver()
            interested_var = [time_loss, freq_loss, quantization_loss, ent_loss, bins, ent_loss, encoded]
            with tf.compat.v1.Session() as sess:
                sess.run(tf.compat.v1.global_variables_initializer())
                self.model_training(sess, x=x, x_=x_, lr=lr, the_share=the_share, tau=tau,
                                    is_quan_on=is_quan_on, encoded=encoded, loss1=time_loss, mfcc_loss=freq_loss,
                                    quan_loss=quan_loss, ent_loss=ent_loss, trainop2_list=trainop2_list,
                                    decoded=decoded, alpha=alpha,
                                    bins=bins, saver=saver,
                                    the_learning_rate=self._learning_rate_tanh, epoch=self._epoch_tanh,
                                    flag='pretrain', interested_var=interested_var, save_id='',
                                    the_tau_val=self._coeff_term[3])

    def one_ae_vq(self):
        """
        Train one neural codec. 
        """
        with tf.Graph().as_default():
            x, x_, lr, the_share = self.init_placeholder_end_to_end()
            tau = tf.compat.v1.placeholder(dtype=tf.float32, shape=None, name='tau')
            is_quan_on = tf.compat.v1.placeholder(dtype=tf.float32, shape=None, name='is_quan_on')
            _softmax_assignment, weight, before_vq, encoded, decoded, alpha, bins, _hard_assignment = \
                self.computational_graph_end2end_quan_on_vq(
                    x,
                    the_share,
                    is_quan_on,
                    self._num_bins_for_follower[0],
                    'scope_1',
                    self._the_strides)

            the_quan_loss = mse_loss(encoded, before_vq)
            # the_quan_loss = snr(encoded[0,:], before_vq[0,:])
            time_loss = mse_loss(decoded, x_[:, :, 0]) # + the_quan_loss
            freq_loss = mfcc_loss(decoded, x_[:, :, 0])  # + tp_psd_corr(x_[:, :, 0] - decoded, x_[:, :, 0])
            quantization_loss = quan_loss(_softmax_assignment)
            ent_loss = entropy_coding_loss(_softmax_assignment)
            loss_no_quan = self._coeff_term[0] * time_loss + self._coeff_term[1] * freq_loss
            loss_quan_init = self._coeff_term[0] * time_loss + \
                             self._coeff_term[1] * freq_loss + \
                             self._coeff_term[2] * quantization_loss + \
                             tau * ent_loss  # self._coeff_term[3] * ent_loss  #

            trainop2_no_quan = tf.compat.v1.train.AdamOptimizer(lr, beta1=0.9, beta2=0.999).\
                minimize(loss_no_quan, var_list=tf.compat.v1.trainable_variables())
            trainop2_quan_init = tf.compat.v1.train.AdamOptimizer(lr, beta1=0.9, beta2=0.999).\
                minimize(loss_quan_init, var_list=tf.compat.v1.trainable_variables())
            trainop2_list = [trainop2_no_quan, trainop2_quan_init]

            saver = tf.compat.v1.train.Saver()
            interested_var = [time_loss, freq_loss, the_quan_loss, quantization_loss, ent_loss, ent_loss]
            with tf.compat.v1.Session() as sess:
                sess.run(tf.compat.v1.global_variables_initializer())
                self.model_training(sess, x=x, x_=x_, lr=lr, the_share=the_share, tau=tau,
                                    is_quan_on=is_quan_on, encoded=encoded, loss1=time_loss, mfcc_loss=freq_loss,
                                    quan_loss=quan_loss, ent_loss=ent_loss, trainop2_list=trainop2_list,
                                    decoded=decoded, alpha=alpha,
                                    bins=bins, saver=saver,
                                    the_learning_rate=self._learning_rate_tanh, epoch=self._epoch_tanh,
                                    flag='pretrain', interested_var=interested_var, save_id='',
                                    the_tau_val=self._coeff_term[3])

    def one_ae_lpc(self):
        with tf.Graph().as_default():
            x, x_, lr, the_share = self.init_placeholder_end_to_end()
            is_quan_on = tf.compat.v1.placeholder(dtype=tf.float32, shape=None, name='is_quan_on')
            lpc_x = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, self._lpc_order, 1), name='lpc_x')
            with tf.compat.v1.variable_scope('lpc_quan'):
                alpha = tf.Variable(init_alpha, dtype=tf.float32, name='alpha', trainable=self._is_cq)
                lpc_bins_len = len(lpc_coeff_lsf_bins)
                lpc_bins = tf.Variable(lpc_coeff_lsf_bins, dtype=tf.float32, name='bins', trainable=self._is_cq)

                soft_assignment_lpc, quan_lpc_coeff = scalar_softmax_quantization(lpc_x,
                                                                                  alpha,
                                                                                  lpc_bins,
                                                                                  is_quan_on,
                                                                                  the_share,
                                                                                  self._lpc_order,
                                                                                  lpc_bins_len)

                print(quan_lpc_coeff.shape, 'quan_lpc_coeff shape')  # [None, 16, 1]
                quan_lpc_coeff = quan_lpc_coeff[:, :, 0]
                quan_lpc_coeff = tf.reshape(quan_lpc_coeff, (-1, self._lpc_order))
                # quan_lpc_coeff = tf.reshape(tf.matmul(soft_assignment, tf.expand_dims(bins, 1)), (-1, self._lpc_order))  #

            quan_lpc_x_poly = tf.compat.v1.py_func(lsf2poly_after_quan, [quan_lpc_coeff, self._lpc_order], [tf.float32])[0]
            res_x = tf.compat.v1.py_func(lpc_analysis_get_residual, [x, quan_lpc_x_poly], [tf.float32])[0]
            res_x = tf.reshape(res_x, (-1, frame_length, 1))


            tau = tf.compat.v1.placeholder(dtype=tf.float32, shape=None, name='tau')
            the_stride = [2, 2] if self._the_strides[0] == 4 else [2]
            _softmax_assignment, weight, decoded_fully, encoded, decoded, alpha, bins \
                = self.computational_graph_end2end_quan_on_lpc(res_x * self._res_scalar,
                                                               quan_lpc_x_poly,
                                                               the_share,
                                                               is_quan_on,
                                                               self._num_bins_for_follower[0],
                                                               'scope_1',
                                                               the_stride)

            decoded = decoded / self._res_scalar  # decoded here is the estimated res
            synthesized = tf.compat.v1.py_func(lpc_synthesizer_tr, [quan_lpc_x_poly, decoded], [tf.float32])[0]

            # time_loss = mse_loss(synthesized, x_[:, :, 0])
            # time_loss = mse_loss_v1(decoded, res_x[:, :, 0])
            ent_list = tf.cast([16.0, 256.0], tf.float32) if self._the_strides[0] == 2 \
                else tf.cast([16.0, 128.0], tf.float32)

            time_loss = mse_loss(decoded, res_x[:, :, 0])
            freq_loss = mfcc_loss(decoded, res_x[:, :, 0])

            quantization_loss = quan_loss(soft_assignment_lpc) * ent_list[0]/tf.reduce_sum(ent_list) + \
                                quan_loss(_softmax_assignment) * ent_list[1]/tf.reduce_sum(ent_list)

            ent_loss_list = tf.cast([entropy_coding_loss(soft_assignment_lpc),
                             entropy_coding_loss(_softmax_assignment)], tf.float32)


            ent_loss = tf.reduce_mean(tf.reduce_sum(ent_list / tf.reduce_sum(ent_list) * ent_loss_list))


            loss_no_quan = self._coeff_term[0] * time_loss + self._coeff_term[1] * freq_loss
            loss_quan_init = self._coeff_term[0] * time_loss + \
                             self._coeff_term[1] * freq_loss + \
                             self._coeff_term[2] * quantization_loss + \
                             tau * ent_loss  # self._coeff_term[3] * ent_loss  #

            trainop2_no_quan = tf.compat.v1.train.AdamOptimizer(lr, beta1=0.9, beta2=0.999). \
                minimize(loss_no_quan, var_list=tf.compat.v1.trainable_variables())
            trainop2_quan_init = tf.compat.v1.train.AdamOptimizer(lr, beta1=0.9, beta2=0.999). \
                minimize(loss_quan_init, var_list=tf.compat.v1.trainable_variables())
            trainop2_list = [trainop2_no_quan, trainop2_quan_init]

            saver = tf.compat.v1.train.Saver()
            interested_var = [time_loss, freq_loss, quantization_loss, lpc_bins, _softmax_assignment, ent_loss, ent_loss_list, encoded]
            with tf.compat.v1.Session() as sess:
                sess.run(tf.compat.v1.global_variables_initializer())
                self.model_training_lpc(sess, x=x, x_=x_, lr=lr, the_share=the_share, lpc_x=lpc_x, synthesized=synthesized,
                                        res_x=res_x, tau=tau,
                                        is_quan_on=is_quan_on, encoded=encoded, loss1=time_loss, mfcc_loss=freq_loss,
                                        quan_loss=quan_loss, ent_loss=ent_loss, trainop2_list=trainop2_list, decoded=decoded,
                                        alpha=alpha,
                                        bins=bins, saver=saver,
                                        the_learning_rate=self._learning_rate_tanh, epoch=self._epoch_tanh,
                                        flag='pretrain', interested_var=interested_var, save_id='',
                                        the_tau_val=self._coeff_term[3])

    def _update_lpc_residual(self):
        tr_data = np.load(self._root_path + 'cmrl_v2/flat_raw_data.npy')[:int(self._tr_data_size), :]  # * self._lpc_res_scaler
        tr_lpc_coeff = np.load(self._root_path + 'cmrl_v2/end2end_lpc_coeff_in_lsf_16_signal_based_final_flat.npy')[:int(self._tr_data_size), :]
        x1 = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, self._lpc_order, 1), name='x1')
        x2 = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, frame_length, 1), name='x2')
        the_share = tf.compat.v1.placeholder(dtype=tf.bool, shape=None, name='the_share')
        # is_quan_on = tf.compat.v1.placeholder(dtype=tf.float32, shape=None, name='is_quan_on')
        with tf.compat.v1.variable_scope('lpc_quan'):
            alpha = tf.Variable(init_alpha, dtype=tf.float32, name='alpha')
            lpc_coeff_lsf_bins = np.load('lpc_coeff_lsf_bins_updated_' + self._rand_model_id + '.npy')
            lpc_bins_num = len(lpc_coeff_lsf_bins)
            print('lpc_bins_num len', lpc_bins_num)
            lpc_bins = tf.Variable(np.sort(lpc_coeff_lsf_bins), dtype=tf.float32, name='bins')

            soft_assignment_lpc, quan_lpc_coeff = scalar_softmax_quantization(x1,
                                                                              alpha,
                                                                              lpc_bins,
                                                                              1.0,
                                                                              the_share,
                                                                              self._lpc_order,
                                                                              lpc_bins_num)
            print(quan_lpc_coeff.shape, 'quan_lpc_coeff shape')  # [None, 16, 1]
            quan_lpc_coeff = quan_lpc_coeff[:, :, 0]
            quan_lpc_coeff = tf.reshape(quan_lpc_coeff, (-1, self._lpc_order))

        quan_lpc_x_poly = tf.compat.v1.py_func(lsf2poly_after_quan, [quan_lpc_coeff, self._lpc_order], [tf.float32])[0]
        res_x = tf.compat.v1.py_func(lpc_analysis_get_residual, [x2, quan_lpc_x_poly], [tf.float32])[0]
        res_x = tf.reshape(res_x, (-1, frame_length, 1))

        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            start, end = 0, 50000
            tr_lpc_res = np.empty([])
            while start < training_data_size:
                _res_x_1 = sess.run(res_x, feed_dict={
                    x1: np.expand_dims(self._tr_data[start:end, frame_length:frame_length + self._lpc_order], -1).
                                    astype(np.float32),
                    x2: np.expand_dims(self._tr_data[start:end, :frame_length], -1).
                                    astype(np.float32),
                    the_share: False})
                if start == 0:
                    tr_lpc_res = _res_x_1[:, :, 0]
                else:
                    tr_lpc_res = np.concatenate([tr_lpc_res, _res_x_1[:, :, 0]], 0)
                start += 50000
                end += 50000
                print(start)
            self._tr_data = np.concatenate([tr_data, tr_lpc_coeff, tr_lpc_res], 1)
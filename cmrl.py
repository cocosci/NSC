"""
File name: cmrl.py
Description: Implementation of cross module residual learning.
			 Current version supports 2 cascaded coding modules.
"""
from neural_speech_coding_module import *


class CMRL(neuralSpeechCodingModule):
    """
    Cross Module Residual Learning is termed as CMRL.
    CMRL is to enable a multi-stage quantization scheme among multiple neural autoencoders (AE).
    CMRL trains the first AE, and then the second AE to quantize what has not been reconstructed, the residual, by the
    first AE.
    """
    def __init__(self, arg):
        super(CMRL, self).__init__(arg)
        self._num_resnets = arg.num_resnets
        self._from_where_step = int(arg.from_where_step)
        self._learning_rate_greedy_followers = list(map(lambda x: float(x), arg.learning_rate_greedy_followers.split()))

    def _greedy_followers(self, num_res):
        # This function trains the num_res AE on top of the previous num_res-1 AE.
        with tf.Graph().as_default():
            x, x_, lr, the_share = self.init_placeholder_end_to_end()
            tau = tf.compat.v1.placeholder(dtype=tf.float32, shape=None, name='tau')
            is_quan_on = tf.compat.v1.placeholder(dtype=tf.float32, shape=None, name='is_quan_on')

            residual_coding_x = [None] * (num_res + 1)
            all_var_list = []

            the_stride = [2, 2] if self._the_strides[0] == 4 else [2]

            _softmax_assignment_1, weight, dist, encoded, residual_coding_x[0], \
            alpha, bins, soft_assignment_fully_1 = self.computational_graph_end2end_quan_on(
                x,
                the_share,
                is_quan_on,
                self._num_bins_for_follower[0],
                'scope_1',
                the_stride)
            # residual_coding_x[0] = residual_coding_x[0]
            all_var_list += tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
                                                        scope='scope_1')

            for i in range(1, num_res):
                _softmax_assignment_1, weight, dist, encoded, residual_coding_x[i], \
                alpha, bins, soft_assignment_fully_1 = self.computational_graph_end2end_quan_on(
                    (x - tf.expand_dims(tf.reduce_sum(input_tensor=residual_coding_x[:i], axis=0), axis=2))
                    * self._res_scalar,
                    #mu_law_mapping((x - tf.expand_dims(tf.reduce_sum(input_tensor=residual_coding_x[:i], axis=0),
                    #                                   axis=2)))* self._res_scalar,
                    the_share,
                    is_quan_on,
                    self._num_bins_for_follower[i],
                    'scope_' + str(i + 1),
                    the_stride)
                residual_coding_x[i] = residual_coding_x[i] / self._res_scalar
                # residual_coding_x[i] = inverse_mu_law_mapping(residual_coding_x[i]/ self._res_scalar)
                all_var_list += tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
                                                            scope='scope_' + str(i + 1))
            # print(all_var_list)
            # saver = tf.compat.v1.train.Saver()
            saver = tf.compat.v1.train.Saver(var_list=all_var_list)
            with tf.compat.v1.Session() as sess:
                if num_res == 1:
                    saver.restore(sess, "./check/model_bnn_ac_" + self._rand_model_id + '_' + ".ckpt")
                    print('model' + "./check/model_bnn_ac_" + self._rand_model_id + '_' + ".ckpt" + ' is restored!')
                else:
                    print("./check/model_bnn_ac_" + self._rand_model_id +
                                  '_follower_' + str(num_res - 1) + self._suffix + ".ckpt" + ' is restored!')
                    saver.restore(sess, "./check/model_bnn_ac_" + self._rand_model_id +
                                  '_follower_' + str(num_res - 1) + self._suffix + ".ckpt")

                _softmax_assignment, weight, dist, encoded, residual_coding_x[-1], \
                alpha, bins, soft_assignment_fully = self.computational_graph_end2end_quan_on(
                    self._res_scalar * (x - tf.expand_dims(tf.reduce_sum(input_tensor=
                                                                         residual_coding_x[:-1], axis=0), axis=2)),
                    the_share,
                    is_quan_on,
                    self._num_bins_for_follower[num_res - 1],
                    'scope_' + str(num_res + 1),
                    the_stride)
                residual_coding_x[-1] = residual_coding_x[-1] / self._res_scalar

                #all_var_list = []
                #for i in range(num_res + 1):
                #    all_var_list += tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
                #                                                scope='scope_' + str(i + 1))
                # print(all_var_list)
                all_var_list += tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
                                                            scope='scope_' + str(num_res + 1))
                saver = tf.compat.v1.train.Saver(var_list=all_var_list)
                decoded = np.sum(residual_coding_x, axis=0)
                time_loss = mse_loss(decoded, x_[:, :, 0])
                freq_loss = mfcc_loss(decoded, x_[:, :, 0])
                quantization_loss = quan_loss(_softmax_assignment)
                ent_loss = entropy_coding_loss(_softmax_assignment)
                interested_var = [time_loss, freq_loss, quantization_loss, alpha, ent_loss, ent_loss, ent_loss, encoded]
                loss2_no_quan = self._coeff_term[0] * time_loss + \
                                self._coeff_term[1] * freq_loss
                loss2_quan_init = self._coeff_term[0] * time_loss + \
                                  self._coeff_term[1] * freq_loss + \
                                  self._coeff_term[2] * quantization_loss +\
                                  tau * ent_loss  # + self._coeff_term[3] * ent_loss  #
                trainop2_no_quan = tf.compat.v1.train.AdamOptimizer(lr, beta2=0.999).\
                    minimize(loss2_no_quan,
                             var_list=tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
                                                                  scope='scope_' + str(num_res + 1)))
                trainop2_quan_init = tf.compat.v1.train.AdamOptimizer(lr, beta2=0.999).\
                    minimize(loss2_quan_init,
                             var_list=tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
                                                                  scope='scope_' + str(num_res + 1)))

                trainop2_list = [trainop2_no_quan, trainop2_quan_init]
                adam_vars = [var for var in tf.compat.v1.global_variables() if
                             'Adam' in var.name or 'beta1_power' in var.name or 'beta2_power' in var.name]
                sess.run(tf.compat.v1.variables_initializer(tf.compat.v1.get_collection(
                    tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='scope_' + str(num_res + 1)) + adam_vars))

                # print('trainable model parameters:',
                #       np.sum([np.prod(v.get_shape().as_list()) for v in
                #               tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
                #                                           scope='scope_' + str(num_res + 1))]))

                self.model_training(sess, x=x, x_=x_, lr=lr, the_share=the_share, tau=tau,
                                    is_quan_on=is_quan_on, encoded=encoded, loss1=time_loss, mfcc_loss=freq_loss,
                                    quan_loss=quan_loss, ent_loss=ent_loss, trainop2_list=trainop2_list,
                                    decoded=decoded, alpha=alpha,
                                    bins=bins, saver=saver,
                                    the_learning_rate=self._learning_rate_greedy_followers[-2],
                                    epoch=self._epoch_greedy_followers[-2],
                                    flag='the_follower', interested_var=interested_var,
                                    save_id='follower_' + str(num_res) + self._suffix,
                                    the_tau_val=self._coeff_term[3])

    def _greedy_followers_lpc(self, num_res):
        # This function trains the num_res AE on top of the previous num_res-1 AE.
        with tf.Graph().as_default():
            x, x_, lr, the_share = self.init_placeholder_end_to_end()
            tau = tf.compat.v1.placeholder(dtype=tf.float32, shape=None, name='tau')
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

            quan_lpc_x_poly = \
            tf.compat.v1.py_func(lsf2poly_after_quan, [quan_lpc_coeff, self._lpc_order], [tf.float32])[0]
            res_x = tf.compat.v1.py_func(lpc_analysis_get_residual, [x, quan_lpc_x_poly], [tf.float32])[0]
            res_x = tf.reshape(res_x, (-1, frame_length, 1))

            residual_coding_x = [None] * (num_res + 1)
            all_var_list = []

            the_stride = [2, 2] if self._the_strides[0] == 4 else [2]

            _softmax_assignment_1, weight, decoded_fully, encoded, residual_coding_x[0], alpha, bins = \
                self.computational_graph_end2end_quan_on_lpc(
                    res_x * self._res_scalar,
                    quan_lpc_x_poly,
                    the_share,
                    is_quan_on,
                    self._num_bins_for_follower[0],
                    'scope_1',
                    the_stride)
            residual_coding_x[0] = residual_coding_x[0] / self._res_scalar
            all_var_list += tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
                                                        scope='scope_1')
            all_var_list += tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='lpc_quan')


            for i in range(1, num_res):
                the_stride = [2, 2] if self._the_strides[i] == 4 else [2]
                print(the_stride)
                _softmax_assignment_1, weight, decoded_fully, encoded, residual_coding_x[i], alpha, bins = \
                    self.computational_graph_end2end_quan_on_lpc(
                        (res_x - tf.expand_dims(tf.reduce_sum(input_tensor=residual_coding_x[:i], axis=0), axis=2))
                        * self._res_scalar,
                        quan_lpc_x_poly,
                        the_share,
                        is_quan_on,
                        self._num_bins_for_follower[i],
                        'scope_' + str(i + 1),
                        the_stride)
                residual_coding_x[i] = residual_coding_x[i] / self._res_scalar
                all_var_list += tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
                                                            scope='scope_' + str(i + 1))


            saver = tf.compat.v1.train.Saver(var_list=all_var_list)
            with tf.compat.v1.Session() as sess:
                if num_res == 1:
                    saver.restore(sess, "./check/model_bnn_ac_" + self._rand_model_id + '_' + ".ckpt")
                    print('model' + "./check/model_bnn_ac_" + self._rand_model_id + '_' + ".ckpt" + ' is restored!')
                else:
                    print("./check/model_bnn_ac_" + self._rand_model_id +
                                  '_follower_' + str(num_res - 1) + self._suffix + ".ckpt" + ' is restored!')
                    saver.restore(sess, "./check/model_bnn_ac_" + self._rand_model_id +
                                  '_follower_' + str(num_res - 1) + self._suffix + ".ckpt")
                the_stride = [2, 2] if self._the_strides[-1] == 4 else [2]
                _softmax_assignment, weight, decoded_fully, encoded, residual_coding_x[-1], alpha, bins = \
                    self.computational_graph_end2end_quan_on_lpc(
                        self._res_scalar * (res_x - tf.expand_dims(
                            tf.reduce_sum(input_tensor=residual_coding_x[:-1], axis=0), axis=2)),
                        quan_lpc_x_poly,
                        the_share,
                        is_quan_on,
                        self._num_bins_for_follower[num_res - 1],
                        'scope_' + str(num_res + 1),
                        the_stride)
                residual_coding_x[-1] = residual_coding_x[-1] / self._res_scalar

                all_var_list += tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
                                                            scope='scope_' + str(num_res + 1))

                print('trainable model parameters:',
                      np.sum([np.prod(v.get_shape().as_list()) for v in tf.compat.v1.trainable_variables()]))
                print('trainable model parameters:',
                      np.sum([np.prod(v.get_shape().as_list()) for v in tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
                                                                  scope='scope_' + str(num_res + 1))]))
                saver = tf.compat.v1.train.Saver(var_list=all_var_list)

                decoded = np.sum(residual_coding_x, axis=0)

                synthesized = \
                    tf.compat.v1.py_func(lpc_synthesizer_tr, [quan_lpc_x_poly, decoded],
                                         [tf.float32])[0]

                time_loss = mse_loss(decoded, res_x[:, :, 0])
                freq_loss = mfcc_loss(decoded, res_x[:, :, 0])
                quantization_loss = quan_loss(_softmax_assignment)
                ent_loss = entropy_coding_loss(_softmax_assignment)
                interested_var = [time_loss, freq_loss, quantization_loss, lpc_bins, _softmax_assignment, ent_loss,
                                  ent_loss, encoded]
                loss2_no_quan = self._coeff_term[0] * time_loss + \
                                self._coeff_term[1] * freq_loss
                loss2_quan_init = self._coeff_term[0] * time_loss + \
                                  self._coeff_term[1] * freq_loss + \
                                  self._coeff_term[2] * quantization_loss +\
                                  tau * ent_loss  # + self._coeff_term[3] * ent_loss  #
                trainop2_no_quan = tf.compat.v1.train.AdamOptimizer(lr, beta2=0.999).\
                    minimize(loss2_no_quan,
                             var_list=tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
                                                                  scope='scope_' + str(num_res + 1)))
                trainop2_quan_init = tf.compat.v1.train.AdamOptimizer(lr, beta2=0.999).\
                    minimize(loss2_quan_init,
                             var_list=tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
                                                                  scope='scope_' + str(num_res + 1)))

                # trainop2_no_quan = tf.compat.v1.train.AdamOptimizer(lr, beta2=0.999). \
                #     minimize(loss2_no_quan,
                #              var_list=tf.compat.v1.trainable_variables())
                # trainop2_quan_init = tf.compat.v1.train.AdamOptimizer(lr, beta2=0.999). \
                #     minimize(loss2_quan_init,
                #              var_list=tf.compat.v1.trainable_variables())

                trainop2_list = [trainop2_no_quan, trainop2_quan_init]
                adam_vars = [var for var in tf.compat.v1.global_variables() if
                             'Adam' in var.name or 'beta1_power' in var.name or 'beta2_power' in var.name]
                #sess.run(tf.compat.v1.variables_initializer(
                #    tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='lpc_quan')))
                sess.run(tf.compat.v1.variables_initializer(
                    tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
                                                scope='scope_' + str(num_res + 1)) + adam_vars))

                self.model_training_lpc(sess, x=x, x_=x_, lr=lr, the_share=the_share, lpc_x=lpc_x,
                                        synthesized=synthesized,
                                        res_x=res_x, tau=tau,
                                        is_quan_on=is_quan_on, encoded=encoded, loss1=time_loss, mfcc_loss=freq_loss,
                                        quan_loss=quan_loss, ent_loss=ent_loss, trainop2_list=trainop2_list,
                                        decoded=decoded,
                                        alpha=alpha,
                                        bins=bins, saver=saver,
                                        the_learning_rate=self._learning_rate_greedy_followers[-2],
                                        epoch=self._epoch_greedy_followers[-2],
                                        flag='the_follower', interested_var=interested_var,
                                        save_id='follower_' + str(num_res) + self._suffix,
                                        the_tau_val= self._coeff_term[3])
                                        # the_tau_val= -0.25)
                                        # the_tau_val=self._coeff_term[3])

    def _finetuning(self, num_res):
        # This function trains the num_res AE on top of the previous num_res-1 AE.
        with tf.Graph().as_default():
            x, x_, lr, the_share = self.init_placeholder_end_to_end()
            # tau = tf.compat.v1.placeholder(dtype=tf.float32, shape=None, name='tau')
            tau = tf.compat.v1.placeholder(dtype=tf.float32, shape=[1, 2], name='tau')
            # tau_1 = tf.compat.v1.placeholder(dtype=tf.float32, shape=None, name='tau_1')
            # tau_2 = tf.compat.v1.placeholder(dtype=tf.float32, shape=None, name='tau_2')
            is_quan_on = tf.compat.v1.placeholder(dtype=tf.float32, shape=None, name='is_quan_on')
            residual_coding_x = [None] * num_res
            all_var_list = []
            _softmax_assignment = [None] * num_res
            _softmax_assignment[0], weight, dist, encoded, residual_coding_x[0], \
            alpha, bins, soft_assignment_fully_1 = self.computational_graph_end2end_quan_on(
                x,
                the_share,
                is_quan_on,
                self._num_bins_for_follower[0],
                'scope_1',
                self._the_strides)

            for i in range(1, num_res):
                _softmax_assignment[i], weight, dist, encoded, residual_coding_x[i], \
                alpha, bins, soft_assignment_fully_1 = self.computational_graph_end2end_quan_on(
                    self._res_scalar * (x - tf.expand_dims(tf.reduce_sum(input_tensor=
                                                                         residual_coding_x[:i], axis=0), axis=2)),
                    the_share,
                    is_quan_on,
                    self._num_bins_for_follower[i],
                    'scope_' + str(i + 1),
                    self._the_strides)
                residual_coding_x[i] = residual_coding_x[i] / self._res_scalar

            decoded = np.sum(residual_coding_x, axis=0)

            saver = tf.compat.v1.train.Saver()
            with tf.compat.v1.Session() as sess:
                if num_res == 1:
                    saver.restore(sess, "./check/model_bnn_ac_" + self._rand_model_id + '_' + ".ckpt")
                    print('model' + "./check/model_bnn_ac_" + self._rand_model_id + '_' + ".ckpt" + ' is restored!')
                else:
                    if self._from_where_step == 2:
                        print("./check/model_bnn_ac_" + self._rand_model_id +
                                      '_follower_' + str(num_res - 1) + 'end2endcascade' + ".ckpt" + ' is restored!')
                        saver.restore(sess, "./check/model_bnn_ac_" + self._rand_model_id +
                                      '_follower_' + str(num_res - 1) + 'end2endcascade' + ".ckpt")
                    elif self._from_where_step == 3:
                        print("./check/model_bnn_ac_" + self._rand_model_id +
                                      '_finetune_' + str(num_res) + self._suffix + ".ckpt" + ' is restored!')
                        saver.restore(sess, "./check/model_bnn_ac_" + self._rand_model_id +
                                      '_finetune_' + str(num_res) + self._suffix + ".ckpt")
                    else:
                        print("Wrong _from_where_step setup.")
                time_loss = mse_loss(decoded, x_[:, :, 0])
                freq_loss = mfcc_loss(decoded, x_[:, :, 0])
                quantization_loss_arr = [0] * num_res
                ent_loss_arr = [0] * num_res
                for i in range(num_res):
                    quantization_loss_arr[i] = quan_loss(_softmax_assignment[i])
                    ent_loss_arr[i] = entropy_coding_loss(_softmax_assignment[i])
                quantization_loss = tf.reduce_sum(quantization_loss_arr)
                # quantization_loss = 0.5 * quantization_loss_arr[0] + 0.5 * quantization_loss_arr[1]
                ent_loss = tf.reduce_sum(ent_loss_arr)

                interested_var = [time_loss, freq_loss, quantization_loss, alpha, ent_loss_arr, ent_loss, ent_loss, encoded]
                loss2_no_quan = self._coeff_term[0] * time_loss + \
                                self._coeff_term[1] * freq_loss
                loss2_quan_init = self._coeff_term[0] * time_loss + \
                                  self._coeff_term[1] * freq_loss + \
                                  self._coeff_term[2] * quantization_loss + \
                                  tau[0, 0] * ent_loss_arr[0] + tau[0, 1] * ent_loss_arr[1]
                                  # tau * ent_loss  ##self._coeff_term[3] * ent_loss  #
                # self._target_entropy *= 2

                trainop2_no_quan = tf.compat.v1.train.AdamOptimizer(lr, beta2=0.999).\
                    minimize(loss2_no_quan, var_list=tf.compat.v1.trainable_variables())
                trainop2_quan_init = tf.compat.v1.train.AdamOptimizer(lr, beta2=0.999).\
                    minimize(loss2_quan_init, var_list=tf.compat.v1.trainable_variables())
                trainop2_list = [trainop2_no_quan, trainop2_quan_init]

                adam_vars = [var for var in tf.compat.v1.global_variables() if
                             'Adam' in var.name or 'beta1_power' in var.name or 'beta2_power' in var.name]
                # print(adam_vars)
                sess.run(tf.compat.v1.variables_initializer(adam_vars))
                print('ALL trainable model parameters:',
                      np.sum([np.prod(v.get_shape().as_list()) for v in tf.compat.v1.trainable_variables()]))
                self.model_training(sess, x=x, x_=x_, lr=lr, the_share=the_share, tau=tau,
                                    is_quan_on=is_quan_on, encoded=encoded, loss1=time_loss, mfcc_loss=freq_loss,
                                    quan_loss=quan_loss, ent_loss=ent_loss, trainop2_list=trainop2_list,
                                    decoded=decoded, alpha=alpha,
                                    bins=bins, saver=saver,
                                    the_learning_rate=self._learning_rate_greedy_followers[-1],
                                    epoch=self._epoch_greedy_followers[-1],
                                    flag='finetune', interested_var=interested_var,
                                    save_id='finetune_' + str(num_res) + self._suffix + self._save_unique_mark,
                                    the_tau_val=self._coeff_term[3])

    def _finetuning_lpc(self, num_res):
        # This function trains the num_res AE on top of the previous num_res-1 AE.
        with tf.Graph().as_default():
            x, x_, lr, the_share = self.init_placeholder_end_to_end()
            is_quan_on = tf.compat.v1.placeholder(dtype=tf.float32, shape=None, name='is_quan_on')
            lpc_x = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, self._lpc_order, 1), name='lpc_x')
            with tf.compat.v1.variable_scope('lpc_quan'):
                alpha = tf.Variable(init_alpha, dtype=tf.float32, name='alpha')
                lpc_bins_len = len(lpc_coeff_lsf_bins)
                lpc_bins = tf.Variable(lpc_coeff_lsf_bins, dtype=tf.float32, name='bins')
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
            residual_coding_x = [None] * (num_res)
            soft_assignment_mat = [None] * (num_res)


            for i in range(0, num_res):
                the_stride = [2, 2] if self._the_strides[i] == 4 else [2]

                cmrl_input = self._res_scalar * res_x if i == 0 else self._res_scalar * (res_x - tf.expand_dims(
                    tf.reduce_sum(input_tensor=residual_coding_x[:i], axis=0), axis=2))
                soft_assignment_mat[i], weight, decoded_fully, encoded, residual_coding_x[i], alpha, bins = \
                    self.computational_graph_end2end_quan_on_lpc(
                        cmrl_input,
                        quan_lpc_x_poly,
                        the_share,
                        is_quan_on,
                        self._num_bins_for_follower[i],
                        'scope_' + str(i + 1),
                        the_stride)
                residual_coding_x[i] = residual_coding_x[i] / self._res_scalar

            # _softmax_assignment, weight, decoded_fully, encoded, decoded, alpha, bins \
            #     = self.computational_graph_end2end_quan_on_lpc_vq(res_x,
            #                                                       quan_lpc_x_poly,
            #                                                       the_share,
            #                                                       is_quan_on,
            #                                                       self._num_bins_for_follower[0],
            #                                                       'scope_1',
            #                                                       self._the_strides)


            decoded = np.sum(residual_coding_x, axis=0)  # decoded here is the estimated res
            synthesized = tf.compat.v1.py_func(lpc_synthesizer_tr, [quan_lpc_x_poly, decoded], [tf.float32])[0]

            saver = tf.compat.v1.train.Saver()
            with tf.compat.v1.Session() as sess:
                if num_res == 1:
                    saver.restore(sess, "./check/model_bnn_ac_" + self._rand_model_id + '_' + ".ckpt")
                    print('model' + "./check/model_bnn_ac_" + self._rand_model_id + '_' + ".ckpt" + ' is restored!')
                else:
                    print("./check/model_bnn_ac_" + self._rand_model_id +
                                  '_follower_' + str(num_res - 1) + 'end2endcascade' + ".ckpt" + ' is restored!')
                    saver.restore(sess, "./check/model_bnn_ac_" + self._rand_model_id +
                                  '_follower_' + str(num_res - 1) + 'end2endcascade' + ".ckpt")

                time_loss = mse_loss(decoded, res_x[:, :, 0])
                freq_loss = mfcc_loss(decoded, res_x[:, :, 0])
                quantization_loss = quan_loss(soft_assignment_lpc) +\
                                    quan_loss(soft_assignment_mat[0]) + \
                                    quan_loss(soft_assignment_mat[1])

                summed_feature_len = [self._lpc_order]
                for i in range(0, num_res):
                    summed_feature_len.append(128.0 if self._the_strides[i] == 4 else 256.0)

                ent_loss = (summed_feature_len[0] / np.sum(summed_feature_len)) * entropy_coding_loss(soft_assignment_lpc) + \
                           (summed_feature_len[1] / np.sum(summed_feature_len)) * entropy_coding_loss(soft_assignment_mat[0]) + \
                           (summed_feature_len[2] / np.sum(summed_feature_len)) * entropy_coding_loss(soft_assignment_mat[1])

                ent_loss_list = [entropy_coding_loss(soft_assignment_lpc),
                                 entropy_coding_loss(soft_assignment_mat[0]),
                                 entropy_coding_loss(soft_assignment_mat[1])]

                loss_no_quan = self._coeff_term[0] * time_loss + self._coeff_term[1] * freq_loss
                loss_quan_init = self._coeff_term[0] * time_loss + \
                                 self._coeff_term[1] * freq_loss + \
                                 self._coeff_term[2] * quantization_loss # + \tau * ent_loss  # self._coeff_term[3] * ent_loss  #

                trainop2_no_quan = tf.compat.v1.train.AdamOptimizer(lr, beta1=0.9, beta2=0.999). \
                    minimize(loss_no_quan, var_list=tf.compat.v1.trainable_variables())
                trainop2_quan_init = tf.compat.v1.train.AdamOptimizer(lr, beta1=0.9, beta2=0.999). \
                    minimize(loss_quan_init, var_list=tf.compat.v1.trainable_variables())
                trainop2_list = [trainop2_no_quan, trainop2_quan_init]
                interested_var = [time_loss, freq_loss, quantization_loss, lpc_bins, soft_assignment_mat, ent_loss,
                                  ent_loss_list, encoded]
                adam_vars = [var for var in tf.compat.v1.global_variables() if
                             'Adam' in var.name or 'beta1_power' in var.name or 'beta2_power' in var.name]
                sess.run(tf.compat.v1.variables_initializer(adam_vars))
                print('trainable model parameters:',
                      np.sum([np.prod(v.get_shape().as_list()) for v in tf.compat.v1.trainable_variables()]))
                self.model_training_lpc(sess, x=x, x_=x_, lr=lr, the_share=the_share, lpc_x=lpc_x,
                                        synthesized=synthesized,
                                        res_x=res_x, tau=tau,
                                        is_quan_on=is_quan_on, encoded=encoded, loss1=time_loss, mfcc_loss=freq_loss,
                                        quan_loss=quan_loss, ent_loss=ent_loss, trainop2_list=trainop2_list,
                                        decoded=decoded,
                                        alpha=alpha,
                                        bins=bins, saver=saver,
                                        the_learning_rate=self._learning_rate_greedy_followers[-1],
                                        epoch=self._epoch_greedy_followers[-1],
                                        flag='finetune', interested_var=interested_var,
                                        save_id='finetune_' + str(num_res) + self._suffix,
                                        the_tau_val=self._coeff_term[3])

    def all_modules_feedforward(self, num_res):
        x, x_, lr, the_share = self.init_placeholder_end_to_end()
        tau = tf.compat.v1.placeholder(dtype=tf.float32, shape=None, name='tau')
        is_quan_on = tf.compat.v1.placeholder(dtype=tf.float32, shape=None, name='is_quan_on')
        residual_coding_x = [None] * (num_res)
        all_var_list = []
        encoded = [None] * (num_res)
        bins = [None] * (num_res)
        _softmax_assignment = [None] * (num_res)
        _softmax_assignment[0], weight, dist, encoded[0], residual_coding_x[0], \
        alpha, bins[0], soft_assignment_fully_1 = self.computational_graph_end2end_quan_on(
            x,
            the_share,
            is_quan_on,
            self._num_bins_for_follower[0],
            'scope_1',
            self._the_strides)

        for i in range(1, num_res):
            _softmax_assignment[i], weight, dist, encoded[i], residual_coding_x[i], \
            alpha, bins[i], soft_assignment_fully_1 = self.computational_graph_end2end_quan_on(
                self._res_scalar * (x - tf.expand_dims(tf.reduce_sum(input_tensor=
                                                                     residual_coding_x[:i], axis=0), axis=2)),
                the_share,
                is_quan_on,
                self._num_bins_for_follower[i],
                'scope_' + str(i + 1),
                self._the_strides)
            residual_coding_x[i] = residual_coding_x[i] / self._res_scalar
        return x, x_, lr, tau, the_share, is_quan_on, _softmax_assignment, \
               encoded, soft_assignment_fully_1, residual_coding_x, alpha, bins

    def cmrl_eval(self, sess, x, x_, lr, the_share, is_quan_on, encoded, decoded, alpha,
                  bins, how_many, the_epoch, interested_var=None):
        """
        Conduct the feedforward to evaluate audio clips in the test dataset.
        """
        print(self._sep_test)
        num_of_test_files = len(self._sep_test)
        min_len, snr_list, si_snr_list = \
            np.array([0] * num_of_test_files), np.array([0.0] * num_of_test_files), np.array([0.0] * num_of_test_files)
        the_stoi, the_pesqs = np.array([0.0] * num_of_test_files), np.array([0.0] * num_of_test_files)
        the_linearitys = np.array([0.0] * num_of_test_files)

        if os.path.exists('./end2end_performance/' + str(self._base_model_id)):
            pass
        else:
            os.mkdir('./end2end_performance/' + str(self._base_model_id))

        for i in range(num_of_test_files):
            print(self._sep_test[i])
            per_sig, the_std = self._load_sig(self._sep_test[i])
            # print(the_std)
            segments_per_utterance = utterance_to_segment(per_sig, True)

            _decoded_sig = np.array(
                [0.0] * (frame_length + (frame_length - overlap_each_side) * (segments_per_utterance.shape[0] - 1)))
            # code_segment_len = code_len_val
            code_segment_len_1 = 128 if self._the_strides[0] == 4 else 256  # int(frame_length / self._the_strides[0])
            code_segment_len_2 = code_segment_len_1
            # code_segment_len_2 = 128 if self._the_strides[1] == 4 else 256  # int(frame_length / self._the_strides[0])
            _encoded_sig_1 = np.array(
                [0.0] * (code_segment_len_1 + (code_segment_len_1) * (segments_per_utterance.shape[0] - 1))).astype(
                np.float32)
            _encoded_sig_2 = np.array(
                [0.0] * (code_segment_len_2 + (code_segment_len_2) * (segments_per_utterance.shape[0] - 1))).astype(
                np.float32)

            all_entropy = np.array([0.0] * segments_per_utterance.shape[0])
            each_entropy = np.array([])

            encoding_start = time.time()
            for j in range(segments_per_utterance.shape[0]):
                feed_x = np.reshape(segments_per_utterance[j], (1, frame_length, 1))
                _interested_var, _decoded = sess.run(
                    [interested_var, decoded], feed_dict=
                    {x: feed_x,
                     x_: feed_x,
                     lr: 0.0,
                     the_share: 1.0,
                     is_quan_on: 1.0})  # share is 0 means it's soft, 1 means it's hard

                _decoded_sig[j * (frame_length - overlap_each_side): j * (
                frame_length - overlap_each_side) + 512] += hann_process(_decoded.flatten(), j,
                                                                         segments_per_utterance.shape[0])
                #_encoded_sig_1[j * (code_segment_len): j * (code_segment_len) + code_segment_len] = _encoded_1[0, :]
                #_encoded_sig_2[j * (code_segment_len): j * (code_segment_len) + code_segment_len] = _encoded_2[0, :]
                all_entropy[j] = _interested_var[-1]
                #ent_list = np.array([16.0, 256]) if len(np.array(_interested_var[6])) == 2 else np.array(
                #    [16.0, 256, 256])
                # print(_interested_var[6], np.sum(ent_list/np.sum(ent_list) * _interested_var[6]))
                # each_entropy.append(np.sum(ent_list/np.sum(ent_list) * _interested_var[6]))
                # each_entropy.append(np.array(_interested_var[6]))
                #if len(_interested_var[6]) == 3:
                #    each_entropy = np.append(each_entropy, np.array(_interested_var[6]))
            decoding_end = time.time()
            exec_time = decoding_end - encoding_start
            sig_duration = len(per_sig)/16000.0
            print('Execute time for the neural codec:', exec_time, sig_duration, exec_time / sig_duration)

            # average_entropy_each_mod = np.mean(each_entropy, axis=0)
            # print(average_entropy_each_mod)
            # print(np.max(per_sig), np.max(_decoded_sig))
            per_sig *= the_std
            _decoded_sig *= the_std
            # print(np.max(per_sig), np.max(_decoded_sig))

            min_len[i], si_snr_list[i], snr_list[i], the_stoi[i], the_pesqs[i], the_linearitys[i] = \
                eval_metrics(per_sig, _decoded_sig, self._rand_model_id)

            print('Test Utterance %1d: SNR: %7.5f dB  PESQ-WB: %6.5f  Entropy: %6.5f  Bit rate: %6.5f  ID: %s' % (
            i, snr_list[i], the_pesqs[i], np.mean(all_entropy),
            entropy_to_bitrate(np.mean(all_entropy), self._the_strides[0]), self._rand_model_id))
            self._write_to_file_and_update_to_display(
                'Test Utterance %1d: SNR: %7.5f dB  PESQ-WB: %6.5f   Entropy: %6.5f \n' % (
                i, snr_list[i], the_pesqs[i], np.mean(all_entropy)))

            # np.save('./end2end_performance/' + str(self._base_model_id) + '/ori_sig' + '_' + str(the_epoch) +
            #         '_' + str(i) + '.npy', per_sig)
            # np.save('./end2end_performance/' + str(self._base_model_id) + '/' +
            #         self._sep_test[i].split('/')[-1][:-4] + '.npy', _decoded_sig)
            sf.write('./end2end_performance/' + str(self._base_model_id) + '/' +
                     self._sep_test[i].split('/')[-1][:-4] + '.wav', _decoded_sig, sample_rate, 'PCM_16')
            # np.save('./end2end_performance/' + str(self._base_model_id) + '/enc_sig' + '_' +
            #         '_' + str(i) + '_1.npy', _encoded_sig_1)
            # np.save('./end2end_performance/' + str(self._base_model_id) + '/enc_sig' + '_' +
            #         '_' + str(i) + '_2.npy', _encoded_sig_2)
        print('Average SNR: %7.5f dB ' %(np.sum(min_len * snr_list / np.sum(min_len))))
        print('Average PESQ: %7.5f ' % (np.sum(min_len * the_pesqs / np.sum(min_len))))
        np.save('./end2end_performance/min_len_' + self._base_model_id + '.npy', min_len)
        np.save('./end2end_performance/snr_' + self._base_model_id + '.npy', snr_list)
        np.save('./end2end_performance/pesq_' + self._base_model_id + '.npy', the_pesqs)

    def cmrl_eval_lpc(self, sess, x, x_, lr, lpc_x, synthesized, tau, the_share, is_quan_on, loss1, encoded_1,
                      encoded_2, decoded, alpha, lpc_bins, bins_1, bins_2, how_many, the_epoch, interested_var=None):
        num_of_test_files = 1  # len(self._sep_test)
        min_len, snr_list = np.array([0] * num_of_test_files), np.array([0.0] * num_of_test_files)
        the_stoi, the_pesqs = np.array([0.0] * num_of_test_files), np.array([0.0] * num_of_test_files)
        _quan_loss_arr = np.array([0.0] * num_of_test_files)
        the_linearitys = np.array([0.0] * num_of_test_files)
        all_loss = [0.0] * num_of_test_files
        _alpha, _bins = 0, 0

        if os.path.exists('./end2end_performance/' + str(self._base_model_id)):
            pass
        else:
            os.mkdir('./end2end_performance/' + str(self._base_model_id))

        _lpc_bins = np.array([0.0] * 256)
        _bins_1 = np.array([0.0] * 64)
        _bins_2 = np.array([0.0] * 64)


        for i in range(num_of_test_files):
            # print(self._sep_test[i])
            per_sig, the_std = self._load_sig_lpc(self._sep_test[i])
            # segments_per_utterance = self.utterance_to_segment(per_sig, True)

            per_sig_highpass_empha_filtered = np.array(list(empha_filter(highpass_filter(per_sig))))
            segments_per_utterance = utterance_to_segment(per_sig_highpass_empha_filtered, True)
            print(segments_per_utterance.shape)
            _decoded_sig = np.array(
                [0.0] * (frame_length + (frame_length - overlap_each_side) * (segments_per_utterance.shape[0] - 2)))
            code_segment_len_1 = 128 if self._the_strides[0] == 4 else 256  # int(frame_length / self._the_strides[0])
            # code_segment_len_2 = 128 if self._the_strides[1] == 4 else 256  # int(frame_length / self._the_strides[0])
            code_segment_len_2 = code_segment_len_1

            _synthesized_sig = np.array(
                [0.0] * (frame_length + (frame_length - overlap_each_side) * (segments_per_utterance.shape[0] - 2)))
            _encoded_sig_1 = np.array(
                [0.0] * (code_segment_len_1 + (code_segment_len_1) * (segments_per_utterance.shape[0] - 2))).astype(
                np.float32)
            _encoded_sig_2 = np.array(
                [0.0] * (code_segment_len_2 + (code_segment_len_2) * (segments_per_utterance.shape[0] - 2))).astype(
                np.float32)

            all_entropy = np.array([0.0] * (segments_per_utterance.shape[0] - 2))
            lpc_entropy = np.array([0.0] * (segments_per_utterance.shape[0] - 2))

            # segments_per_utterance_coeff, segments_per_utterance_res = lpc_analysis_at_test(segments_per_utterance, self._lpc_order)
            segments_per_utterance_coeff = lpc_analysis_at_test(segments_per_utterance, self._lpc_order)
            # segments_per_utterance_res = segments_per_utterance_res * self._lpc_res_scaler
            segments_per_utterance = utterance_to_segment(per_sig_highpass_empha_filtered[256:], True)

            each_entropy = np.array([])
            for j in range(segments_per_utterance.shape[0] - 2):
                feed_x = np.reshape(segments_per_utterance[j], (1, frame_length, 1))
                feed_lpc_coeff = np.reshape(segments_per_utterance_coeff[j], (1, self._lpc_order, 1))
                _interested_var, the_loss, _encoded_1, _encoded_2, _decoded, _synthesized = sess.run(
                    [interested_var, loss1, encoded_1, encoded_2, decoded, synthesized], feed_dict=
                    {x: feed_x,
                     x_: feed_x,
                     lpc_x: feed_lpc_coeff,
                     lr: 0.0,
                     the_share: True,
                     is_quan_on: 1.0})  # share is 0 means it's soft, 1 means it's hard

                _decoded_sig[j * (frame_length - overlap_each_side): j * (frame_length - overlap_each_side) +
                                                                     frame_length] += \
                    hann_process(_decoded.flatten(), j, segments_per_utterance.shape[0])

                _synthesized_sig[j * (frame_length - overlap_each_side): j * (frame_length - overlap_each_side) +
                                                                         frame_length] += \
                    hann_process(_synthesized.flatten(), j, segments_per_utterance.shape[0])

                _encoded_sig_1[j * (code_segment_len_1): j * (code_segment_len_1) + code_segment_len_1] += _encoded_1  # [0,:]
                _encoded_sig_2[j * (code_segment_len_2): j * (code_segment_len_2) + code_segment_len_2] += _encoded_2  # [0,:]
                all_entropy[j] = _interested_var[4]
                lpc_entropy[j] = _interested_var[3]

                ent_list = np.array([16.0, 256]) if len(np.array(_interested_var[6])) == 2 else np.array(
                    [16.0, 256, 256])
                # print(_interested_var[6], np.sum(ent_list/np.sum(ent_list) * _interested_var[6]))
                # each_entropy.append(np.sum(ent_list/np.sum(ent_list) * _interested_var[6]))
                # each_entropy.append(np.array(_interested_var[6]))
                if len(_interested_var[6]) == 3 or len(_interested_var[6]) == 2:
                    each_entropy = np.append(each_entropy, np.array(_interested_var[6]))
            print(each_entropy)
            average_entropy_each_mod = np.mean(each_entropy, axis=0)
            print(average_entropy_each_mod)
            # print(_bins)
            per_sig *= the_std
            _synthesized_sig = np.array(list((1 / empha_filter)(_synthesized_sig)))
            _synthesized_sig *= the_std
            _decoded_sig *= the_std

            min_len[i], _, snr_list[i], the_stoi[i], the_pesqs[i], the_linearitys[i] = eval_metrics(per_sig[256:], _synthesized_sig, self._rand_model_id)
            print(np.mean(lpc_entropy), np.mean(lpc_entropy)*(16000 / 480.0 * 16 / 1024))
            print(self._sep_test[i],
                  'Test Utterance %1d: SNR: %7.5f dB  PESQ-WB: %6.5f  Entropy: %6.5f  Bit rate: %6.5f  ID: %s' % (
                  i, snr_list[i], the_pesqs[i], np.mean(all_entropy),
                  entropy_to_bitrate(np.mean(all_entropy), self._the_strides[0]), self._base_model_id))
            self._write_to_file_and_update_to_display(
                'Test Utterance %1d: SNR: %7.5f dB  PESQ-WB: %6.5f   Entropy: %6.5f \n' % (
                i, snr_list[i], the_pesqs[i], np.mean(all_entropy)))
            sf.write('./end2end_performance/' + str(self._base_model_id) +
                     '/' + self._sep_test[i].split('/')[-1][:-4] + '.wav', _synthesized_sig, sample_rate, 'PCM_16')
            if VERBOSE:
                np.save('./end2end_performance/' + str(self._base_model_id) + '/' +
                        self._sep_test[i].split('/')[-1][:-4] + '_ori.npy', per_sig[256:])
                np.save('./end2end_performance/' + str(self._base_model_id) + '/' +
                        self._sep_test[i].split('/')[-1][:-4] + '_quantized.npy', _encoded_sig_1)
                np.save('./end2end_performance/' + str(self._base_model_id) + '/' +
                        self._sep_test[i].split('/')[-1][:-4] + '_res.npy', _decoded_sig)
                np.save('./end2end_performance/' + str(self._base_model_id) + '/' +
                        self._sep_test[i].split('/')[-1][:-4] + '_syn.npy', _synthesized_sig)

        sdr_return_it = np.sum(min_len * snr_list / np.sum(min_len))
        stoi_return_it = np.sum(min_len * the_stoi / np.sum(min_len))
        pesq_return_it = np.sum(min_len * the_pesqs / np.sum(min_len))
        linearity_return_it = np.sum(min_len * the_linearitys / np.sum(min_len))
        print('Average SNR: %7.5f dB ' % (np.sum(min_len * snr_list / np.sum(min_len))))
        print('Average PESQ: %7.5f ' % (np.sum(min_len * the_pesqs / np.sum(min_len))))
        np.save('./end2end_performance/min_len_' + self._base_model_id + '.npy', min_len)
        np.save('./end2end_performance/snr_' + self._base_model_id + '.npy', snr_list)
        np.save('./end2end_performance/pesq_' + self._base_model_id + '.npy', the_pesqs)

    def all_modules_feedforward_lpc(self, num_res):
        # tf.compat.v1.reset_default_graph()
        # with tf.Graph().as_default():
        x, x_, lr, the_share = self.init_placeholder_end_to_end()
        lpc_x = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, self._lpc_order, 1), name='lpc_x')
        is_quan_on = tf.compat.v1.placeholder(dtype=tf.float32, shape=None, name='is_quan_on')
        print(x.shape, '----')

        with tf.compat.v1.variable_scope('lpc_quan'):
            alpha = tf.Variable(init_alpha, dtype=tf.float32, name='alpha')
            lpc_bins_len = len(lpc_coeff_lsf_bins)
            lpc_bins = tf.Variable(lpc_coeff_lsf_bins, dtype=tf.float32, name='bins')
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

        quan_lpc_x_poly = tf.compat.v1.py_func(lsf2poly_after_quan, [quan_lpc_coeff, self._lpc_order], [tf.float32])[0]

        # res_x = tf.compat.v1.py_func(lpc_analysis_get_residual, [x, quan_lpc_x], [tf.float32])[0]
        res_x = tf.compat.v1.py_func(lpc_analysis_get_residual, [x, quan_lpc_x_poly], [tf.float32])[0]
        res_x = tf.reshape(res_x, (-1, frame_length, 1))

        tau = tf.compat.v1.placeholder(dtype=tf.float32, shape=None, name='tau')

        residual_coding_x = [None] * self._num_resnets
        _softmax_assignment = [None] * self._num_resnets
        # _softmax_assignment_2, encoded_2, bins_2, soft_assignment_fully_2 = 0,0,0,0

        the_stride = [2, 2] if self._the_strides[0] == 4 else [2]
        for i in range(self._num_resnets):

            if i == 0:
                _softmax_assignment[i], weight, decoded_fully, encoded, residual_coding_x[i], alpha, bins \
                    = self.computational_graph_end2end_quan_on_lpc(
                    res_x * self._res_scalar,
                    quan_lpc_x_poly,
                    the_share,
                    is_quan_on,
                    self._num_bins_for_follower[i],
                    'scope_'+str(i + 1),
                    the_stride)
                residual_coding_x[i] = residual_coding_x[i]/self._res_scalar
            else:
                _softmax_assignment[i], weight, decoded_fully, encoded, residual_coding_x[i], alpha, bins \
                    = self.computational_graph_end2end_quan_on_lpc(
                    self._res_scalar * (res_x - tf.expand_dims(tf.reduce_sum(input_tensor=
                                                                         residual_coding_x[:i], axis=0), axis=2)),
                    quan_lpc_x_poly,
                    the_share,
                    is_quan_on,
                    self._num_bins_for_follower[i],
                    'scope_' + str(i + 1),
                    the_stride)
                residual_coding_x[i] = residual_coding_x[i] / self._res_scalar
                # residual_coding_x[0] = inverse_mu_law_mapping(residual_coding_x[0] / self._res_scalar)
            #else:
            #    pass
        if self._num_resnets == 1:
            _softmax_assignment_2, encoded_2, bins_2, soft_assignment_fully_2 = _softmax_assignment, encoded, bins, _softmax_assignment
        return x, x_, lr, lpc_x, res_x, quan_lpc_x_poly, tau, the_share, is_quan_on, soft_assignment_lpc, _softmax_assignment, encoded, encoded, _softmax_assignment, _softmax_assignment, residual_coding_x, alpha, lpc_bins, bins, bins

    def _feedforward_lpc(self, num_res):
        with tf.Graph().as_default():
            x, x_, lr, lpc_x, res_x, quan_lpc_x_poly, tau, the_share, is_quan_on, _soft_assignment_lpc, _, encoded_1, encoded_2, _soft_assignment_, _soft_assignment_fully_2, residual_coding_x, alpha, lpc_bins, bins_1, bins_2 = self.all_modules_feedforward_lpc(num_res)
            decoded = np.sum(residual_coding_x, axis=0)

            synthesized = tf.compat.v1.py_func(lpc_synthesizer_tr, [quan_lpc_x_poly, decoded], [tf.float32])[0]

            #syn_time_loss = mse_loss(synthesized, x_[:, :, 0])
            #syn_freq_loss = mfcc_loss(synthesized, x_[:, :, 0])

            time_loss = mse_loss(decoded, res_x[:, :, 0])
            freq_loss = mfcc_loss(decoded, res_x[:, :, 0])
            ent_loss_lpc = entropy_coding_loss(_soft_assignment_lpc)
            ent_loss_1 = entropy_coding_loss(_soft_assignment_[0])
            ent_loss = ent_loss_1
            ent_loss_list = [ent_loss_lpc, ent_loss_1]
            if num_res == 2:
                ent_loss_2 = entropy_coding_loss(_soft_assignment_[1])
                ent_loss += ent_loss_2
            # quan_loss = tf.reduce_mean((tf.reduce_sum(tf.sqrt(_softmax_assignment + 1e-20), axis = -1) - 1.0), axis = -1)
            interested_var = [time_loss, freq_loss, ent_loss_1, ent_loss_lpc, ent_loss, ent_loss, ent_loss_list]

            saver = tf.compat.v1.train.Saver()
            with tf.compat.v1.Session() as sess:
                if self._num_resnets == 1:
                    saver.restore(sess, "./check/model_bnn_ac_" + self._rand_model_id + '_' + ".ckpt")
                    print("./check/model_bnn_ac_" + self._rand_model_id + '_' + ".ckpt")
                elif self._num_resnets == 2:
                    saver.restore(sess,
                                  "./check/model_bnn_ac_" + self._rand_model_id + '_' + self._save_unique_mark + self._suffix + ".ckpt")
                    print(
                        "./check/model_bnn_ac_" + self._rand_model_id + '_finetuning_' + self._save_unique_mark + self._suffix + ".ckpt")
                else:
                    print('OOOOPS')
                self.cmrl_eval_lpc(sess, x, x_, lr, lpc_x, synthesized, tau, the_share, is_quan_on, time_loss,
                                   encoded_1, encoded_2, decoded, alpha, lpc_bins, bins_1, bins_2, 100, 30,
                                   interested_var)

    def _feedforward(self, num_res):
        with tf.Graph().as_default():
            x, x_, lr, tau, the_share, is_quan_on, _softmax_assignment, encoded, soft_assignment_fully_1, \
            residual_coding_x, alpha, bins = self.all_modules_feedforward(num_res)
            decoded = np.sum(residual_coding_x, axis=0)
            ent_loss_arr = [0] * num_res
            for i in range(num_res):
                ent_loss_arr[i] = entropy_coding_loss(_softmax_assignment[i])
            interested_var = [tf.reduce_sum(ent_loss_arr)]
            saver = tf.compat.v1.train.Saver()
            with tf.compat.v1.Session() as sess:
                if num_res == 1:
                    print(
                        'model' + "./check/model_bnn_ac_" + self._rand_model_id + '_' + ".ckpt" + ' is restored!')
                    saver.restore(sess, "./check/model_bnn_ac_" + self._rand_model_id + '_' + ".ckpt")
                else:
                    print("./check/model_bnn_ac_" + self._rand_model_id +
                          # '_follower_' + str(num_res - 1) + 'end2endcascade' + ".ckpt" + ' is restored!')
                          '_follower_all_' + 'end2endcascade' + ".ckpt" + ' is restored!')
                    if self._from_where_step == 1:
                        saver.restore(sess, "./check/model_bnn_ac_" + self._rand_model_id +
                                      '_follower_' + str(num_res - 1) + 'end2endcascade' + ".ckpt")
                    elif self._from_where_step == 2:
                        saver.restore(sess, "./check/model_bnn_ac_" + self._rand_model_id +
                                      '_finetune_' + str(num_res) + 'end2endcascade' + ".ckpt")
                    else:
                        print('Wrong from_where_step.')
                # saver.restore(sess, "./check/model_bnn_ac_" + self._rand_model_id + '_' + ".ckpt")
                self.cmrl_eval(sess, x, x_, lr, the_share, is_quan_on, encoded, decoded,
                               alpha, bins, 100, 30, interested_var)
                    #sess, x, x_, lr, the_share, is_quan_on, encoded, decoded, \
                    #alpha, bins, how_many, the_epoch, interested_var = None

    def model(self, training_mode, arg):
        if training_mode == 'one_ae':
            # Train just one AE
            # self.one_ae_vq()
            print('one_ae')
            if is_pure_time_domain:
                self.one_ae()
            else:
                self.one_ae_lpc()
        elif training_mode == 'cascaded':
            # Train multiple AE in CMRL
            # self.one_ae_vq()
            if is_pure_time_domain:
                self.one_ae()
                for i in range(1, self._num_resnets):
                    self._greedy_followers(i)
            else:
                self.one_ae_lpc()
                for i in range(1, self._num_resnets):
                    self._greedy_followers_lpc(i)
            # self._finetuning(self._num_resnets)
            # self._greedy_followers_dumb(self._num_resnets)
        elif training_mode == 'retrain_from_somewhere':
            self._rand_model_id = arg.base_model_id
            if is_pure_time_domain:
                for i in range(1, self._num_resnets):
                    self._greedy_followers(i)
            else:
                for i in range(1, self._num_resnets):
                    self._greedy_followers_lpc(i)

            # self._finetuning(self._num_resnets)
            # self._finetuning_lpc(self._num_resnets)
        elif training_mode == 'finetune':
            self._rand_model_id = arg.base_model_id
            # self._finetuning(self._num_resnets)
            if is_pure_time_domain:
                self._finetuning(self._num_resnets)
            else:
                self._finetuning_lpc(self._num_resnets)
        elif training_mode == 'feedforward':
            model_id = arg.base_model_id
            self._rand_model_id = model_id
            if is_pure_time_domain:
                self._feedforward(self._num_resnets)
            # self._feedforward(self._num_resnets)
            else:
                self._feedforward_lpc(self._num_resnets)
        else:
            pass

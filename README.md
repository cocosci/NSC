# Efficient And Scalable Neural Residual Waveform Coding With Collaborative Quantization
[![LICENSE](https://img.shields.io/badge/license-MIT-green)](https://github.com/cocosci/pam-nac-v2/master/LICENSE)
[![Python](https://img.shields.io/badge/Python-3.6-purple)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0-orange)](https://www.tensorflow.org/)
[![Paper](https://img.shields.io/badge/PDF-IEEEXplore-blue)](https://ieeexplore.ieee.org/document/9054347/)

Scalability and efficiency are desired in neural speech codecs, which supports a wide range of bitrates for applications
 on various devices. We propose a collaborative quantization (CQ) scheme to jointly learn the codebook of LPC
 coefficients and the corresponding residuals. CQ does not simply shoehorn LPC to a neural network, but bridges
 the computational capacity of advanced neural network models and traditional, yet efficient and domain-specific
 digital signal processing methods in an integrated manner. We demonstrate that CQ achieves much higher quality
 than its predecessor at 9 kbps with even lower model complexity. We also show that CQ can scale up to 24 kbps where it
 outperforms AMR-WB and Opus. As a neural waveform codec, CQ models are with less than 1 million parameters,
 significantly less than many other generative models.

Please consider citing our papers if this helps.
```
@inproceedings{zhen2020cq,
  author={Kai Zhen and Mi Suk Lee and Jongmo Sung and Seungkwon Beack and Minje Kim},
  title={{Efficient And Scalable Neural Residual Waveform Coding with Collaborative Quantization}},
  year=2020,
  booktitle={Proceedings of the IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP) 2020},
  doi={10.1109/ICASSP40776.2020.9054347}
  url={https://ieeexplore.ieee.org/document/9054347}
}
@inproceedings{Zhen2019,
  author={Kai Zhen and Jongmo Sung and Mi Suk Lee and Seungkwon Beack and Minje Kim},
  title={{Cascaded Cross-Module Residual Learning Towards Lightweight End-to-End Speech Coding}},
  year=2019,
  booktitle={Proc. Interspeech 2019},
  pages={3396--3400},
  doi={10.21437/Interspeech.2019-1816},
  url={http://dx.doi.org/10.21437/Interspeech.2019-1816}
}

```

# Demos
- Project Page - I: https://saige.sice.indiana.edu/research-projects/neural-audio-coding/
- Project Page - II: http://kaizhen.us/collaborative-quantization

# The Code Structure

- utilities.py: supporting functions for Hann windowing, waveform segmentation, and objective measure calculation
- lpc_utilities.py: LPC analyzer, synthesizer and related functions implemented in Python
- neural_speech_coding_module.py: model configuration, training and evaluation for one neural codec
- cmrl.py: model training and evaluation with multiple neural codecs
- loss_terms_and_measures: loss functions and others to calculate objective measures such as pesq
- nn_core_operator.py: some fundamental operations such as convolution and quantization
- constants.py: definitions on the frame size, sample rate and other initializations
- main.py: the entry file


# The Dataset

The experiment is conducted on TIMIT corpus. https://catalog.ldc.upenn.edu/LDC93S1


# Run The Code

## Training
```
python main.py --learning_rate_tanh 0.0002  # the learning rate for the 1st codec
               --learning_rate_greedy_followers '0.00002 0.000002'  # the learning rate for the added codecs and finetuning
               --epoch_tanh 200  # the epoch for the 1st codec
               --epoch_greedy_followers '50 50'  # the epoch for the added codecs and finetuning
               --batch_size 128
               --num_resnets 2  # number of neural codecs involved
               --training_mode 4  # see main.py for specifications
               --base_model_id '1993783'  # used for finetuning and evaluation
               --from_where_step 2  # used for finetuning and evaluation
               --suffix '_greedy_all_'  # the suffix of the name of the model to be saved
               --bottleneck_kernel_and_dilation '9 9 100 20 1 2'  # configuration of the ResNet block
               --save_unique_mark 'follower_all'  # the name of the model to be saved
               --the_strides '2'  # the stride value for the down sampling CNN layer
               --coeff_term '60 10 10 0'  # coefficients for the loss terms
               --res_scalar 1.0
               --pretrain_step 2  # number of pretrained step with no quantization
               --target_entropy 2.2  # target entropy
               --num_bins_for_follower '32 32'  # number of quantization bins
               --is_cq 1  # is collaborative quantization being enabled
```
## Evaluation
```
python main.py --training_mode 0  # the base_model_id will need to be set correctly, other settings do not need to be changed
```



# References
Our work is built upon several recent publications on end-to-end speech coding, trainable quantizer and LPCNet.
- [1] Douglas O’Shaughnessy, “Linear predictive coding,” IEEE potentials, vol. 7, no. 1, pp. 29–32, 1988.
- [2] J.-M. Valin and J. Skoglund, “LPCNet: Improving neural speech synthesis through linear prediction,” in Proceedings of the IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP), 2019.
- [3] S. Kankanahalli, “End-to-end optimized speech coding with deep neural networks,” in Proceedings of the IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP), 2018.
- [4] E. Agustsson, F. Mentzer, M. Tschannen, L. Cavigelli, R. Timofte, L. Benini, and L. V. Gool, “Soft-to-hard vector quantization for end-to-end learning compressible representations,” in Advances in Neural Information Processing Systems (NIPS), 2017, pp. 1141–1151.

Some of the code is borrowed from https://github.com/sri-kankanahalli/autoencoder-speech-compression
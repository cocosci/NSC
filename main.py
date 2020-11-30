# from neural_speech_coding_module import *
from cmrl import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test and see it argument parser works.')
    parser.add_argument('--learning_rate_tanh', type=float, help='learning_rate for training tanh NN')
    parser.add_argument('--learning_rate_greedy_followers', type=str,
                        help='learning_rate for training greedy followers NN')
    parser.add_argument('--epoch_tanh', type=int, help='epoch to train tanh NN.')
    parser.add_argument('--epoch_greedy_followers', type=str, help='epoch to fine tuning NN.')
    parser.add_argument('--from_where_step', type=int,
                        help='0: from beginning; 1 from the second resnet or the first follower...')
    parser.add_argument('--batch_size', type=int, help='batch size.')
    parser.add_argument('--num_resnets', type=int, help='num_resnets.')
    parser.add_argument('--training_mode', type=str, help='How to train the NN.')
    parser.add_argument('--base_model_id', type=str, help='which model to re-train?')
    parser.add_argument('--suffix', type=str, help='save model name suffix..')
    parser.add_argument('--window_size', type=int, help='window_size')
    parser.add_argument('--bottleneck_kernel_and_dilation', type=str, help='bottleneck_kernel_and_dilation')
    parser.add_argument('--is_cq', type=int, help='is_cq')
    parser.add_argument('--the_strides', type=str, help='the_strides')
    parser.add_argument('--save_unique_mark', type=str, help='save_unique_mark')
    parser.add_argument('--coeff_term', type=str, help='coeff_term')
    parser.add_argument('--res_scalar', type=float, help='res_scalar')
    parser.add_argument('--pretrain_step', type=int, help='pretrain_step')
    parser.add_argument('--target_entropy', type=float, help='target_entropy')
    parser.add_argument('--num_bins_for_follower', type=str, help='num_bins_for_follower')

    args = parser.parse_args()
    print(args)
    audio_coding_ae = CMRL(args)
    # audio_coding_ae = neuralSpeechCodingModule(args)

    if args.training_mode == '1':
        audio_coding_ae.model(training_mode='one_ae', arg=args)
    elif args.training_mode == '2':
        audio_coding_ae.model(training_mode='retrain_from_somewhere', arg=args)
    elif args.training_mode == '3':
        audio_coding_ae.model(training_mode='cascaded', arg=args)
    elif args.training_mode == '4':
        audio_coding_ae.model(training_mode='cascaded', arg=args)
    elif args.training_mode == '5':
        audio_coding_ae.model(training_mode='finetune', arg=args)
    elif args.training_mode == '0':
        audio_coding_ae.model(training_mode='feedforward', arg=args)
    else:
        print('WRONG INPUT...')

    train_more = input('Type 1 if you want to train more, 0 to move to binarization.')
    train_more = input('Type 1 if you want to train more, 0 to move to binarization.')
    train_more = input('Type 1 if you want to train more, 0 to move to binarization.')
import argparse
import os
import torch


class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        # data 
        self.parser.add_argument('--dataset_type', type=str, default='fine-grained', help='dataset type')
        self.parser.add_argument('--train_json', type=str, default="json/pose_train.json", help='path to train json')
        self.parser.add_argument('--val_json', type=str, default="json/pose_val.json", help='path to validation json')
        self.parser.add_argument('--test_json', type=str, default="json/pose_test.json", help='path to test json')
        self.parser.add_argument('--downsample', type=int, default=0, help='downsample density map')
        self.parser.add_argument('--roi', default=False, action='store_true', help='use roi mask')
        self.parser.add_argument('--gray', default=False, action='store_true', help='use gray image')
        self.parser.add_argument('--smap', default=False, action='store_true', help='load semantic map')
        self.parser.add_argument('--dmap_type', type=str, default='dot', help='density map type, dot, fix4, fix16, or adapt')
        self.parser.add_argument('--seg_gt_act', type=str, default='ignore', help='action for overlapped segmentation map')
        self.parser.add_argument('--loader', type=str, default='single', help='which data loader')

        # model 
        self.parser.add_argument('--model', type=str, default='direct_regression', help='model to use')
        self.parser.add_argument('--net', type=str, default='fcn', help='model to use')
        self.parser.add_argument('--input_cn', type=int, default=3, help='# of input image channels')
        self.parser.add_argument('--output_cn', type=int, default=2, help='# of output image channels')
        self.parser.add_argument('--hard_assign', type=bool, default=False, help='semantic hard assignment')
        self.parser.add_argument('--ignore_index', type=int, default=-1, help='ignore index in seg')
        self.parser.add_argument('--hourglass_iter', type=int, default=1, help='hourglass iterations')
        self.parser.add_argument('--multi_reg', default=False, action='store_true', help='use multi regressors')

        # train 
        self.parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
        self.parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for generator')
        self.parser.add_argument('--seg_lr', type=float, default=0.0001, help='learning rate for seg net')
        self.parser.add_argument('--weight', type=float, default=1, help='weight of final loss')
        self.parser.add_argument('--seg_w', type=float, default=1, help='weight of segmentation loss')
        self.parser.add_argument('--gpu_ids', type=int, default=0, help='gpu ids: e.g. 0,1,2')
        self.parser.add_argument('--scale', type=int, default=16, help='scale for detection')
        self.parser.add_argument('--grid', type=int, default=7, help='pooling width')
        self.parser.add_argument('--align', type=int, default=3, help='align pooling')
        self.parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
        self.parser.add_argument('--total_epochs', type=int, default=800, help='total epochs for training')
        self.parser.add_argument('--train_shuffle', type=bool, default=True, help='shuffle training data')
        self.parser.add_argument('--train_counter', default=False, action='store_true', help='train counter flag')
        self.parser.add_argument('--seg_loss', default=False, action='store_true', help='use seg loss')
        self.parser.add_argument('--final_loss', default=False, action='store_true', help='use final loss')
        self.parser.add_argument('--count_loss', default=False, action='store_true', help='use final loss')
        self.parser.add_argument('--prop', default=False, action='store_true', help='use dmap for propagation')
        self.parser.add_argument('--seg', default=False, action='store_true', help='use dmap for segmentation')
        self.parser.add_argument('--soft', default=False, action='store_true', help='use soft cross entropy for segmentation')
        self.parser.add_argument('--per', default=False, action='store_true', help='use perspective weight for propagation')
        self.parser.add_argument('--att', default=False, action='store_true', help='use attention')

        # experiment
        self.parser.add_argument('--name', type=str, default='0', help='name of the experiment') 
        self.parser.add_argument('--pre', type=str, default='', help='load checkpoint') 
        self.parser.add_argument('--pre_counter', type=str, default='', help='load counter checkpoint') 
        self.parser.add_argument('--nThreads', default=1, type=int, help='# threads for loading data')
        self.parser.add_argument('--ckp_dir', type=str, default='./checkpoints', help='model save dir')
        self.parser.add_argument('--log_dir', type=str, default='./runs', help='model save logs')
        self.parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results on screen')
        self.parser.add_argument('--log_freq', type=int, default=1, help='frequency of logging info to tensorboard')
        self.parser.add_argument('--val_freq', type=int, default=1, help='frequency of validating')
        self.parser.add_argument('--test', default=False, action='store_true', help='testing flag')
        self.parser.add_argument('--test_freq', type=int, default=5, help='frequency of testing')
        self.parser.add_argument('--save_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        opt = self.parser.parse_args()

        args = vars(opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        expr_dir = os.path.join(opt.ckp_dir, opt.name)
        if not os.path.exists(expr_dir):
            os.mkdir(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        self.opt = opt
        return self.opt

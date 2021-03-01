from __future__ import print_function
from collections import OrderedDict
from tensorboardX import SummaryWriter
import time
import os
import pdb
import torch
import numpy as np

class Logger():
    def __init__(self, opt):
        self.opt = opt
        self.log_dir = os.path.join(opt.log_dir, opt.name + '_' + opt.model)
        self.writer = SummaryWriter(self.log_dir)

    def log(self, log, step, prefix='Err'):
        for k in log.keys():
            self.writer.add_scalar(prefix + '/' + k, log[k], step)


class Validator():
    def __init__(self, suffix='val'):
        self.best_results = None # best group results
        self.best_p = 1000.0
        self.suffix = suffix
        self.results = None
        self.avg_results = None
    
    def validate(self, model, loader):
        self.global_err = 0
        self.acc = 0 # classification error for segmentation model
        self.relative_mae = np.zeros([1,2])
        self.prec = np.zeros([1,2])
        self.recall = np.zeros([1,2])
        #if 'back' in model.opt.model:
        #    self.prec = np.zeros([1,3])
        #    self.recall = np.zeros([1,3])

        self.p = 0.0
        self.results = None
        self.avg_results  = None
        p = 0
        with torch.no_grad():
            for i, data in enumerate(loader):
                p, results, rel_mae = model.validate(data)
                self.p += p # sum of error of all groups 
                self.relative_mae += rel_mae
                self.accumulate(results) # error of indivudual groups
                if 'det' in model.opt.model:
                    self.global_err += 0
                    acc, prec, recall = model.get_acc_prec_recall(model.output.detach(), model.target)
                    self.acc += acc
                    self.prec += prec
                    self.recall += recall
                elif 'direct' not in model.opt.model and 'sep' not in model.opt.model and 'localize' not in model.opt.model:
                    self.global_err += abs(model.dmap.detach().sum()-model.target.detach().sum())
                    acc, prec, recall = model.get_acc_prec_recall(model.seg_output.detach(), model.smap_target)
                    self.acc += acc
                    self.prec += prec
                    self.recall += recall
                else:
                    self.global_err += abs(model.dmap.detach().sum()-model.target.detach().sum())
                    #self.p = self.global_err 
        
        self.p = self.p/len(loader)/model.opt.output_cn
        self.relative_mae = self.relative_mae/len(loader)

        #pdb.set_trace()
        avg_results = self.avg(len(loader))
        if self.p < self.best_p:
            self.best_p = self.p
            self.best_results = avg_results
            model.save('best' + self.suffix)
            best = True
        else:
            best = False
        print('#' + self.suffix, end=': ')
        for k in avg_results.keys():
            print("%s: %.4f" % (k, avg_results[k]), end=', ')
        print('Avg %s MAE: %.4f, Relative Avg.: %.4f ' % (self.suffix, self.p, self.relative_mae.mean()))
        if 'direct' not in model.opt.model and 'sep' not in model.opt.model and 'localize' not in model.opt.model:
            print('#' + self.suffix, end=': ')
            print('total:  %.4f, Acc: %.2f' % (self.global_err / len(loader), (self.acc / len(loader))))
            print('#' + self.suffix, end=': ')
            print('Precsion and recall', end=', ')
            print(self.prec/len(loader), self.recall/len(loader))
        elif 'localize' in model.opt.model:
            print('total:  %.4f' % (self.global_err / len(loader)))
        if self.best_results is not None:
            print('#' + self.suffix, end=': ')
            for k in self.best_results.keys():
                print("Best %s: %.4f" % (k, self.best_results[k]), end=', ')
        print('best %s MAE: %.4f' % (self.suffix, self.best_p))
        return best

    def get_info(self):
        info = OrderedDict()
        info['performance'] = self.p
        info['best_performance'] = self.best_p
        for k in self.avg_results.keys():
            info[k] = self.avg_results[k]
        return info

    def accumulate(self, results):
        if self.results is None:
            self.results = results
        else:
            for k in results.keys():
                self.results[k] += results[k]

    def avg(self, n):
        self.avg_results = self.results
        for k in self.results.keys():
            self.avg_results[k] /= n
        return self.avg_results
        
class Printer():

    def display(self, counter, timer, model):
        counter.display()
        timer.display_steps()
        model.display()

class Counter():
    def __init__(self):
        self.curr_epochs = 0
        self.curr_steps = 0
        self.total_steps = 0

    def update_epoch(self):
        self.curr_epochs += 1
        self.curr_steps = 0
    
    def update_step(self):
        self.curr_steps += 1
        self.total_steps += 1

    def get_epochs(self):
        return self.curr_epochs

    def get_total_steps(self):
        return self.total_steps

    def get_steps(self):
        return self.curr_steps

    def display(self):
        print("Epoch: %d, steps: %d" % (self.get_epochs(), self.get_steps()), end=', ')

class Timer():
    def __init__(self):
        self.steps = 0
        self.epochs = 0

        self.start_time = time.time()

        self.total_time = 0
        self.total_data_time = 0
        self.total_opt_time = 0

        self.last_time = time.time()

    def get_epoch_time(self):
        if self.epochs == 0:
            return 0
        return (time.time()-self.start_time)/self.epochs

    def get_data_time(self):
        if self.steps == 0:
            return 0
        return self.total_data_time/self.steps

    def get_opt_time(self):
        if self.steps == 0:
            return 0
        return self.total_opt_time/self.steps

    def update_epoch(self):
        self.epochs += 1

    def update_data(self):
        curr_time = time.time()
        self.total_data_time += (curr_time-self.last_time)
        self.last_time = curr_time

    def update_step(self):
        curr_time = time.time()
        self.total_opt_time += (curr_time-self.last_time)
        self.last_time = curr_time
        self.steps += 1

    def get_times(self):
        return self.get_epoch_time(), self.get_data_time(), self.get_opt_time()

    def display_steps(self):
        print("Data time: %.4f, optimize time: %.4f" % (self.get_data_time(), self.get_opt_time()), end=', ')

    def display_epochs(self):
        print("Epoch time: %.4f" % (self.get_epoch_time()))


import numpy as np
from utils.utils import Counter, Timer, Printer, Validator, Logger
from models.model import Model
from datasets.data_loader import DataLoader
from options.options import Options
import pdb

# options
opt = Options().parse()

# data
data_loader = DataLoader(opt)

#pdb.set_trace()
# model
model = Model(opt).get_model()
if opt.pre is not '':
    model.load(opt.pre)
if opt.pre_counter is not '':
    model.load_counter(opt.pre_counter)

# utils classes
tester = Validator(suffix='test')
tester.best_p = 0 # will never save the model

# start testing
tester.validate(model, data_loader.get_test_loader()) 
    


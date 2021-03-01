import numpy as np
from utils.utils import Counter, Timer, Printer, Validator, Logger
from models.model import Model
from datasets.data_loader import DataLoader
from options.options import Options

# options
opt = Options().parse()

# data
data_loader = DataLoader(opt)

# model
model = Model(opt).get_model()
if opt.pre is not '':
    model.load(opt.pre)
if opt.pre_counter is not '':
    model.load_counter(opt.pre_counter)

# utils classes
counter = Counter()
timer = Timer()
printer = Printer()
validator = Validator()
tester = Validator(suffix='test')
logger = Logger(opt)

# start training
for epoch in range(opt.total_epochs):
    
    model.reset()

    for i, data in enumerate(data_loader.get_train_loader()):

        timer.update_data()

        # optimize
        model.set_data(data)
        model.optimize()

        counter.update_step()
        timer.update_step()

        if counter.get_steps() % opt.display_freq == 0:
            printer.display(counter, timer, model)
        if counter.get_steps() % opt.log_freq == 0:
            logger.log(model.get_info(), counter.get_total_steps(), prefix='Loss')

    counter.update_epoch()
    timer.update_epoch()
    timer.display_epochs()


    if counter.get_epochs() % opt.val_freq == 0:
        # validate model
        best = validator.validate(model, data_loader.get_val_loader()) # will save when appropriate
        if best and opt.test:
            tester.validate(model, data_loader.get_test_loader()) # will save when appropriate
            logger.log(tester.get_info(), counter.get_epochs(), prefix='Test_Err')
        # log 
        logger.log(validator.get_info(), counter.get_epochs(), prefix='Val_Err')

    if opt.test and (counter.get_epochs() % opt.test_freq == 0):
        # test model
        tester.validate(model, data_loader.get_test_loader()) # will save when appropriate
        # log
        logger.log(tester.get_info(), counter.get_epochs(), prefix='Test_Err')
    
    if counter.get_epochs() % opt.save_freq == 0:
        # save current model
        model.save('latest')




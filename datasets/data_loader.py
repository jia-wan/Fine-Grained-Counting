import torch.utils.data
from datasets.fine_grained_dataset import FineGrainedDataset

class DataLoader():
    def __init__(self, opt):
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.opt = opt
        if 'fine-grained' in opt.dataset_type:
            self.train_dataset = FineGrainedDataset(opt, 'train')
            self.val_dataset = FineGrainedDataset(opt, 'val')
            self.test_dataset = FineGrainedDataset(opt, 'test')
        else:
            raise ValueError("Dataset [%s] not recognized." % opt.dataset_type)

    def get_train_loader(self):
        if self.train_loader is None:
            self.train_loader = torch.utils.data.DataLoader(
                    self.train_dataset,
                    batch_size=self.opt.batchSize,
                    shuffle=self.opt.train_shuffle,
                    num_workers=int(self.opt.nThreads))
        return self.train_loader

    def get_val_loader(self):
        if not self.val_loader:
            self.val_loader = torch.utils.data.DataLoader(
                    self.val_dataset,
                    batch_size=self.opt.batchSize,
                    shuffle=self.opt.train_shuffle,
                    num_workers=int(self.opt.nThreads))
        return self.val_loader

    def get_test_loader(self):
        if not self.test_loader:
            self.test_loader = torch.utils.data.DataLoader(
                    self.test_dataset,
                    batch_size=self.opt.batchSize,
                    shuffle=False,
                    num_workers=int(self.opt.nThreads))
        return self.test_loader

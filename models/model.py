from models.seg_att_prop_vgg import SegAttPropVGG

class Model():
    def __init__(self, opt):
        if 'vgg' in opt.model:
            self.model = SegAttPropVGG(opt)
        else:
            raise ValueError("Model %s not implemented!" % opt.model)

    def get_model(self):
        return self.model

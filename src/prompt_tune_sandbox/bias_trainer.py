from torch import nn
from transformers import Trainer

class BiasTrainerMaskedModel(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        import pdb; pdb.set_trace()
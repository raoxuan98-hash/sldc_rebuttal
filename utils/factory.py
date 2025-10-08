import torch
from models.sldc import SLDC
def get_model(model_name, args):
    name = model_name.lower()
    if 'sldc' in name:
        return SLDC(args)
    else:
        assert 0

from models.acil import ACILLearner
from models.dsal import DSALLearner
from models.sldc import SLDC


def get_model(model_name, args):
    name = model_name.lower()
    if 'sldc' in name:
        return SLDC(args)
    if 'acil' in name:
        return ACILLearner(args)
    if 'dsal' in name:
        return DSALLearner(args)
    raise ValueError(f'Unknown model name: {model_name}')

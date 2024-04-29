from .vsr_model import VSRModel
from .vsrgan_model import VSRGANModel
from .vsree_model import VSREEModel
from .ee_model import EEOnly

# register vsr model
vsr_model_lst = [
    'frvsr',
]

# register vsrgan model
vsrgan_model_lst = [
    'tecogan',
]


def define_model(opt):
    if opt['model']['name'].lower() in vsr_model_lst:
        model = VSRModel(opt)
    elif opt['model']['name'].lower() in vsrgan_model_lst:
        model = VSRGANModel(opt)
    elif opt['model']['name'].lower() == 'vsree':
        model = VSREEModel(opt)
    elif opt['model']['name'].lower() == 'ee_only':
        model = EEOnly(opt)
    else:
        raise ValueError(f'Unrecognized model: {opt["model"]["name"]}')

    return model

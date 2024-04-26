import math

import torch.nn as nn
import torch.optim as optim
import torch
import math

def define_criterion(criterion_opt):
    if criterion_opt is None:
        return None

    # parse
    if criterion_opt['type'] == 'MSE':
        criterion = nn.MSELoss(reduction=criterion_opt['reduction'])

    elif criterion_opt['type'] == 'L1':
        criterion = nn.L1Loss(reduction=criterion_opt['reduction'])

    elif criterion_opt['type'] == 'CB':
        from .losses import CharbonnierLoss
        criterion = CharbonnierLoss(reduction=criterion_opt['reduction'])

    elif criterion_opt['type'] == 'CosineSimilarity':
        from .losses import CosineSimilarityLoss
        criterion = CosineSimilarityLoss()

    elif criterion_opt['type'] == 'GAN':
        from .losses import VanillaGANLoss
        criterion = VanillaGANLoss(reduction=criterion_opt['reduction'])

    elif criterion_opt['type'] == 'LSGAN':
        from .losses import LSGANLoss
        criterion = LSGANLoss(reduction=criterion_opt['reduction'])

    elif criterion_opt['type'] == "EE":
        from .losses import EdgeEnhancedCriterion
        criterion = EdgeEnhancedCriterion(reduction=criterion_opt['reduction'])
    else:
        raise ValueError(f'Unrecognized criterion: {criterion_opt["type"]}')

    return criterion


def define_lr_schedule(schedule_opt, optimizer):
    if schedule_opt is None:
        return None

    # parse
    if schedule_opt['type'] == 'FixedLR':
        schedule = None

    elif schedule_opt['type'] == "StepLR":
        schedule = optim.lr_scheduler.StepLR(
            optimizer,
            step_size = schedule_opt['step_size'],
            gamma=schedule_opt['gamma']
        )

    elif schedule_opt['type'] == 'MultiStepLR':
        schedule = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=schedule_opt['milestones'],
            gamma=schedule_opt['gamma'])

    elif schedule_opt['type'] == 'CosineAnnealingRestartLR':
        from .lr_schedules import CosineAnnealingRestartLR
        schedule = CosineAnnealingRestartLR(
            optimizer,
            periods=schedule_opt['periods'],
            restart_weights=schedule_opt['restart_weights'],
            eta_min=schedule_opt['eta_min'])

    else:
        raise ValueError(f'Unrecognized lr schedule: {schedule_opt["type"]}')

    return schedule


class AlphaScheduler:
    def __init__(self,initial,decay_rate,decay_step):
        self.initial = initial
        self.decay_rate = decay_rate
        self.decay_step = decay_step
        self.global_step = 0

    def step(self):
        self.global_step += 1
        return self.initial * math.pow(self.decay_rate,(self.global_step/self.decay_step))





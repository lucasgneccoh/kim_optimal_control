import time
import os
import torch
import logging

# Function to create a timestap
def timestamp():
    return time.strftime('%Y%m%d_%H%M%S')

# Function to create a directoy with a timestamp
def create_dir(basedir='./', dirname = "results"):
    dir_path = os.path.join(basedir, f"{dirname}_{timestamp()}")
    os.makedirs(dir_path, exist_ok=True)
    return dir_path

# Function to save a torch model, optimizer, sceduler and epoch
def save_checkpoint(model, optimizer = None, scheduler = None, epoch = None, basedir='models', suffix = None, additional_info = {}):
    if suffix is None:
        suffix = timestamp()
    checkpoint_path = os.path.join(basedir, f'checkpoint_{suffix}.pth')
    to_save = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if not optimizer is None else None,
        'scheduler_state_dict': scheduler.state_dict()if not scheduler is None else None,
        'additional_info': additional_info
    }
    torch.save(to_save, checkpoint_path)
    return checkpoint_path

# Function to load a torch model, optimizer, sceduler and epoch
def load_checkpoint(checkpoint_path, model, optimizer = None, scheduler = None):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if not optimizer is None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if not scheduler is None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint['epoch'], checkpoint['additional_info']

# Function to set the logger
def setup_logging(root_path, log_level = 'INFO', fname='log.log'):
    level_dict = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARN': logging.WARNING,
        'ERROR': logging.ERROR,
        'FATAL': logging.CRITICAL
    }
    level = level_dict.get(log_level, logging.INFO)

    format_ = "[%(asctime)s %(filename)s:%(lineno)s] %(message)s"
    filename = '{}/{}'.format(root_path, fname)

    logger = logging.getLogger('kim')
    logger.setLevel(level)
    fh = logging.FileHandler(filename)
    fh.setFormatter(logging.Formatter(format_))
    fh.setLevel(level)
    logger.addHandler(fh)

    return logger
import json
from argparse import ArgumentParser
import torch
import numpy as np
from numpy import exp, log
import matplotlib.pyplot as plt
import time
import tqdm

import torch.nn as nn
import torch.optim as optim

# Load all the functions related to the problem itself
from optimal_control_bounday_value_utils import next_state, generate_initial_state, generate_sto_factor, generate_asset_info, composite_loss, GlobalNetworks

import utils



# Create argument parser to read the configuration file
parser = ArgumentParser(description='Train: Optimal Control with Boundary Value Problem: Optimal control network')
parser.add_argument('--config_model', type=str, default='./config/base_model.json', help='path to the configuration file with all the parameters for the base model')
parser.add_argument('--config_train', type=str, default='./config/base_train_optimal_control.json', help='path to the configuration file with all the parameters for the training of the optimal control network (u)')

args = parser.parse_args()






# ------- Base model parameters
# Read the configuration file for the base model
config = json.load(open(args.config_model, 'r'))


### Global Parameters
# - Dimension of variables
dim_S = config['dim_S']      # number of assets
dim_state = dim_S*2 + 4      # number of variables in the state process
dim_control = dim_S + 1      # number of variables in the control process
dim_sto_factor = dim_S + 1   # number of variables in the stochastic factor

# - Time steps
T = config['T']                 # terminal
n_period = config['n_period']   # number of periods until terminal
dt = T/n_period     # time step

### Parameters
# - Intervals for S_0, r_0, and c_t
min_moneyness, max_moneyness =  config['min_moneyness'], config['max_moneyness']
min_rate, max_rate = config['min_rate'], config['max_rate']
min_coupon, max_coupon = config['min_coupon'], config['max_coupon']

# - Dynamics of S and r
mu_S = config['mu_S']
sigma_S = config['sigma_S']
a, kappa, sigma_r = config['a'], config['kappa'], config['sigma_r']

# - Initial cash endownment and interval for liability proportion
initial_cash = config['initial_cash'] # changed to 10**1
min_liab, max_liab = config['min_liab'], config['max_liab']












# ------- Base model parameters
# Read the configuration file for the training
config = json.load(open(args.config_train, 'r'))


# - Hyper parameters and training info
n_hidden_layers = config["n_hidden_layers"]  # number of hidden layers in each sub-network used to estimate the optimal control
n_neuron = config["n_neuron"]        # number of neuron per hidden layer
n_sample = config["n_sample"]    # batch size for each iteration (epoch)
n_epoch = config["n_epoch"]       # number of training iterations

# - Parameter for loss functions
upper_bound = config["upper_bound"]  
lower_bound = config["lower_bound"]  
k = config["k"]                # coefficient of penalty for state constraints

# Optimizer and scheduler
lr = config["lr"]        # learning rate
milestones = config["milestones"]  # milestones for the learning rate scheduler
gamma = config["gamma"]  # decay factor for the learning rate scheduler



# ------ Start training
# Create a base directory to store results, checkpoints and figures
results_dir = utils.create_dir(basedir='./results', dirname = "train_u")
# Save configuration files for replication
json.dump(config, open(f'{results_dir}/config_train.json', 'w'))

logger = utils.setup_logging(results_dir, log_level = 'INFO', fname='train_u.log')

# - torch device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {device}")

# - Create the model
model = GlobalNetworks(dim_input = dim_state,
                           dim_output = dim_control,
                           dim_hidden = n_neuron,
                           n_hidden = n_hidden_layers,
                           n_subnetworks = n_period,
                           update_func=next_state)
model = model.to(device)
# - Optimizer

optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = milestones, gamma = gamma)
start_epoch = 0
losses = []
terminal_losses =[]
time_exec = []

# Load checkpoint if needed
if config['checkpoint_path'] is not None:
    start_epoch, additional_info = utils.load_checkpoint(config['checkpoint_path'], model, optimizer, scheduler)
    
    if "losses" in additional_info:
        losses = additional_info['losses']
    if "terminal_losses" in additional_info:
        terminal_losses = additional_info['terminal_losses']
    if "time_exec" in additional_info:
        time_exec = additional_info['time_exec']

    logger.info(f"Checkpoint loaded from {config['checkpoint_path']} at epoch {start_epoch}")
    # In case we load the checkpoint just to make the plots
    epoch = start_epoch
else:
    logger.info("No checkpoint loaded, starting from scratch")


# - Training data
initial_states = generate_initial_state(n_sample, dim_S, min_moneyness, max_moneyness, min_rate, max_rate, initial_cash, min_liab, max_liab)
sto_factors = generate_sto_factor(n_sample, n_period, initial_states, mu_S, sigma_S, a, kappa, sigma_r, dt)
coupon_intensity, S_tilde = generate_asset_info(n_sample, n_period, initial_states, min_coupon, max_coupon)

logger.info(f'Batch size = {n_sample}, Number of Assets = {dim_S}, Number of Periods/Subnetworks = {n_period}')
logger.info(f'Number of Layers per Subnetworks = {n_hidden_layers}, Number of Neurons per Layer = {n_neuron}')

start = time.time()

# - Training loop
for epoch in range(start_epoch, n_epoch - start_epoch):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    list_state = model(initial_states.to(device), sto_factors.to(device), coupon_intensity.to(device), S_tilde.to(device))
    loss_composition = composite_loss(list_state, upper_bound, lower_bound)
    terminal_loss, inter_loss_1, inter_loss_2 = loss_composition[0], loss_composition[1], loss_composition[2]
    loss = terminal_loss + k * (inter_loss_1 + inter_loss_2)

    # Backward propagation
    loss.backward()
    optimizer.step()

    losses.append(loss.item())
    terminal_losses.append(terminal_loss.item())

    scheduler.step()

    now = time.time()
    time_exec.append(now-start)
    
    # Progress tracking
    if (epoch + 1) % 50 == 0:
        logger.info(f"Epoch [{epoch + 1}/{n_epoch}], Training Loss: Total = {loss.item()}, Terminal = {terminal_loss}, Intermediate - bankruptcy = {inter_loss_1}, Intermediate - negative PM = {inter_loss_2} ")
        # Save model
        utils.save_checkpoint(model, optimizer = optimizer, scheduler = scheduler,
                                epoch = epoch, basedir=results_dir, suffix = f'epoch_{epoch+1}',
                                additional_info = {"losses": losses, "terminal_losses": terminal_losses, "time_exec": time_exec})



# - Save the model
utils.save_checkpoint(model, optimizer = optimizer, scheduler = scheduler,
                                epoch = epoch, basedir=results_dir, suffix = f'final',
                                additional_info = {"losses": losses, "terminal_losses": terminal_losses, "time_exec": time_exec})


# - Plot the loss curve
fig, ax = plt.subplots(1,3, figsize=(21,6),sharex=True)

ax[0].plot(losses)
ax[0].set_title('Training Loss')
ax[0].set(xlabel='Epoch', ylabel='G(X_T) + sum(g(X_t))')
ax[1].plot(terminal_losses)
ax[1].set_title('Terminal Loss')
ax[1].set(ylabel='G(X_T)')
ax[2].plot(time_exec)
ax[2].set_title('Training Time')
ax[2].set(ylabel='Time (in seconds)')

fig.savefig(f'{results_dir}/loss_curve.pdf')
logger.info(f"Curves saved to {results_dir}/loss_curve.pdf")
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
from optimal_control_bounday_value_utils import next_state, generate_initial_state, generate_sto_factor, generate_asset_info, composite_loss, GlobalNetworks, G, BoundaryValueNetwork

import utils



# Create argument parser to read the configuration file
parser = ArgumentParser(description='Train: Optimal Control with Boundary Value Problem: Value Network')
parser.add_argument('--config_model', type=str, default='./config/base_model.json', help='path to the configuration file with all the parameters for the base model')
parser.add_argument('--config_train', type=str, default='./config/base_train_value.json', help='path to the configuration file with all the parameters for the training of the value network (w)')
parser.add_argument('--config_unn', type=str, default='./config/base_train_optimal_control.json', help='path to the configuration file with all the parameters for the optimal control network (u)')

args = parser.parse_args()






# ------- Base model parameters
# Read the configuration file for the base model
config = json.load(open(args.config_model, 'r'))
# Create a base directory to store results, checkpoints and figures
results_dir = utils.create_dir(basedir='./results', dirname = "train_value")
# Save the configuration files for replication
json.dump(config, open(f'{results_dir}/config_base_model.json', 'w'))

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
# Save the configuration files for replication
json.dump(config, open(f'{results_dir}/config_train.json', 'w'))

# - New hyper parameters and training info
n_hidden_layers = config["n_hidden_layers"]       # number of hidden layers in each sub-network used to estimate the optimal control
n_neuron = config["n_neuron"]             # number of neuron per hidden layer
n_sample = config["n_sample"]         # number of sample trajectories (X_t)_{t=1,...,N}
n_sample_val = config["n_sample_val"]         # number of sample trajectories (X_t)_{t=1,...,N} in validation set
n_epoch =  config["n_epoch"]            # number of training batches
batch_size = config["batch_size"]
# Same parameters for the stochastic factors (S_t, r_t) and asset information (S_tilde, coupon_intensity)

# Optimizer and scheduler
lr = config["lr"]        # learning rate
milestones = config["milestones"]  # milestones for the learning rate scheduler
gamma = config["gamma"]  # decay factor for the learning rate scheduler




# ------ Create data

# Load the u network
config_unn = json.load(open(args.config_unn, 'r'))
# Save the configuration files for replication
json.dump(config_unn, open(f'{results_dir}/config_train_optimal_control.json', 'w'))

u_NN = GlobalNetworks(dim_input = dim_state,
                        dim_output = dim_control,
                        dim_hidden = config_unn['n_neuron'],
                        n_hidden = config_unn['n_hidden_layers'],
                        n_subnetworks = n_period,
                        update_func=next_state)

u_NN.load_state_dict(torch.load(config["u_nn_checkpoint"])['model_state_dict'])

### Training Data

# - Generate new initial state
new_initial_states = generate_initial_state(n_sample, dim_S, min_moneyness, max_moneyness, min_rate, max_rate, initial_cash, min_liab, max_liab)
new_sto_factors = generate_sto_factor(n_sample, n_period, new_initial_states, mu_S, sigma_S, a, kappa, sigma_r, dt)
new_coupon_intensity, new_S_tilde = generate_asset_info(n_sample, n_period, new_initial_states, min_coupon, max_coupon)
# - Validation initial states
new_initial_states_val = generate_initial_state(n_sample_val, dim_S, min_moneyness, max_moneyness, min_rate, max_rate, initial_cash, min_liab, max_liab)
new_sto_factors_val = generate_sto_factor(n_sample_val, n_period, new_initial_states_val, mu_S, sigma_S, a, kappa, sigma_r, dt)
new_coupon_intensity_val, new_S_tilde_val = generate_asset_info(n_sample_val, n_period, new_initial_states_val, min_coupon, max_coupon)

# - Apply the estimated control process to push the state variables forward until terminal
u_NN.eval()
with torch.no_grad():
    new_training_data = u_NN(new_initial_states, new_sto_factors, new_coupon_intensity, new_S_tilde) # dim = (n_sample, dim_state, n_period)
    new_data_val = u_NN(new_initial_states_val, new_sto_factors_val, new_coupon_intensity_val, new_S_tilde_val) # dim = (n_sample, dim_state, n_period)

    # - Calculate the target and rescale to size (n_sample * n_period,)
    new_final_states = new_training_data[:,:,-1].squeeze()
    target = G(new_final_states)                                        # dim = (n_sample,)
    print(f'target current size {target.size()}')
    target_rescale = target.repeat_interleave(n_period)                            # dim = (n_sample * n_period, )
    print(f'target rescale size {target_rescale.size()}')

    new_final_states_val = new_data_val[:,:,-1].squeeze()
    target_val = G(new_final_states_val)                                        # dim = (n_sample,)
    target_rescale_val = target_val.repeat_interleave(n_period)                            # dim = (n_sample * n_period, )


    # - Reshape new_training_data to size (n_sample * n_period, dim_state + 1) to feed into the NN
    new_training_data = torch.transpose(new_training_data, 1, 2)        # dim = (n_sample, n_period, dim_state)
    new_data_val = torch.transpose(new_data_val, 1, 2)        # dim = (n_sample, n_period, dim_state)

    training_data_rescale = torch.zeros((n_sample, n_period, dim_state + 1))
    training_data_rescale[:,:,1:] = new_training_data

    val_data_rescale = torch.zeros((n_sample_val, n_period, dim_state + 1))
    val_data_rescale[:,:,1:] = new_data_val

    time_var = torch.linspace(dt, T, n_period)
    time_var_train = time_var.repeat(n_sample, 1)
    time_var_val = time_var.repeat(n_sample_val, 1)
    training_data_rescale[:,:,0] = time_var_train
    val_data_rescale[:,:,0] = time_var_val


    training_data_rescale = training_data_rescale.reshape(n_period * n_sample, dim_state + 1)
    val_data_rescale = val_data_rescale.reshape(n_period * n_sample_val, dim_state + 1)





# ------ Start training
logger = utils.setup_logging(results_dir, log_level = 'INFO', fname='train_value.log')

# - torch device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {device}")
### Training Process

# - Create the model
model = BoundaryValueNetwork(dim_input = dim_state + 1,
                             dim_hidden = n_neuron,
                             n_hidden = n_hidden_layers,
                             activ_func = nn.ReLU())

model = model.cuda().float()

# Data
dataset = torch.utils.data.TensorDataset(training_data_rescale.to(torch.float64), target_rescale.to(torch.float64))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

val_dataset = torch.utils.data.TensorDataset(val_data_rescale.to(torch.float64), target_rescale_val.to(torch.float64))
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

# - Optimizer
optimizer = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = config["milestones"], gamma = config["gamma"])

# - Loss Function
loss_funct = nn.MSELoss()

losses = []
losses_val = []
time_exec = []
start_epoch = 0
# Load checkpoint if needed
if config['checkpoint_path'] is not None:
    start_epoch, additional_info = utils.load_checkpoint(config['checkpoint_path'], model, optimizer, scheduler)
    
    if "losses" in additional_info:
        losses = additional_info['losses']
    if "losses_val" in additional_info:
        losses_val = additional_info['losses_val']
    if "time_exec" in additional_info:
        time_exec = additional_info['time_exec']

    logger.info(f"Checkpoint loaded from {config['checkpoint_path']} at epoch {start_epoch}")
    # In case we load the checkpoint just to make the plots
    epoch = start_epoch
else:
    logger.info("No checkpoint loaded, starting from scratch")


logger.info(f'Number of training points = {n_sample*n_period}, Batch size = {batch_size}, Number of Layers = {n_hidden_layers}, Number of Neurons per Layer = {n_neuron}')
logger.info(f'Sample of an input (t,X_t) = {training_data_rescale[10,:]}')

start = time.time()

# - Training loop
N_train = len(dataset)
N_val = len(val_dataset)
for epoch in range(start_epoch, n_epoch-start_epoch):

    loss_sum = 0

    model.train()
    for batch_idx, (data, target) in enumerate(dataloader):
        optimizer.zero_grad()

        # Forward pass
        predict = model(data.cuda().float())

        # Loss computation
        loss = loss_funct(predict, target.cuda().float())

        # Backward propagation
        loss.backward()
        optimizer.step()

        # Remember to re-normalize
        loss_sum += loss.item() * data.size(1) / N_train

    losses.append(loss_sum)
    scheduler.step()

    now = time.time()
    time_exec.append(now-start)

    # Progress tracking
    if (epoch + 1) % 25 == 0:
        # Evaluation
        model.eval()
        with torch.no_grad():
            loss_sum_val = 0
            for batch_idx, (data, target) in enumerate(val_dataloader):
                predict = model(data.cuda().float())
                loss = loss_funct(predict, target.cuda().float())
                loss_sum_val += loss.item() * data.size(1) / N_val
        logger.info(f"Epoch [{epoch + 1}/{n_epoch}], Training Loss: {loss_sum}, Validatin Loss: {loss_sum_val}")
        losses_val.append(loss_sum_val)
        # Save model
        utils.save_checkpoint(model, optimizer = optimizer, scheduler = scheduler,
                                epoch = epoch, basedir=results_dir, suffix = f'epoch_{epoch+1}',
                                additional_info = {"losses": losses, "losses_val": losses_val, "time_exec": time_exec})
    else:
        # It is taking too long to obtain any feedback, printing at every epoch
        logger.info(f"Epoch [{epoch + 1}/{n_epoch}], Training Loss: {loss_sum}")


    


# - Save the model
utils.save_checkpoint(model, optimizer = optimizer, scheduler = scheduler,
                                epoch = epoch, basedir=results_dir, suffix = 'final',
                                additional_info = {"losses": losses, "losses_val": losses_val, "time_exec": time_exec})


# - Plot the loss curve
fig, ax = plt.subplots(1,3, figsize=(15,6),sharex=False)

ax[0].plot(losses)
ax[0].set_title('Training Loss')
ax[0].set(xlabel='Epoch', ylabel='MSE Loss')

ax[1].plot(losses_val)
ax[1].set_title('Val Loss')
ax[1].set(xlabel='Epoch', ylabel='MSE Loss')


ax[2].plot(time_exec)
ax[2].set_title('Training Time')
ax[2].set(ylabel='Time (in seconds)')


fig.savefig(f'{results_dir}/loss_curve.pdf')
logger.info(f"Curves saved to {results_dir}/loss_curve.pdf")

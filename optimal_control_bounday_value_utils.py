import torch
import numpy as np
from numpy import exp, log
import matplotlib.pyplot as plt
import time
import tqdm

import torch.nn as nn
import torch.optim as optim

# TODO: Add documentation for the function
# - Function to generate initial state
def generate_initial_state(n_sample, dim_S,
                           min_moneyness, max_moneyness,
                           min_rate, max_rate,
                           initial_cash,
                           min_liab, max_liab):
    initial_state = torch.empty(size=(n_sample, 2*dim_S + 4), dtype=torch.float32)
    initial_state[:,:dim_S] = torch.rand(size=(n_sample, dim_S))*(max_moneyness - min_moneyness) + min_moneyness  # Initial price
    initial_state[:,dim_S] = torch.rand(size=(n_sample,))*(max_rate - min_rate) + min_rate                        # Initial interest rate
    initial_state[:,dim_S+1] = torch.full(size=(n_sample,), fill_value=initial_cash)                              # Initial cash endowment
    initial_state[:, dim_S+2:-2] = torch.full(size=(n_sample, dim_S), fill_value=0)                               # Initial asset quantity : 0 asset to begin with
    initial_state[:,-2] = torch.full(size=(n_sample,), fill_value=initial_cash)                                   # Initial wealth (all in cash)
    initial_state[:,-1] = torch.rand(size=(n_sample,))*(max_liab - min_liab) * initial_cash                       # Initial liability (as a percentage of wealth)
    return initial_state


# TODO: Add documentation for the function
# - Function to generate stochastic factor (S_t, r_t) for the entire time horizon (t = 1, ..., N)
def generate_sto_factor(n_sample, n_period,
                        initial_state,
                        mu_S, sigma_S,
                        a, kappa, sigma_r, dt):
    dim_S = int((initial_state.size(dim=1) - 4)/2)

    xi = torch.empty(size=(n_sample, dim_S + 1, n_period + 1), dtype=torch.float32)
    Z = torch.normal(size=(n_sample, dim_S + 1, n_period), mean=0.0, std=1.0)       # Z : brownians

    xi[:, :, 0] = initial_state[:,:dim_S+1]

    for i in range(0,n_period):
        xi[:,:-1,i+1] = xi[:,:-1,i] + xi[:,:-1,i]*(mu_S * dt + sigma_S * np.sqrt(dt) * Z[:,:-1, i])
        xi[:,-1,i+1] = xi[:,-1,i] + a * (kappa -  xi[:,-1,i]) * dt + sigma_r * np.sqrt(dt) * Z[:,-1, i]

    return xi[:,:,1:]

# TODO: Add documentation for the function
# - Function to generate asset information
def generate_asset_info(n_sample, n_period,
                        initial_state,
                        min_coupon, max_coupon):
    dim_S = int((initial_state.size(dim=1) - 4)/2)

    coupon_intensity = torch.rand(size=(n_sample, dim_S, n_period))*(max_coupon - min_coupon) + min_coupon
    S_tilde = initial_state[:,:dim_S].clone().detach()
    S_tilde = S_tilde.unsqueeze(dim=2).expand(n_sample, dim_S, n_period)
    return coupon_intensity, S_tilde


class Subnetwork(nn.Module):
    def __init__(self, dim_input, dim_output, dim_hidden, n_hidden, activ_func=nn.ReLU()):
        '''
        dim_input : number of variables in the state process
            since dim(phi) = dim(S) and since we need to store values of r, beta, A, and L
            >> dim_input = dim(S) * 2 + 4
        dim_output : number of variable in the control process
            since dim(phi_dot) = dim(S) and since we need to store values of pi
            >> dim_output = dim(S) + 1
        dim_hidden : numer of neuron in each hidden layer
        n_hidden : number of hidden layers in each subnetwork

        '''
        super(Subnetwork, self).__init__()

        # Check dimension
        dim_S = int((dim_input - 4)/2)
        if dim_output != dim_S + 1 :
            raise ValueError(f'The dimensions of the control process and the state process do not match (dim_S by control = {dim_output - 1}, dim_S by state = {dim_S})')

        # Initiate a list to store layers
        layers = []

        # First hidden layer
        # [Lucas] Added BatchNorm1d
        layers.append(nn.BatchNorm1d([dim_input]))
        layers.append(nn.Linear(dim_input, dim_hidden))
        layers.append(activ_func)

        # In-between hidden layers
        for _ in range(n_hidden-1):
            # [Lucas] Added BatchNorm1d
            layers.append(nn.BatchNorm1d([dim_hidden]))
            layers.append(nn.Linear(dim_hidden, dim_hidden))
            layers.append(activ_func)

        # Last layer (output layer)
        layers.append(nn.Linear(dim_hidden, dim_output))

        # Combine layers into a sequential model
        self.model = nn.Sequential(*layers)



    def forward(self, x):
        '''
        x _ dim = (n_sample, dim_control) = (n_sample, dim(S)+1)
        '''
        # Go through the hidden layers
        x = self.model(x)

        # Last output (profit sharing proportion) is bounded between .85 and 1.0
        x[:, -1] = .85 + (1.0 - .85) * torch.sigmoid(x[:, -1])

        return x


# - Function to calculate percentage of lapse based on profit sharing
def lapse_percent(pi, b_0=-12, b_1=15):
    return torch.pow(1 + torch.exp(b_0 + b_1 * pi), -1)


# - Function to update the state variable given the control
def next_state(last_state, sto_factor, control, coupon_intensity, S_tilde, b_0=-12, b_1=15):
    '''
    input : assuming that we are at time t
        last_state : dim = (n_sample, 2 * dim(S) + 4) _ state variables of period t [ S_t, r_t, beta_t, phi_t, A_t, L_t ]
        sto_factor : dim = (n_sample, dim(S) + 1) _ independent state variables (stochastic factors) of period t + 1 [ S_(t+1), r_(t+1) ]
          >> may add new Brownians to avoid degeneration later on
        control : dim = (n_sample, dim(S) + 1) _ control variables for period t [ phi-dot_t , pi_t ]
        coupon_intensity : dim = (n_sample, dim_S)
        S_tilde : dim = (n_sample, dim_S)

    output :
        new_state : same dim as last_state - state variables of period t+1
    '''
    # check dimension
    dim_S = int((last_state.size(dim=1) - 4)/2)
    if sto_factor.size(dim=1) != dim_S + 1 :
        raise ValueError(f'The dimensions of the state process and the stochastic factors do not match (dim_S by sto factor = {sto_factor.size(dim=1) - 1}, dim_S by state = {dim_S})')

    # update price and interest rate
    dS = sto_factor[:, :dim_S] - last_state[:, :dim_S]                            # capital gain/loss per asset _ dim = (n_sample, dim_S)

    # make a new tensor for the new state (copy the last state to reserve the dimensions)
    new_state = torch.zeros_like(last_state)

    # calculate common drifts (to simplify the computation later) _ each drift should be of dimension (n_sample, )
    drift_1 = last_state[:, dim_S + 2] * sto_factor[:,-1] + torch.sum(last_state[:, dim_S+2:-2] * coupon_intensity, axis=1)   # revenue of the period (cash interest + coupon received)
    drift_2 = -1 * lapse_percent(control[:, -1], b_0, b_1) * last_state[:,-1]                                                 # amount of lapse (in euro)
    drift_3 = -1 * torch.sum(torch.min(torch.tensor(0.0), control[:,:-1]) * (new_state[:, 0:dim_S] - S_tilde), axis=1)        # realized capital gain/loss (only recognized in case of sale)

    new_state[:, :dim_S] = sto_factor[:, :-1]       # Price
    new_state[:, dim_S] = sto_factor[:, -1]         # Interest rate

        # Cash (beta)
    new_state[:, dim_S+1] = last_state[:, dim_S+1] + drift_1 + drift_2 - torch.sum(control[:,:-1] * last_state[:,0:dim_S], axis=1)
        # Asset portfolio (A)
    new_state[:, -2] = last_state[:, -2] + drift_1 + drift_2 - torch.sum(last_state[:,dim_S+2:-2] * dS, axis=1)
        # Liability portoflio (L)
    new_state[:, -1] = last_state[:, -1] + control[:,-1] * (drift_1 + drift_3) + drift_2
        # Quantity of asset (phi)
    new_state[:,dim_S+2:-2] = last_state[:,dim_S+2:-2] + control[:,:-1]

    return new_state


### Elementary functions of loss
def g(state):
    '''
    state : the state variable at a chosen period        _ dim = (n_sample, dim_state)
    g^1_(t) = (L_t - A_t)^2 if L_t > A_t and 0 otherwise _ dim = (n_sample,)
    g^2_(t) = (L_t)^2 if L_t < 0 and 0 otherwise         _ dim = (n_sample,)
    '''
    diff = state[:,-1] - state[:,-2]                              # L_t - A_t
    diff = torch.where(diff > 0.0, diff, 0)                       # select cases where L_t - A_t > 0
    g_1 = torch.where(diff > 1, .5 * torch.pow(diff,2), diff)     # dim = (n_sample,)

    liab = torch.where(state[:,-1] < 0.0, -1 * state[:,-1], 0)  # select cases where L_t < 0 and change it to its absolute value
    g_2 = torch.where(liab > 1, .5 * torch.pow(liab,2), liab)     # dim = (n_sample,)

    return g_1, g_2

def G(final_state, upper_bound=10**20, lower_bound=-10**20):
    '''
    final loss = min(upper_bound, max(lower_bound, L_T - A_T)) - this is the function G_T in our model

    final_state : the state variable at terminal         _ dim = (n_sample, dim_state)
    upper_bound, lower_bound : limits to bound G_T       _ scalars

    '''
    diff = final_state[:, -1] - final_state[:, -2]
    upper = torch.tensor(upper_bound, dtype=torch.float64)
    lower = torch.tensor(lower_bound, dtype=torch.float64)
    return torch.min(upper, torch.max(lower, diff))      # dim = (n_sample,)

### Aggregate loss functions
# - Intermediate loss
def inter_loss(state):
   '''
   state _ dim = (n_sample, dim_state)
   '''
   g_1, g_2 = g(state)                           # g_1 and g_2 have dim = (n_sample,)
   return torch.mean(g_1), torch.mean(g_2)       # scalar

# - Terminal loss
def final_loss(final_state, upper_bound=10**20, lower_bound=-10**20):
    '''
    final_state _ dim = (n_sample, dim_state)
    upper_bound and lower_bound are scalars
    '''
    g = G(final_state, upper_bound, lower_bound)  # dim = (n_sample,)
    return torch.mean(g)                          # scalar

# - Loss function
def composite_loss(all_states, upper_bound=10**20, lower_bound=-10**20):
    '''
    all_state - dim = (n_sample, dim_state, n_period) : all values of the state process excluding the initial state (t = 1,...,N)
    coeff : coefficient for penalty terms (intermediate cost)
    '''
    n_period = all_states.size(dim=2)
    inter_cost_1 = 0.0
    inter_cost_2 = 0.0
    for i in range(n_period):
        state_i = all_states[:,:,i].squeeze()
        cost_1, cost_2 = inter_loss(state_i)
        inter_cost_1 += cost_1
        inter_cost_2 += cost_2

    final_state = all_states[:,:,-1].squeeze()
    final_cost = final_loss(final_state, upper_bound, lower_bound)

    return final_cost, inter_cost_1, inter_cost_2


class GlobalNetworks(nn.Module):
    '''
    This Neural Network (NN) represents the control process.
    '''
    def __init__(self, dim_input, dim_output, dim_hidden, n_hidden, n_subnetworks,
                 update_func, activ_func = nn.ReLU()):
        super(GlobalNetworks, self).__init__()
        self.num_subnetworks = n_subnetworks
        self.update_func = update_func
        self.subnetworks = nn.ModuleList([
                                            Subnetwork(dim_input, dim_output, dim_hidden, n_hidden, activ_func)
                                            for _ in range(n_subnetworks)
                                        ])

    def forward(self, initial_state, sto_factor, coupon_intensity, S_tilde):

        '''
        inputs :
            initial_state : dim = (n_sample, dim_state)
            sto_factor : dim = (n_sample, dim_S + 1, n_period)
        outputs :
            all_states : list of all the states excluding the initial state _ dim = (n_sample, dim_state, n_period)
        '''
        # Stock dimension info
        n_sample = sto_factor.size(dim=0)     # batch size
        n_period = sto_factor.size(dim=2)     # number of periods
        dim_state = initial_state.size(dim=1) # number of variable in the state process

        # Initiation
        current_state = initial_state                 # taken as input
        list_state = torch.empty(size=(n_sample, dim_state, n_period))

        # Loop : go through the time horizon, compute the control (NN) at each period, and apply it on the current state to get to the new state
        for t, subnet in enumerate(self.subnetworks):
            # Variables
            sto_factor_t = sto_factor[:,:,t]
            control_t = subnet(current_state)
            coupon_intensity_t = coupon_intensity[:,:,t]
            S_tilde_t = S_tilde[:,:,t]

            # Apply the control to get a new state
            current_state = self.update_func(current_state, sto_factor_t, control_t, coupon_intensity_t, S_tilde_t)

            # Store the new state
            list_state[:,:,t] = current_state

        return list_state



class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()

    def forward(self, x):
        print()
        print(x)
        print()
        return x

class BoundaryValueNetwork(nn.Module):
    def __init__(self, dim_input, dim_hidden, n_hidden, activ_func = nn.ReLU(), print_layers=False):
        '''
        dim_input : 1 time dimension and dim_state dimension for the state process
            >> dim_input = dim_state + 1
        dim_hidden : numer of neuron in each hidden layer
        n_hidden : number of hidden layers in each subnetwork
        '''
        super(BoundaryValueNetwork, self).__init__()

        # Initiate a list to store layers
        layers = []

        # First hidden layer
        layers.append(nn.BatchNorm1d(dim_input))
        if print_layers:
            layers.append(PrintLayer())
        layers.append(nn.Linear(dim_input, dim_hidden))
        if print_layers:
            layers.append(PrintLayer())
        layers.append(activ_func)

        # In-between hidden layers
        for _ in range(n_hidden-1):
            if print_layers:
                layers.append(PrintLayer())
            layers.append(nn.BatchNorm1d(dim_hidden))
            if print_layers:
                layers.append(PrintLayer())
            layers.append(nn.Linear(dim_hidden, dim_hidden))
            layers.append(activ_func)

        # Last layer (output layer) - output is a scalar
        if print_layers:
            layers.append(PrintLayer())
        layers.append(nn.BatchNorm1d(dim_hidden))
        if print_layers:
            layers.append(PrintLayer())
        layers.append(nn.Linear(dim_hidden, 1))
        if print_layers:
            layers.append(PrintLayer())

        # Combine layers into a sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, z):
        '''
        z _ dim = (batch_size, dim_state + 1) : a sample of (t,X^{u_NN}_t)
          >> note that z includes time as a variable, so its second dimension is dim(X) + dim(t) = dim_state + 1
          >> given that we have n_sample trajectories, each of which contains n_period states, batch_size = n_sample * n_period
        '''
        z = self.model(z)
        z = torch.squeeze(z)
        return z  # should have dim = (batch_size,)
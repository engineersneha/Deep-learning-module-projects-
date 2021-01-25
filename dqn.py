import torch as T
#No convolutional layers needed for LunarLander environment
#So linear layers are sufficient in this problem
import torch.nn as nn
#for ReLu network
import torch.nn.functional as F
#for Adam optimizer
import torch.optim as optim
import numpy as np
# from torch.utils.tensorboard import SummaryWriter
import math
import skopt
from skopt import gp_minimize, forest_minimize
from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_convergence
from skopt.plots import plot_objective, plot_evaluations

#DQN built
class DQN(nn.Module):
    #Initialisation function
    def __init__(self,lr_rate,input_dims,fc1_dims,fc2_dims,n_actions):
        #Calls the constructor for the base class
        super (DQN,self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(*self.input_dims,self.fc1_dims)
        self.fc2 = nn.Linear (self.fc1_dims,self.fc2_dims)
        #Estimate value of set of actions given a certain state
        self.fc3 = nn.Linear (self.fc2_dims,self.n_actions)
        #Adam optimizer for backpropagation
        self.optimizer = optim.Adam (self.parameters(),lr=lr_rate)
        #Mean squared error loss
        self.loss = nn.MSELoss()
        self.device = T.device ('cpu')
        self.to(self.device)

    #Define forward propagation function
    def forward (self,state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        #No activation here, we just want to get agent's raw estimate
        actions = self.fc3(x)

        return actions

#RL agent defined
class Agent():
    def __init__(self,gamma,epsilon,lr_rate,input_dims,batch_size,n_actions
              ,max_memory_size = 100000,eps_end = 0.01,eps_decay= 0.005):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_decay
        self.lr = lr_rate
        #List of integer representations of available actions
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_memory_size
        self.batch_size = batch_size
        #Keep track of position of first available memory
        self.mem_counter = 0

        self.Q_eval = DQN(self.lr,n_actions=n_actions, input_dims=input_dims
                         ,fc1_dims=256,fc2_dims=256)
        self.state_memory = np.zeros((self.mem_size,*input_dims)
                                    ,dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size,*input_dims)
                                        ,dtype=np.float32)
        #dtype integers because action space is discrete
        self.action_memory = np.zeros(self.mem_size,dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size,dtype = np.float32)
        #value of terminal state is always 0, terminal memory tells
        #agent to update Q value
        self.terminal_memory = np.zeros (self.mem_size,dtype=np.bool)

    #function to store memories for replay
    def store_transition (self,state,action,reward,state_,done):
        #modulus ensure memory size is 0-99999, once 100000th
        #memory reached, goes back to 0
        index= self.mem_counter%self.mem_size
        self.state_memory[index]= state
        self.new_state_memory[index]=state_
        self.reward_memory[index]=reward
        self.action_memory[index]=action
        self.terminal_memory[index]=done

        self.mem_counter +=1

    #function for action selection (epsilon-greedy)
    def choose_action(self,observation):
        if np.random.random()>self.epsilon:
            state=T.tensor([observation]).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    #Learning function
    def learn(self):
        #start learning as soon as batch size of memory is filled up
        #until then just return, no point learning yet
        if self.mem_counter<self.batch_size:
            return
        #Zero the gradient
        self.Q_eval.optimizer.zero_grad()
        max_memory = min(self.mem_counter,self.mem_size)
        #Do not select same memory again
        batch = np.random.choice(max_memory,self.batch_size,replace=False)
        #Important to be able to slice properly later
        batch_index = np.arange(self.batch_size,dtype=np.int32)
        state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = T.tensor (self.new_state_memory[batch]).to(self.Q_eval.device)
        reward_batch = T.tensor (self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = T.tensor (self.terminal_memory[batch]).to(self.Q_eval.device)
        #actions as a numpy array not tensor
        action_batch = self.action_memory[batch]

        #Start performing feed forwards through NN to get parameters for loss function
        #Choose maximal actions

        #Array slicing to get values of actions we actually took at each set of memory batch
        q_eval=self.Q_eval.forward(state_batch)[batch_index,action_batch]
        q_next =self.Q_eval.forward(new_state_batch)
        q_next[terminal_batch] = 0.0
        #Max value- choosing greedy action
        q_target=reward_batch+self.gamma*T.max(q_next,dim=1)[0]
        loss = self.Q_eval.loss(q_target,q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()
        if (self.epsilon>self.eps_min):
            self.epsilon = self.epsilon - self.eps_dec
        else:
            self_epsilon = self.eps_min
        #Bayesian optimization for hyperparameter tuning
        # self.batch_size = [32,64,128,200]
        # self.eps_decay = Real(low = 0.0001, high =0.01, prior = 'log-uniform')
        # self.gamma = Real(low = 0.90, high =0.99, prior = 'log-uniform')
        # self.lr = Real(low = 0.00001, high =0.0001, prior = 'log-uniform')
        # self.epsilon = Real(low = 0.9 , high =1.0, prior = 'log-uniform')
        # hyper_param = [self.batch_size,self.eps_decay,self.epsilon,self.gamma,self.lr]
        # results = skopt.forest_minimize(reward_batch,hyper_param)
        # skopt.plots.plot_convergence (results)
        return loss.item()

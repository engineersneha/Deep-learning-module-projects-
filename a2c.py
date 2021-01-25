import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

#ActorCritic neural network defined
class ActorCriticNetwork (nn.Module):
    #Initialisation function
    def __init__(self,alpha,input_dims,fc1_dims,fc2_dims,n_actions):
        super(ActorCriticNetwork,self).__init__()
        #List or tuple for input dimensions
        self.input_dims = input_dims
        #Fully connected layers dimensions for Actor and Critic (should be the same)
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        #1st NN layer: unpacking input_dims (expects list or tuple) and provide as input, output as fc1_dims
        self.fc1 = nn.Linear(*self.input_dims,self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims,self.fc2_dims)
        #Actor- approximates agent policy (probability of taking an action)
        self.pi =nn.Linear(self.fc2_dims,n_actions)
        #Critic- approximates value function (value of each action that was just taken)
        self.v = nn.Linear(self.fc2_dims,1)
        #Adam optimizer for backpropagation
        self.optimizer = optim.Adam(self.parameters(),lr=alpha)
        self.device = T.device ('cpu')
        self.to(self.device)

    #function for forward pass
    def forward (self,observation):
        #Receives numpy array from environment and converts to tensor here
        state = T.Tensor(observation).to(self.device)
        #Pass state to 1st fully connected layer and activate with a relu
        x = F.relu(self.fc1(state))
        #Pass x to 1st fully connected layer and activate with a relu
        x = F.relu(self.fc2(x))
        #Find agent policy and value function
        pi =self.pi(x)
        v  =self.v(x)

        return pi, v

#RL agent
class Agent(object):

    def __init__(self,alpha,input_dims,gamma=0.99, layer1_size=256,layer2_size=256,n_actions=2):
        self.gamma = gamma
        self.ActorCritic = ActorCriticNetwork(alpha,input_dims,layer1_size,layer2_size,n_actions=n_actions)
        #Log of probability of action taken (value used to update weights of NN)
        self.log_probs = None

    def choose_action (self,observation):
        #Expect two values when this function is called but we do not need 2nd value now (value)
        policy,_= self.ActorCritic.forward(observation)
        #Sum of probabilities = 1
        policy = F.softmax(policy)
        #Categorised by probabilities dictated by our policy
        action_probs = T.distributions.Categorical(policy)
        action = action_probs.sample()
        #For use in calculation of loss function
        self.log_probs = action_probs.log_prob(action)
        #Integer representation of action tensor
        return action.item()
    #Learn function
    def learn (self,state, reward,state_,done):
        #Zero the gradients at the top of every learning iteration - because pytorch keeps track of intermediary steps but we do not want their influence here
        self.ActorCritic.optimizer.zero_grad()
        #Propagate state and new state to get critic value
        _,critic_value = self.ActorCritic.forward(state)
        _,critic_value = self.ActorCritic.forward(state_)
        #Convert numpy reward to tensor
        reward = T.tensor (reward, dtype= T.float).to(self.ActorCritic.device)
        #Calculate TD error (how far we are from optimal)
        delta = reward + self.gamma *critic_value*(1-int(done))-critic_value
        actor_loss = self.log_probs *delta
        critic_loss = delta**2
        #Sum of losses to backpropagate to NN
        (actor_loss+critic_loss).backward()
        total_loss = actor_loss.item()+critic_loss.item()
        self.ActorCritic.optimizer.step()
        return total_loss

#Import libraries
import numpy as np
import gym
import random
import time

#Create custom map for extended grid of 10x10 (25 hole states, 1 goal state and 1 start state)
custom_map =['SFFFFFFFFF',
             'FFHFFFFFFH',
             'FFFFHFFHFF',
             'FFHFHFFFHF',
             'FFFFHFFHFF',
             'HHFFFFHFFF',
             'FHHFFFFFFH',
             'FFFHFHHFFF',
             'HFHFFFHFFH',
             'FFHFHFFHFG']

#Initialize Frozen Lake environment from OpenAI Gym with our defined map of the grid
env = gym.make("FrozenLake-v0", desc = custom_map)
#Initialize variable for number of actions (because env object is not subscriptable)
n_actions = env.action_space.n
#Intialize variable for number of states (because env object is not subscriptable)
n_states = env.observation_space.n
#Intialize q-table with random samples from uniform distribution of number of states and actions
q_table = np.random.uniform (0,1e-4,(n_states,n_actions))
#States where holes are
holes = {12,19,24,27,32,34,38,44,47,50,51,56,61,62,69,73,75,76,80,82,86,89,92,94,97}
#Goal state
goal ={99}
#Terminal states
terminal= {12,19,24,27,32,34,38,44,47,50,51,56,61,62,69,73,75,76,80,82,86,89,92,94,97,99}
for state in terminal:
    q_table[state,:] = 0
#Prints initial q-table
#print(q_table)

##Intialize parameters
#Total number of training episodes
total_episodes = 20000
#Step size was initally set to 1 but then 0.8 gave the best score
learning_rate = 0.8
#Set maximum steps for each episode
max_steps =200
#Set discount factor to 0.9
gamma = 0.9
#Set epsilon initially to 1 (will be discounted after each episode)
epsilon = 1.0
#Maximum epsilon stays 1 for all episodes
max_epsilon =1.0
#Minimum epsilon 0.001 for all episodes
min_epsilon = 0.001
#Decay factor for exponential decrease of epsilon after each episode (to encourage exploration)
decay = 0.0001
#Empty array assigned to store rewards
rewards =[]
#Loop for each episode
for episode in range (total_episodes):
    #Reset state to initial/start state
    state = env.reset()
    #Step initialized as 0
    step = 0
    #Environment does not need to be reset or goal not reached
    done = False
    #Total rewards start with 0 (initial state is frozen)
    total_rewards = 0
    #Loop for each step (number of steps per episode defined)
    for step in range (max_steps):
        #Increments step value for next loop
        step +=1
        #Variable that will contain random sample from uniform distribution of 0 and 1
        e_tradeoff = random.uniform (0,1)
        #Checks if random sample is more than epsilon (condition always false for initial case of epsilon=1)
        if (e_tradeoff>epsilon):
            #Greedy action chosen by choosing max q value (exploitation)
            action = np.argmax(q_table[state,:])
        else:
            #random action (exploration)
            action = env.action_space.sample()
        #print (action)
        #Take action on the environment
        new_state, reward, done, info = env.step(action)
        #Choose greedy action for new state reached after taking previous greedy action
        action_max = np.argmax(q_table[new_state])
        #if state after action is in hole then reward -1 if not 1 for goal or 0 otherwise
        if new_state in holes:
            reward = -1.0
        elif new_state in goal:
            reward = 1.0
        else:
            reward = 0.0
        if reward==1.0:
            env.render()
            print ("Done!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        #Calculate q-value (equations from pseudo code-table 5.2)
        q = reward+gamma*(q_table[new_state,action_max])-q_table[state,action]
        q_table[state,action] = q_table[state,action]+(learning_rate*q)
        #Add reward to total rewards
        total_rewards +=reward
        #Reset state to new state reached
        state = new_state
        #print (state)
        #Reset action to new greedy action to take at new state
        action = action_max
        #print(action)
        #Display environment after action taken
        #env.render()
        #If goal achieved or in hole (finish episode)
        if (done==True):
            break
    #Update epsilon value for next episode(reduce epsilon to encourage exploration)
    epsilon = min_epsilon + (max_epsilon-min_epsilon)*np.exp(-decay*episode)
    #Append total rewards computed to rewards array intialized
    rewards.append(total_rewards)
#Calculate score based on average rewards
score = (sum(rewards)/total_episodes)
print("score", str(score) )
print("Episode finished after {} timesteps".format(step+1))
print(q_table)

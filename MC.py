#Import libraries
import gym
import numpy as np
import operator
from time import sleep
import random

#Initialize Frozen Lake environment from OpenAI Gym
env = gym.make ('FrozenLake-v0')
#Check number of discrete actions and states for this environment
print (env.action_space)
print(env.observation_space)
#Random sample generation for action and state
print (env.action_space.sample())
print(env.observation_space.sample())

#Define function for creating arbitrarily e-soft policy (for Frozen Lake environment)
def create_random_policy (env):
    #Initialize policy as empty list, (s,a) returned at end of function to this list for all s in S and a in A(s)
    policy ={}
    #Loop for as long as state in state space
    for key in range(0,env.observation_space.n):
        current_end =0
        #Initialize probability as empty list
        p={}
        #Loop for as long as action in action space
        for action in range(0,env.action_space.n):
            #Probability of taking each action assigned equal probabilty of at least (e/|A(s)|), where 0<e<1
            p[action] = 0.9/env.action_space.n
        #Policy assigns to state, probability of all actions that can be taken
        policy[key]=p
    return policy

#Define function to store rewards for action-values
def create_state_action_dict(env,policy):
    #Initialize empty list for rewards
    Q={}
    #Loop for each state
    for key in policy.keys():
        #Initialize 0 for all action values that is possible at that state
        Q[key]={a:0.0 for a in range(0,env.action_space.n)}
    return Q

#Define function to play game (step through Frozen Lake environment)
def play_game(env,policy,display=True):
    #Set state to starting state
    env.reset()
    #States where holes are
    holes = {5,7,11,12}
    #Goal state
    goal = {15}
    #Empty array assigned to store episodes
    episode= []
    #Episode incomplete
    done = False
    rewards =[]
    #When done is True (loop for each episode when dropped in hole or goal achieved)
    while not done:
        #Store state
        s = env.env.s
        #print(s)
        #Display environment for each action taken
        #env.render()
        #Empty list assigned to store information from steps of an episode
        timestep=[]
        #Add state of step
        timestep.append(s)
        #Random sample from uniform distribution of sum of action values at state
        n = random.uniform(0, sum(policy[s].values()))
        top_range = 0
        #Probability of each action added (probability of all actions at state)
        for prob in policy[s].items():
            top_range += prob[1]
            #Compare value of n and sum of actions probabilities, if doesn't exceed assign minimal probability (non-greedy) action to be taken-exploration
            if n < top_range:
                action = prob[0]
                break
        #Take action in the environment
        state,reward,done,info = env.step(action)
        #Reward structure
        if state in holes:
            reward = -1.0
        elif state in goal:
            reward = 1.0
        else:
            reward = 0.0
        if reward==1.0:
            print ("Done!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            #env.render()
        #Add action taken at that state
        timestep.append(action)
        #Add reward from next state
        timestep.append(reward)
        #Add steps information to episode information
        episode.append(timestep)
    return episode

#Define function to test policy
def test_policy (policy,env):
    wins=0
    r=1000
    for i in range(r):
        #Calculate reward from episode
        w= play_game(env,policy,display=False)[-1][-1]
        #If reward is 1, goal reached and it is a win!
        if w==1:
            wins+=1
            #Print percentage of wins over number of episodes
            print("percentage:",wins/r)
    return wins/r

#Define function for first visit MC control without exploring starts
def mc (env,episodes=1000,policy=None,epsilon=0.001):
    #To initialize e-soft policy
    if not policy:
        #Run create random policy function (creating empty dictionary to store state action values)
        policy = create_random_policy(env)
    #Run function to store rewards for action-value pairs in policy
    Q= create_state_action_dict(env,policy)
    #Initialize empty list for returns
    returns={}
    #Repeat for each episode as long as within total episodes given
    for _ in range(episodes):
        #Store cumulative reward term 'G' set intially to 0
        G= 0
        #Run play game function given environment and stores state, action and reward
        episode = play_game(env=env, policy=policy, display=False)
        #Loop for each timestep of episode
        #Steps counted from T-1,T-2,...,0, so we use 'reversed operation' (long-term reward obtained at the end so we have to go backwards from last timestep)
        for i in reversed (range(0,len(episode))):
            #Store state, action and reward in episode
            s_t,a_t,r_t = episode[i]
            #Store state and action
            state_action= (s_t,a_t)
            #Initialize value for discount factor on returns
            gamma = 0.9
            #Calculate returns with discount factor and reward from current timestep
            G = (gamma*G)+r_t
            #Check if state and action appear in list of state and action pairs in episode
            if not state_action in [(x[0],x[1]) for x in episode [0:i]]:
                #If not appeared, then we check if state and action are s_t and a_t
                if returns.get (state_action):
                    #if condition satisfied (to ensure no exploring starts!), we add G term to returns of state and action
                    returns[state_action].append(G)
                else:
                    #if condition not satisfied save current G term to returns
                    returns[state_action]=[G]
                #Calculate average reward across episode (average returns only for first time s is visited in an episode)
                Q[s_t][a_t]= sum (returns[state_action])/len(returns[state_action])
                score = Q[s_t][a_t]
                if score>0.5:
                    print ("score:",Q[s_t][a_t])
                    print("Episode finished after {} timesteps".format(i+1))
                #Define list for lambda expression to store all possible action values for that state
                Q_list = list (map(lambda x: x[1],Q[s_t].items()))
                #Create array of maximal action values in list
                indices = [i for i, x in enumerate (Q_list) if x==max(Q_list)]
                #Choose any one of maximal action
                max_Q = random.choice(indices)
                #Assign maximum action value to A_star (variable used in pseudo code)
                A_star = max_Q
                #Check if action in actions under policy, at s_t
                for a in policy[s_t].items():
                    #If action is maximal action chosen, continue
                    if a[0]==A_star:
                        #Assign probability for choosing greedy action to this action, (1-epsilon) is extra probability only for greedy action
                        policy[s_t][a[0]] = 1- epsilon +(epsilon/abs(sum(policy[s_t].values())))
                    else:
                        #Assign probability for non-greedy actions
                        policy[s_t][a[0]] = (epsilon/abs(sum(policy[s_t].values())))
    return policy

policy = mc (env,episodes=1000)
test_policy (policy,env)

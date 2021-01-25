import gym
from a2c import Agent
from plotcurve import plot_learning_curve
import numpy as np
import matplotlib.pyplot as plt
from lunarlanderenv import LunarLander

if __name__ == '__main__':
    env = LunarLander()
    #Start learning
    agent = Agent(alpha=0.00001,input_dims=[8],gamma=0.99,n_actions=4,layer1_size=2048,layer2_size=512)
    score_history,losses = [],[]
    total_episodes = 1000
    for i in range (total_episodes):
        done = False
        observation= env.reset()
        score =0
        total_loss = 0
        while not done:
            action = agent.choose_action(observation)
            observation_,reward,done,info = env.step(action)
            total_loss = agent.learn(observation,reward,observation_,done)
            observation=observation_
            score+=reward
        losses.append(total_loss)
        score_history.append(score)
        print('episode',i,'score%0.2f'%score)
    x =[i+1 for i in range (total_episodes)]
    fig = plt.figure()
    ax = fig.add_subplot(111,label="1")
    # ax1 = fig.add_subplot(111,label="2")
    ax.scatter(x,score_history,color='r')
    ax.set_xlabel("Epoch",color='r')
    ax.set_ylabel("score_history",color='r')
    # ax1.plot(x,losses,color='b')
    # ax1.set_xlabel("Epoch",color='b')
    # ax1.set_ylabel("losses",color='b')
    plt.savefig('score_a2c.png')

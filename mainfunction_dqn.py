import gym
from dqn import Agent
import matplotlib.pyplot as plt
import numpy as np
from plotcurve import plot_learning_curve
import argparse
from lunarlanderenv import LunarLander

if __name__ =='__main__':
    #Automate tuning of hyperparameters
    #Parse command lines from text to any data type (then pass to model)
    #parser = argparse.ArgumentParser(description='commandline')
    #hyphen to make 'n_games' argument optional
    # parser.add_argument('-n_games',type=int,default=500,help='number of games')
    # parser.add_argument('-lr_rate',type=float,default=0.0001,help='learning rate for optimizer')
    # parser.add_argument('-eps_end',type=float,default=0.01,help='final value for epsilon in greedy action selection')
    # parser.add_argument('-eps_decay',type=float,default=0.005,help='decrement factor to decrease epsilon')
    # parser.add_argument('-epsilon',type=float,default=1.0,help='start value for epsilon')
    # parser.add_argument('-batch_size',type=int,default=64,help='batch size for replay memory sampling')
    # parser.add_argument('-gamma',type=float,default=0.95,help='discount factor for future rewards')
    # args = parser.parse_args()
    # args.dims = [args.dims]
    env = LunarLander()
    agent = Agent(gamma=0.9,epsilon=1.0,batch_size=128,n_actions=4,eps_end=0.01,input_dims =[8],lr_rate=0.0001)
    #agent = Agent (args.gamma,args.batch_size,args.epsilon,args.eps_decay,args.eps_end,args.lr_rate,args.n_games)
    scores,eps_history, losses=[],[],[]
    n_games =500
    for i in range(n_games):
        score = 0
        done=False
        loss = 0
        observation=env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_,reward,done,info = env.step(action)
            score+=reward
            agent.store_transition(observation,action,reward,observation_,done)
            loss = agent.learn()
            observation=observation_
        losses.append(loss)
        scores.append(score)
        eps_history.append(agent.epsilon)
        avg_score = np.mean(scores[-100:])
        print('episode',i,'score%0.2f'%score)
        print('average_score%0.2f'%avg_score)
        print('epsilon%0.2f'%agent.epsilon)
    x =[i+1 for i in range (n_games)]
    filename = 'lunar_lander.png'
    plot_learning_curve(x,scores,eps_history,filename)
    fig = plt.figure()
    ax = fig.add_subplot(111,label="1")
    ax.plot(x,losses,color='b')
    ax.set_xlabel("Epoch",color='b')
    ax.set_ylabel("loss",color='b')
    plt.savefig('loss.png')

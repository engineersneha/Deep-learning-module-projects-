# Deep-learning-module-projects-
Project 1
- Solve Frozen Lake environment from OpenAI gym using Monte Carlo control method and Temporal Difference learning methods such as Q-learning and SARSA. 

Training codes for MC control: 
Mc.py - for original 4x4 grid problem, MC_extendedgrid.py- grid extended to 10x10 

Training codes for Q-learning: 
Q.py - for original 4x4 grid problem, Q_extendedgrid.py- grid extended to 10x10

Training codes for SARSA: 
sarsa.py - for original 4x4 grid problem, sarsa_extendedgrid.py- grid extended to 10x10 


Project 2
- Used Reinforcement learning techniques (Deep Q-Network and Advantage Actor-Critic methods) to solve the LunarLander environment from OpenAI gym 

Training and validation codes for DQN:
dqn.py - 2 classes defined (DQN and Agent), 
mainfunction_dqn.py - makes environment and uses agent to play a number of episodes on the environment, plots 
are produced for scores and losses 

Training and validation codes for A2C:
a2c.py - 2 classes defined (actorcriticnetwork and agent), 
mainfunction_a2c.py- makes environment and uses agent to play a number of episodes on the environment,
plots are produced for scores

Environment: 
lunarlanderenv.py

Plot function: 
plotcurve.py

Python libraries that need to be installed- gym and mujoco

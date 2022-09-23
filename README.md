# Deep_Q_network_RL
Reinforcement learning behaviour for single agent in gym environment

## Project Description
Deep_Q_Learning is the single agent based RL technique which combines the epsilon greedy strategy along with exploration and exploitation behaviour to learn the intelligent actions.
Open_AI based gym environments such as lunar_lander, cartpole etc having discrete actions are used here.


## Requirements:
* Python >= 3.8.10
* torch >=1.8.1+cpu
* numpy
* matplotlib
* math

## Installation
Run this command on command prompt to clone the repository


`git clone https://github.com/love481/Deep_Q_network_RL.git`

## Running code
To train or evaluate the models run on command line

`cd src`

`python main.py`

## Code Structure
* `\src\main.py` --> To start training or testing of models.Use evaluate == false to start training else evaluating.
* `\src\agent.py` --> Actions to each agent
* `\src\runner.py` --> Integrating all modules together to run the program.
* `\src\model.py` --> Pytorch implementation of policy part
* `\src\dqn.py` --> Implementation modules of dqn algorithm
* `\src\model\lunar_lander\..` --> All trained model learning for lunar_lander environment
* `\src\model\cart_pole\..` --> All trained model for cart_pole environment


## Results for cart_pole
https://user-images.githubusercontent.com/54012619/191902618-1bd6b84e-4733-4483-8a29-58dce5e6681d.mp4
### Training
![game_rewards_train](https://user-images.githubusercontent.com/54012619/191900257-d1273e0d-8078-4711-b988-d92e703c6fa9.png)

### Testing
![game_rewards](https://user-images.githubusercontent.com/54012619/191900263-88693d78-2941-4c3e-88cc-188d6b486c79.png)


## Results for lunar_lander
https://user-images.githubusercontent.com/54012619/191902758-0c4f0d11-f215-47f2-8e54-911a44a5372f.mp4
### Training
![rewards_training](https://user-images.githubusercontent.com/54012619/191902876-7158c091-5090-4a71-bbbb-2350a449336f.png)
### Testing
![game_rewards_test](https://user-images.githubusercontent.com/54012619/191902887-ec27fb7c-5d7c-420b-8643-848c1f3d5c92.png)

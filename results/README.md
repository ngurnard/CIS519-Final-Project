## Folder Naming convention

| Name | Obstacle | Living cost | Reward |
|---------------------------------: | :-------------------: | :-------------------------------------------: | :-------------------------------------------------: |
|[Baseline](https://github.com/ngurnard/CIS519-Final-Project/tree/master/results/Baseline)| NO | NO | Negative reward based on distance |
|[Baseline_withObstacle](https://github.com/ngurnard/CIS519-Final-Project/tree/master/results/Baseline_withObstacle)| YES | NO | Negative reward based on distance |
|[Hover_Obstacle_reward50](https://github.com/ngurnard/CIS519-Final-Project/tree/master/results/Hover_Obstacle_reward50)| YES | NO | Positve reward of 50 near goal |
|[Hover_Obstacle_reward50_withlivingcost](https://github.com/ngurnard/CIS519-Final-Project/tree/master/results/Hover_Obstacle_reward50_withlivingcost)| YES | YES | Positve reward of 50 near goal |
|[Hover_l2Reward](https://github.com/ngurnard/CIS519-Final-Project/tree/master/results/Hover_l2Reward)| NO | NO | Euclidean reward |
|[Hover_l2Reward_Obstacle_withlivingcost](https://github.com/ngurnard/CIS519-Final-Project/tree/master/results/Hover_l2Reward_Obstacle_withlivingcost)| YES | YES | Euclidean reward |
|[Hover_l2Reward_withlivingcost](https://github.com/ngurnard/CIS519-Final-Project/tree/master/results/Hover_l2Reward_withlivingcost)| NO | YES | Euclidean reward |
|[Hover_reward50_withlivingcost](https://github.com/ngurnard/CIS519-Final-Project/tree/master/results/Hover_reward50_withlivingcost)| NO | YES | Positve reward of 50 near goal |
|[Hover_reward50](https://github.com/ngurnard/CIS519-Final-Project/tree/master/results/Hover_reward50)| NO | NO | Positve reward of 50 near goal |

### Model folder naming convention(Sub folder)
`save-\<env\>-\<algo\>-\<obs\>-\<act\>-\<iter\>-\<time_date\>`
- env: Action to be performed
- algo: RL algorithm used from [Stable Baselines3](https://stable-baselines3.readthedocs.io/en/master/)
- obs: Observation type
- act: Action type
- iter: Number of total timesteps required for training
- time_date: Date and time when the model was trained

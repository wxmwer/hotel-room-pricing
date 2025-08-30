# hotel-room-pricing
Language: Python
Software: Pycharm

The file "model.zip" contains all the trained models, including PPO, DDPG, TD3, and AC. Please unzip it to the root directory.
The file "results.zip" contains all the episode rewards for the picture draw. If you run 'draw_pictures.py', please unzip it to the root directory first.

If you run the 'PPOtrain.py', please correctly use the parameters:
    parser.add_argument('--SEASON', type=int, default=0)  # off-season=0，peak-season=1
    
    parser.add_argument('--MODEL_TYPE', type=str, default='without_constraints')
    # MODEL_TYPE: without_constraints(DDP-N), with_constraints(DDP-C), dynamic_without_constraints(DP-N), dynamic_with_constraints(DP-C)

    parser.add_argument('--alpha_g', type=float, default=0.5)
    parser.add_argument('--alpha_t', type=float, default=0.5)
    
    parser.add_argument('--train', type=bool, default=False) # False: test the trained model, 
    parser.add_argument('--test', type=bool, default=True)
    # train_adjust is used for stage II
    parser.add_argument('--train_adjust', type=bool, default=False)
    
We do not upload the environment file of the deep reinforcement learning for the privacy protection, please contact wangxinmin@upc.edu.cn if you need.

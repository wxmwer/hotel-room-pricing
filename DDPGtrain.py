import environment
import numpy as np
import os
import argparse
#from utils import create_directory, plot_learning_curve
from DDPG import DDPG, ReplayBuffer
from tqdm import tqdm
import tensorflow as tf
#import xlwt
import matplotlib.pyplot as plt
import random
import keras
import time

# 动态定价 淡季 T=30 eps_dec=0.00001 tau=0.005 eps_end=0.005 alpha=0.01 reward/100000 - 3 , start_step = 500, gamma=0.90
# 动态定价 旺季 T=30 eps_dec=0.00001 tau=0.005 eps_end=0.005 alpha=0.01 reward/100000 - 3 , start_step = 500, gamma=0.90
# 差别动态定价 旺季 T=30 eps_dec=0.00005 tau=0.005 eps_end=0.005 alpha=0.02 reward/100000 - 10, gamma=0.92,start_step = 500,
# 差别动态定价 淡季 T=30 eps_dec=0.00001 tau=0.005 eps_end=0.005 alpha=0.02 reward/100000 - 7 , start_step = 500, gamma=0.92
# 差别定价 T=30 eps_dec=0.00005 tau=0.005 eps_end=0.01
def embedding(arr, input_dim, output_dim, RANDDOMSEED):
    # 创建Embedding层（使用随机初始化权重）
    embedding_layer = keras.layers.Embedding(
        input_dim=input_dim,  # 词汇表大小（确保覆盖所有输入整数）
        output_dim=output_dim,
        embeddings_initializer=keras.initializers.RandomNormal(seed=RANDDOMSEED),
        name='my_embedding'
    )
    # 将输入转换为Tensor并进行Embedding
    input_tensor = tf.convert_to_tensor(arr)
    embedded_tensor = embedding_layer(input_tensor)

    # 将TensorFlow张量转换回NumPy数组
    embedded_array = embedded_tensor.numpy().flatten()
    return embedded_array
def save_rewards(file_path, arr):
    """
    保存numpy数组到文件
    参数:
        arr: 要保存的numpy数组
        file_path: 保存文件的路径（通常使用.npy扩展名）
    """
    try:
        np.save(file_path, arr)
        print(f"episode奖励数据已成功保存到 {file_path}")
    except Exception as e:
        print(f"保存episode奖励数据时出错: {e}")

def load_rewards(file_path):
    """
    从文件加载numpy数组
    参数:
        file_path: 保存数组的文件路径
    返回:
        加载的numpy数组
    """
    try:
        arr = np.load(file_path)
        print(f"episode奖励数据已从 {file_path} 成功加载，形状为: {arr.shape}")
        return arr
    except FileNotFoundError:
        print(f"错误: 文件 {file_path} 未找到")
    except Exception as e:
        print(f"加载episode奖励数据时出错: {e}")
        return None
def action_check_1(env, action):
    """
    检查第一个约束
    """
    #获得self.delta_t[step],检查action中的self.N个值是否符合约束
    # for i in range(env.N - 1):
    #     for j in range(i+1, env.N):
    #         if abs(action[i] - action[j]) * 1000 > (1 - env.alpha_g) * env.delta_t[step]:
    #             return False
    # return True
    gap = (action.max()- action.min())
    bound = (1 - env.alpha_g) * env.delta_t[step] / 1000
    if gap > bound:
        return action.mean() + (action - action.mean())*bound/gap
    else:
        return action

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--TRAIN_EPISODES', type=int, default=2000)
    parser.add_argument('--TEST_EPISODES', type=int, default=10)
    parser.add_argument('--RANDOM_SEED', type=int, default=25)
    parser.add_argument('--T', type=int, default=30) #模拟天数
    parser.add_argument('--N', type=int, default=3) #客户类别数量
    parser.add_argument('--SEASON', type=int, default=0) #淡季0，旺季1
    parser.add_argument('--alpha_g', type=float, default=0.5)
    parser.add_argument('--alpha_t', type=float, default=0.5)
    parser.add_argument('--MEMORY_CAPACITY', type=int, default=10000)
    parser.add_argument('--ALG_NAME', type=str, default='DDPG')
    #parser.add_argument('--ENV_ID', type=str, default='Hotel_Price_Offseason')
    # MODEL_TYPE: without_constraints, with_constraints, dynamic_without_constraints, dynamic_with_constraints
    parser.add_argument('--MODEL_TYPE', type=str, default='without_constraints')
    parser.add_argument('--train', type=bool, default=True)
    parser.add_argument('--test', type=bool, default=True)
    parser.add_argument('--INPUT_DIM', type=int, default=200) #这是state的embedding使用的参数，嵌入的最大整数上限
    parser.add_argument('--OUTPUT_DIM', type=int, default=2) #这是state的embedding使用的参数，嵌入的每个整数输出的维度
    #parser.add_argument('--VAR_DISCOUNT', type=int, default=0.999)
    args = parser.parse_args()

    #随机数设置一定要写在程序的最前面，不要放在env = environment.Hotel_room_env后面
    np.random.seed(args.RANDOM_SEED)
    tf.random.set_seed(args.RANDOM_SEED)

    if args.SEASON == 1:
        ENV_ID = 'Hotel_Price_Peakseason_' + args.MODEL_TYPE + '_' + str(args.alpha_g)[2] + '_' + str(args.alpha_t)[2]
    else:
        ENV_ID = 'Hotel_Price_Offseason_' + args.MODEL_TYPE + '_' + str(args.alpha_g)[2] + '_' + str(args.alpha_t)[2]

    env = environment.Hotel_room_env(args.MODEL_TYPE, args.T, args.N, args.SEASON, args.RANDOM_SEED, args.alpha_g, args.alpha_t)
    env.reset()
    state_dim = env.state.shape[0] * args.OUTPUT_DIM # 状态纬度 8
    if args.MODEL_TYPE == 'without_constraints' or args.MODEL_TYPE == 'with_constraints': #动态区别定价（不考虑公平约束和考虑公平约束）
        action_dim = env.N
    else: #动态定价，仅考虑时间公平约束或不考虑约束
        action_dim = 1
    action_para_a = env.action_para_a
    action_para_b = env.action_para_b
    buffer = ReplayBuffer(args.MEMORY_CAPACITY)
    agent = DDPG(action_dim, state_dim, action_para_a, action_para_b, buffer)
    print(args.ALG_NAME + '_' + ENV_ID)
    t0 = time.time()

    # 不考虑约束时，淡季2000 episode VAR=0.02, VAR_DISCOUNT=0.995 ;旺季时VAR=0.02, VAR_DISCOUNT=0.995 ;
    # 不考虑公平性因子的约束时，淡季2000 episode即可，VAR_DISCOUNT=0.999，而考虑约束时，需要15000 epiosdes，VAR_DISCOUNT=0.9999
    # 淡季 存在约束 0.5 0.5公平参数下：VAR = 0.1  VAR_DISCOUNT = 0.999
    if args.MODEL_TYPE == 'without_constraints' and args.SEASON == 0:
        agent.var = 0.02
        agent.var_discount = 0.995
    if args.MODEL_TYPE == 'without_constraints' and args.SEASON == 1:
        agent.var = 0.02
        agent.var_discount = 0.9999

    if args.train:  # train
        all_episode_reward = []
        for episode in range(args.TRAIN_EPISODES):
            env.reset()
            episode_reward = 0
            state = env.state
            # env.price_min = np.zeros(env.N)
            # env.price_max = np.zeros(env.N)
            #if args.MODEL_TYPE == 'with_constraints' or 'dynamic_with_constraints':
            # if args.MODEL_TYPE == 'without_constraints' and args.SEASON == 0:
            #     if episode == 0:
            #         agent.var_discount = 0.9999
            #         agent.var = 0.1
                # if agent.var < 0.005:
                #     agent.var_discount = 1.0
                # if episode == 1000:
                #     agent.var_discount = 0.9999
                #     agent.var = 0.05
                # if episode == 1400:
                #     agent.var_discount = 0.9999
                #     agent.var = 0.01
                # if episode == 1800:
                #     agent.var_discount = 1
                #     agent.var = 0.005
            for step in range(env.T):
                embed_state = embedding(state, args.INPUT_DIM, args.OUTPUT_DIM, args.RANDOM_SEED)
                action = agent.get_action(embed_state)
                if args.MODEL_TYPE == 'with_constraints' or args.MODEL_TYPE == 'dynamic_with_constraints':
                    action = action_check_1(env, action)
                state_, reward, done, _, _ = env.step(action, step)
                done = 1 if done is True else 0
                embed_state_ = embedding(state_, args.INPUT_DIM, args.OUTPUT_DIM, args.RANDOM_SEED)
                buffer.push(embed_state, action, reward, embed_state_, done)
                if len(buffer) >= args.MEMORY_CAPACITY:
                    agent.learn()
                state = state_
                #env.state = state
                episode_reward += reward
                # if episode == 226:
                #     print('episode 226:', action)


                if done:
                    break

            all_episode_reward.append(episode_reward)
            # if episode == 0:
            #     all_episode_reward.append(episode_reward)
            # else:
            #     all_episode_reward.append(all_episode_reward[-1] * 0.9 + episode_reward * 0.1)
            print(
                'Training  | Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time(m): {:.4f}|VAR={:.5f}'.format(
                    episode + 1, args.TRAIN_EPISODES, episode_reward, (time.time() - t0)/60, agent.var)
            )


        agent.save(args.ALG_NAME, ENV_ID)
        plt.plot(all_episode_reward)
        if not os.path.exists('image'):
            os.makedirs('image')
        plt.savefig(os.path.join('image', '_'.join([args.ALG_NAME, ENV_ID])))

        if not os.path.exists('results'):
            os.makedirs('results')
        npy_name = '_'.join([args.ALG_NAME, ENV_ID]) + '.npy'
        save_rewards(os.path.join('results', npy_name), np.array(all_episode_reward))



    # if args.test:
    #     # test
    #     for episode in range(args.TEST_EPISODES):
    #         env.reset()
    #         episode_reward = 0
    #         state = env.state
    #         action_save = []
    #         for step in range(env.T):
    #             embed_state = embedding(state, args.INPUT_DIM, args.OUTPUT_DIM, args.RANDOM_SEED)
    #             action = agent.get_action(embed_state, greedy=True)
    #             state_, reward, done = env.step(action, step)
    #             state = state_
    #             episode_reward += reward
    #             action_save.append(action.numpy().tolist())
    #             if done:
    #                 break
    #         print('不使用loading的Testing  | Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
    #                 episode + 1, args.TEST_EPISODES, episode_reward, time.time() - t0) )
    #         print(action_save)



    if args.test:
        # test
        agent.load(args.ALG_NAME, ENV_ID)
        for episode in range(args.TEST_EPISODES):
            env.reset()
            episode_reward = 0
            state = env.state
            action_save = []
            state_save = []
            new_customers_save = []
            for step in range(env.T):
                embed_state = embedding(state, args.INPUT_DIM, args.OUTPUT_DIM, args.RANDOM_SEED)
                action = agent.get_action(embed_state, greedy=True)
                if args.MODEL_TYPE == 'with_constraints' or args.MODEL_TYPE == 'dynamic_with_constraints':
                    action = action_check_1(env, action.numpy())
                state_, reward, done, new_customers,_ = env.step(action, step)
                state = state_
                episode_reward += reward
                action_save.append((np.round(action*1000)).astype(int).tolist())
                state_save.append((np.around(state)).tolist())
                new_customers_save.append(new_customers.tolist())
                # if done:
                #     break
            print('Testing  | Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
                    episode + 1, args.TEST_EPISODES, episode_reward, time.time() - t0) )
            print(action_save)
            print(state_save)
            averge_price = []
            for i in range(env.T):
                divisor = sum(new_customers_save[i])
                aver_i = [a * b / divisor for a, b in zip(action_save[i], new_customers_save[i])]
                averge_price.append(np.round(np.sum(aver_i)).astype(int).tolist())
            print(np.array(averge_price))

            #averge_price = (action_save * new_customers_save)/


    #读取奖励数据并绘图
    npy_name = '_'.join([args.ALG_NAME, ENV_ID]) + '.npy'
    if os.path.exists('results/'+ npy_name):
        all_episode_reward = load_rewards(os.path.join('results', npy_name))
        plt.plot(all_episode_reward)
        plt.show()
    else:
        print("奖励数据不存在，无法读取！")

#Testing 中最好的一个  | Episode: 4/10  | Episode Reward: 0.8413  | Running Time: 2199.8439
# [[300, 368, 398], [300, 375, 406], [300, 386, 416], [300, 375, 409], [300, 381, 377], [300, 406, 470], [300, 406, 476], [300, 369, 411], [300, 384, 423], [300, 377, 420], [300, 366, 414], [300, 363, 398], [300, 394, 421], [300, 387, 468], [300, 381, 433], [300, 363, 418], [300, 366, 429], [300, 360, 426], [300, 394, 415], [300, 388, 438], [300, 416, 476], [300, 376, 426], [300, 331, 407], [300, 342, 413], [300, 380, 415], [300, 377, 414], [300, 390, 462], [300, 399, 476], [300, 355, 407], [300, 350, 372]]
# [[104, 30, 37, 22], [94, 25, 36, 26], [95, 30, 37, 22], [90, 33, 29, 24], [86, 39, 42, 29], [74, 28, 43, 15], [90, 25, 32, 24], [93, 28, 49, 19], [88, 28, 40, 16], [88, 21, 36, 15], [105, 24, 36, 21], [96, 30, 46, 21], [92, 35, 46, 21], [82, 31, 55, 29], [81, 19, 39, 31], [83, 19, 45, 31], [75, 21, 34, 22], [82, 30, 37, 24], [92, 33, 44, 16], [82, 30, 49, 18], [80, 18, 37, 25], [86, 32, 24, 28], [87, 23, 33, 36], [79, 27, 37, 30], [88, 23, 40, 18], [93, 31, 42, 15], [78, 25, 47, 27], [77, 24, 45, 24], [83, 27, 43, 24], [82, 0, 0, 0]]

import environment
import numpy as np
import os
import argparse
#from utils import create_directory, plot_learning_curve
from PPO import PPO
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import keras
import time

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
    gap = (action.max()- action.min())
    bound = (1 - env.alpha_g) * env.delta_t[step] / 1000
    if gap > bound:
        return action.mean() + (action - action.mean())*bound/gap
    else:
        return action

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ALG_NAME', type=str, default='PPO')
    parser.add_argument('--TRAIN_EPISODES', type=int, default=2000)
    parser.add_argument('--TEST_EPISODES', type=int, default=10)
    parser.add_argument('--SEASON', type=int, default=0)  # 淡季0，旺季1

    # MODEL_TYPE: without_constraints, with_constraints, dynamic_without_constraints, dynamic_with_constraints
    parser.add_argument('--MODEL_TYPE', type=str, default='without_constraints')
    # dynamic_with_constraints时alpha_g==1，alpha_t不为零；
    # with_constraints时 alpha_g和和alpha_t均不为零；
    # without_constraints和dynamic_without_constraints时不考虑alpha_g和alpha_t，即二者为0，结果中不包含公平因子的影响
    parser.add_argument('--alpha_g', type=float, default=0.5)
    parser.add_argument('--alpha_t', type=float, default=0.5)

    parser.add_argument('--MEMORY_CAPACITY', type=int, default=1000) # MEMORY_CAPACITY只适用于DDPG
    parser.add_argument('--RANDOM_SEED', type=int, default=25)
    parser.add_argument('--T', type=int, default=30) #模拟天数
    parser.add_argument('--N', type=int, default=3) #客户类别数量

    parser.add_argument('--train', type=bool, default=False)
    parser.add_argument('--test', type=bool, default=True)
    # train_adjust只用于解的调整阶段是否训练
    parser.add_argument('--train_adjust', type=bool, default=False)
    parser.add_argument('--INPUT_DIM', type=int, default=200) #这是state的embedding使用的参数，嵌入的最大整数上限
    parser.add_argument('--OUTPUT_DIM', type=int, default=2) #这是state的embedding使用的参数，嵌入的每个整数输出的维度
    parser.add_argument('--BATCH', type=int, default=32)
    args = parser.parse_args()

    #随机数设置一定要写在程序的最前面，不要放在env = environment.Hotel_room_env后面
    np.random.seed(args.RANDOM_SEED)
    tf.random.set_seed(args.RANDOM_SEED)

    #set the file names
    if args.alpha_g == 0 or args.alpha_g == 1:
        str_alpha_g = str(args.alpha_g)
    else:
        str_alpha_g = str(args.alpha_g)[2]
    if args.alpha_t == 0 or args.alpha_t == 1:
        str_alpha_t = str(args.alpha_g)
    else:
        str_alpha_t = str(args.alpha_t)[2]
    if args.SEASON == 1:
        ENV_ID = 'Hotel_Price_Peakseason_' + args.MODEL_TYPE + '_' + str_alpha_g + '_' + str_alpha_t
    else:
        ENV_ID = 'Hotel_Price_Offseason_' + args.MODEL_TYPE + '_' + str_alpha_g + '_' + str_alpha_t

    env = environment.Hotel_room_env(args.MODEL_TYPE, args.T, args.N, args.SEASON, args.RANDOM_SEED, args.alpha_g, args.alpha_t)
    env.reset()
    state_dim = env.state.shape[0] * args.OUTPUT_DIM # 状态纬度 8
    if args.MODEL_TYPE=='without_constraints' or args.MODEL_TYPE=='with_constraints':  # 动态区别定价（不考虑公平约束和考虑公平约束）
        action_dim = env.N
    else: # when dynamic 动态定价，仅考虑时间公平约束或不考虑约束
        action_dim = 1
    action_para_a = env.action_para_a
    action_para_b = env.action_para_b
    agent = PPO(state_dim, action_dim, action_para_a, action_para_b)
    print(args.ALG_NAME + '_' + ENV_ID)
    t0 = time.time()
    if args.train:  # train
        all_episode_reward = []
        for episode in range(args.TRAIN_EPISODES):
            env.reset()
            episode_reward = 0
            state = env.state
            if args.MODEL_TYPE == 'without_constraints' and args.SEASON==0:
                if episode == 0:
                    agent.var_discount = 0.99985
                    agent.var = 0.1
            if args.MODEL_TYPE == 'without_constraints' and args.SEASON==1:
                if episode == 0:
                    agent.var_discount = 0.999935
                    agent.var = 0.2
                # if episode == 0:
                #     agent.var_discount = 0.9999
                #     agent.var = 0.2
                # if episode == 2000:
                #     agent.var_discount = 0.9999
                #     agent.var = 0.1
                # if episode == 3000:
                #     agent.var_discount = 0.9999
                #     agent.var = 0.05
                # if episode == 4000:
                #     agent.var_discount = 0.9999
                #     agent.var = 0.02
                # if episode == 4400:
                #     agent.var_discount = 0.9999
                #     agent.var = 0.01
                # if episode == 4800:
                #     agent.var_discount = 1
                #     agent.var = 0.005
            if args.MODEL_TYPE == 'dynamic_without_constraints' and args.SEASON==0:
                if episode == 0:
                    agent.var_discount = 0.99985
                    agent.var = 0.1
            if args.MODEL_TYPE == 'dynamic_without_constraints' and args.SEASON==1:
                if episode == 0:
                    agent.var_discount = 0.999935
                    agent.var = 0.2
                # if episode == 0:
                #     agent.var_discount = 0.99995
                #     agent.var = 0.2
                # if episode == 2000:
                #     agent.var_discount = 0.9999
                #     agent.var = 0.1
                # if episode == 3000:
                #     agent.var_discount = 0.99993
                #     agent.var = 0.05
                # if episode == 4000:
                #     agent.var_discount = 0.99995
                #     agent.var = 0.02
                # if episode == 4400:
                #     agent.var_discount = 0.99995
                #     agent.var = 0.01
                # if episode == 4800:
                #     agent.var_discount = 1
                #     agent.var = 0.005
            if args.MODEL_TYPE == 'with_constraints' and args.SEASON==0:
                if episode == 0:
                    agent.var_discount = 0.9998646
                    agent.var = 0.15
            if args.MODEL_TYPE == 'dynamic_with_constraints' and args.SEASON==0:
                if episode == 0:
                    agent.var_discount = 0.99985
                    agent.var = 0.1
            if args.MODEL_TYPE == 'with_constraints' and args.SEASON==1:
                if episode == 0:
                    agent.var_discount = 0.9999323
                    agent.var = 0.3
            if args.MODEL_TYPE == 'dynamic_with_constraints' and args.SEASON==1:
                if episode == 0:
                    agent.var_discount = 0.9999323
                    agent.var = 0.3


            for step in range(env.T):
                embed_state = embedding(state, args.INPUT_DIM, args.OUTPUT_DIM, args.RANDOM_SEED)
                action = agent.choose_action(embed_state, greedy=False)
                if args.MODEL_TYPE=='with_constraints':
                    action = action_check_1(env, action)
                state_, reward, done, _, _ = env.step(action, step)
                #done = 1 if done is True else 0
                embed_state_ = embedding(state_, args.INPUT_DIM, args.OUTPUT_DIM, args.RANDOM_SEED)
                agent.store_transition(embed_state, action, reward)
                state = state_
                episode_reward += reward
                # if done == True:
                #     break
                if step == env.T - 1:
                    end_period = True
                else:
                    end_period = False
                if (step + 1) % args.BATCH == 0 or end_period or done:
                    agent.finish_path(embed_state_, end_period, done)
                    agent.update()
                if done:
                    break
            all_episode_reward.append(episode_reward)
            print('Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time(m): {:.4f} | var:{:.5f}'.format(
                    episode, args.TRAIN_EPISODES, episode_reward, (time.time() - t0)/60, agent.var))
        agent.save_ckpt(args.ALG_NAME, ENV_ID)
        plt.plot(all_episode_reward)
        if not os.path.exists('image'):
            os.makedirs('image')
        plt.savefig(os.path.join('image', '_'.join([args.ALG_NAME, ENV_ID])))
        if not os.path.exists('results'):
            os.makedirs('results')
        npy_name = '_'.join([args.ALG_NAME, ENV_ID]) + '.npy'
        save_rewards(os.path.join('results', npy_name), np.array(all_episode_reward))


    action_save_max = None
    if args.test:
        agent.load_ckpt(args.ALG_NAME, ENV_ID)
        reward_max = 0

        episode_max = 0
        time0 = time.time()
        for episode in range(args.TEST_EPISODES):
            env.reset()
            episode_reward = 0
            state = env.state
            action_save = []
            state_save = []
            new_customers_save = []


            for step in range(env.T):
                embed_state = embedding(state, args.INPUT_DIM, args.OUTPUT_DIM, args.RANDOM_SEED)
                action = agent.choose_action(embed_state, greedy=True)
                if args.MODEL_TYPE == 'with_constraints':
                    action = action_check_1(env, action)
                state_, reward, done, new_customers, _ = env.step(action, step)
                state = state_
                episode_reward += reward
                action_save.append((np.round(action * 1000)).astype(int).tolist())
                state_save.append((np.around(state)).tolist())
                new_customers_save.append(new_customers.tolist())
                # if done:
                #     break
            print('Testing  | Episode: {}/{}  | Episode Reward: {:.4f} '.format(
                episode + 1, args.TEST_EPISODES, episode_reward))
            print(action_save)
            print(state_save)
            if args.MODEL_TYPE == 'with_constraints':
                averge_price = []
                for i in range(env.T):
                    divisor = sum(new_customers_save[i])
                    aver_i = [a * b / divisor for a, b in zip(action_save[i], new_customers_save[i])]
                    averge_price.append(np.round(np.sum(aver_i)).astype(int).tolist())
                print(np.array(averge_price))

            if reward_max < episode_reward:
                reward_max = episode_reward
                episode_max = episode
                action_save_max = action_save
        time1 = time.time()
        print('The inference runing time is ', time1-time0)
        print('The best reward is {}, the best action is :{}'.format(reward_max, action_save_max))
    # 读取奖励数据并绘图
    npy_name = '_'.join([args.ALG_NAME, ENV_ID]) + '.npy'
    if os.path.exists('results/' + npy_name):
        all_episode_reward = load_rewards(os.path.join('results', npy_name))
        plt.plot(all_episode_reward)
        plt.show()
    else:
        print("奖励数据不存在，无法读取！")

    #继续针对选出的最优解进行调整，使之符合时间周期内的公平约束
    if args.test and args.train_adjust and args.MODEL_TYPE in {'with_constraints', 'dynamic_with_constraints'}:
        #agent = PPO(state_dim, action_dim, action_para_a, action_para_b)
        action_save_max = np.array(action_save_max)
        #计算非周末的均值
        if args.MODEL_TYPE == 'with_constraints':
            ave_action = np.zeros(env.N)
        else:
            ave_action = 0
        counter = 0
        for step in range(env.T):
            if (step - 5) % 7 != 0 and (step - 6) % 7 != 0:
                counter += 1
                ave_action += action_save_max[step]
                # for i in range(env.N):
                #     ave_action[i] += action_save_max[step][i]
        ave_action = (ave_action / counter).astype(int)  #非周末的均值

        #重新训练，只针对最优解的不符合约束的位置进行再次寻优
        all_episode_reward = []
        print('#########Start the second traings to adjust the \'action_save_max\'#########')
        t0 = time.time()
        MAX_EPISODE = args.TRAIN_EPISODES // 4
        for episode in range(MAX_EPISODE):
            env.reset()
            episode_reward = 0
            state = env.state
            if args.SEASON == 0:
                if episode == 0:
                    agent.var_discount = 0.9994585366
                    agent.var = 0.1
            if args.SEASON == 1:
                if episode == 0:
                    if args.MODEL_TYPE == 'with_constraints':
                        agent.var_discount = 0.99982
                        agent.var = 0.06
                    if args.MODEL_TYPE == 'dynamic_with_constraints':
                        agent.var_discount = 0.99977
                        agent.var = 0.1
            for step in range(env.T):
                #is_adjust = False
                if (step - 5) % 7 != 0 and (step - 6) % 7 != 0:
                    if args.MODEL_TYPE == 'with_constraints':
                        action_upperbound = ave_action + (1 - args.alpha_t) * env.sigma_i[:, 0] / 2
                        action_lowerbound = ave_action - (1 - args.alpha_t) * env.sigma_i[:, 0] / 2
                    else:
                        action_upperbound = ave_action[0] + (1 - args.alpha_t) * env.sigma_i[0] / 2
                        action_lowerbound = ave_action[0] - (1 - args.alpha_t) * env.sigma_i[0] / 2
                else:
                    if args.MODEL_TYPE == 'with_constraints':
                        action_upperbound = ave_action + (1 - args.alpha_t) * env.sigma_i[:, 1] / 2
                        action_lowerbound = ave_action - (1 - args.alpha_t) * env.sigma_i[:, 1] / 2
                    else:
                        action_upperbound = ave_action[0] + (1 - args.alpha_t) * env.sigma_i[1] / 2
                        action_lowerbound = ave_action[0] - (1 - args.alpha_t) * env.sigma_i[1] / 2
                embed_state = embedding(state, args.INPUT_DIM, args.OUTPUT_DIM, args.RANDOM_SEED)
                #if is_adjust is True:
                #     action = agent.choose_action(embed_state, greedy=False)
                #     if args.MODEL_TYPE == 'with_constraints' or args.MODEL_TYPE == 'dynamic_with_constraints':
                #         action = np.clip(action, action_lowerbound/1000, action_upperbound/1000)
                #         action = action_check_1(env, action)
                # else:
                #     action = action_save_max[step] / 1000
                action = agent.choose_action(embed_state, greedy=False)
                action = np.clip(action, action_lowerbound/1000, action_upperbound/1000)
                if args.MODEL_TYPE == 'with_constraints':
                    action = action_check_1(env, action)
                state_, reward, done, _, _ = env.step(action, step)
                # done = 1 if done is True else 0
                embed_state_ = embedding(state_, args.INPUT_DIM, args.OUTPUT_DIM, args.RANDOM_SEED)
                agent.store_transition(embed_state, action, reward)
                state = state_
                episode_reward += reward
                # if done == True:
                #     break
                if step == env.T - 1:
                    end_period = True
                else:
                    end_period = False
                if (step + 1) % args.BATCH == 0 or end_period or done:
                    agent.finish_path(embed_state_, end_period, done)
                    agent.update()
                if done:
                    break
            all_episode_reward.append(episode_reward)
            print('Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time(m): {:.4f} | var:{}'.format(
                episode, MAX_EPISODE, episode_reward, (time.time() - t0)/60, agent.var))
        agent.save_ckpt(args.ALG_NAME, ENV_ID+'_adjusted')
        plt.plot(all_episode_reward)
        if not os.path.exists('image'):
            os.makedirs('image')
        plt.savefig(os.path.join('image', '_'.join([args.ALG_NAME, ENV_ID+'_adjusted'])))
        if not os.path.exists('results'):
            os.makedirs('results')
        npy_name = '_'.join([args.ALG_NAME, ENV_ID+'_adjusted']) + '.npy'
        save_rewards(os.path.join('results', npy_name), np.array(all_episode_reward))

        reward_max = 0
        action_save_max = None
        episode_max = 0
        for episode in range(args.TEST_EPISODES):
            env.reset()
            episode_reward = 0
            state = env.state
            action_save = []
            state_save = []
            new_customers_save = []

            for step in range(env.T):
                if (step - 5) % 7 != 0 and (step - 6) % 7 != 0:
                    if args.MODEL_TYPE == 'with_constraints':
                        action_upperbound = ave_action + (1 - args.alpha_t) * env.sigma_i[:, 0] / 2
                        action_lowerbound = ave_action - (1 - args.alpha_t) * env.sigma_i[:, 0] / 2
                    else:
                        action_upperbound = ave_action[0] + (1 - args.alpha_t) * env.sigma_i[0] / 2
                        action_lowerbound = ave_action[0] - (1 - args.alpha_t) * env.sigma_i[0] / 2
                else:
                    if args.MODEL_TYPE == 'with_constraints':
                        action_upperbound = ave_action + (1 - args.alpha_t) * env.sigma_i[:, 1] / 2
                        action_lowerbound = ave_action - (1 - args.alpha_t) * env.sigma_i[:, 1] / 2
                    else:
                        action_upperbound = ave_action[0] + (1 - args.alpha_t) * env.sigma_i[1] / 2
                        action_lowerbound = ave_action[0] - (1 - args.alpha_t) * env.sigma_i[1] / 2
                embed_state = embedding(state, args.INPUT_DIM, args.OUTPUT_DIM, args.RANDOM_SEED)
                action = agent.choose_action(embed_state, greedy=True)
                action = np.clip(action, action_lowerbound / 1000, action_upperbound / 1000)
                if args.MODEL_TYPE == 'with_constraints':
                    action = action_check_1(env, action)
                state_, reward, done, new_customers, _ = env.step(action, step)
                state = state_
                episode_reward += reward
                action_save.append((np.round(action * 1000)).astype(int).tolist())
                state_save.append((np.around(state)).tolist())
                new_customers_save.append(new_customers.tolist())
                # if done:
                #     break
            print('Testing  | Episode: {}/{}  | Episode Reward: {:.4f} '.format(
                episode + 1, args.TEST_EPISODES, episode_reward))

            print(action_save)
            print(state_save)
            if args.MODEL_TYPE == 'with_constraints':
                averge_price = []
                for i in range(env.T):
                    divisor = sum(new_customers_save[i])
                    aver_i = [a * b / divisor for a, b in zip(action_save[i], new_customers_save[i])]
                    averge_price.append(np.round(np.sum(aver_i)).astype(int).tolist())
                print(np.array(averge_price))

            if reward_max < episode_reward:
                reward_max = episode_reward
                episode_max = episode
                action_save_max = action_save
        print('The best reward is {}, the best action is :{}'.format(reward_max, action_save_max))

    if args.train is False and args.test and args.train_adjust is False and args.MODEL_TYPE in {'with_constraints', 'dynamic_with_constraints'}:
        agent.load_ckpt(args.ALG_NAME, ENV_ID + '_adjusted')
        reward_max = 0
        episode_max = 0

        action_save_max = np.array(action_save_max)
        # 计算非周末的均值
        if args.MODEL_TYPE == 'with_constraints':
            ave_action = np.zeros(env.N)
        else:
            ave_action = 0
        counter = 0
        for step in range(env.T):
            if (step - 5) % 7 != 0 and (step - 6) % 7 != 0:
                counter += 1
                ave_action += action_save_max[step]
                # for i in range(env.N):
                #     ave_action[i] += action_save_max[step][i]
        ave_action = (ave_action / counter).astype(int)  # 非周末的均值

        action_save_max = None
        time_start = time.time()
        for episode in range(args.TEST_EPISODES):
            env.reset()
            episode_reward = 0
            state = env.state
            action_save = []
            state_save = []
            new_customers_save = []

            for step in range(env.T):
                if (step - 5) % 7 != 0 and (step - 6) % 7 != 0:
                    if args.MODEL_TYPE == 'with_constraints':
                        action_upperbound = ave_action + (1 - args.alpha_t) * env.sigma_i[:, 0] / 2
                        action_lowerbound = ave_action - (1 - args.alpha_t) * env.sigma_i[:, 0] / 2
                    else:
                        action_upperbound = ave_action[0] + (1 - args.alpha_t) * env.sigma_i[0] / 2
                        action_lowerbound = ave_action[0] - (1 - args.alpha_t) * env.sigma_i[0] / 2
                else:
                    if args.MODEL_TYPE == 'with_constraints':
                        action_upperbound = ave_action + (1 - args.alpha_t) * env.sigma_i[:, 1] / 2
                        action_lowerbound = ave_action - (1 - args.alpha_t) * env.sigma_i[:, 1] / 2
                    else:
                        action_upperbound = ave_action[0] + (1 - args.alpha_t) * env.sigma_i[1] / 2
                        action_lowerbound = ave_action[0] - (1 - args.alpha_t) * env.sigma_i[1] / 2
                embed_state = embedding(state, args.INPUT_DIM, args.OUTPUT_DIM, args.RANDOM_SEED)
                action = agent.choose_action(embed_state, greedy=True)
                action = np.clip(action, action_lowerbound / 1000, action_upperbound / 1000)
                if args.MODEL_TYPE == 'with_constraints':
                    action = action_check_1(env, action)
                state_, reward, done, new_customers,_ = env.step(action, step)
                state = state_
                episode_reward += reward
                action_save.append((np.round(action * 1000)).astype(int).tolist())
                state_save.append((np.around(state)).tolist())
                new_customers_save.append(new_customers.tolist())
                # if done:
                #     break
            print('Testing  | Episode: {}/{}  | Episode Reward: {:.4f} '.format(
                episode + 1, args.TEST_EPISODES, episode_reward))

            print(action_save)
            print(state_save)
            if args.MODEL_TYPE == 'with_constraints':
                averge_price = []
                for i in range(env.T):
                    divisor = sum(new_customers_save[i])
                    aver_i = [a * b / divisor for a, b in zip(action_save[i], new_customers_save[i])]
                    averge_price.append(np.round(np.sum(aver_i)).astype(int).tolist())
                print(np.array(averge_price))

            if reward_max < episode_reward:
                reward_max = episode_reward
                episode_max = episode
                action_save_max = action_save
        time_end = time.time()
        print('The best reward is {}, the best action is :{}'.format(reward_max, action_save_max))
        print('Inference runing time is ', time_end - time_start)
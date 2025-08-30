import environment
import numpy as np
import os
import argparse
#from utils import create_directory, plot_learning_curve
from AC import AC
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--TRAIN_EPISODES', type=int, default=2000)
    parser.add_argument('--TEST_EPISODES', type=int, default=10)
    parser.add_argument('--RANDOM_SEED', type=int, default=25)
    parser.add_argument('--T', type=int, default=30) #模拟天数
    parser.add_argument('--N', type=int, default=3) #客户类别数量
    parser.add_argument('--SEASON', type=int, default=0) #淡季0，旺季1
    #parser.add_argument('--alpha_g', type=float, default=0.5)
    #parser.add_argument('--alpha_t', type=float, default=0.5)
    parser.add_argument('--MEMORY_CAPACITY', type=int, default=300)
    parser.add_argument('--ALG_NAME', type=str, default='AC')
    #parser.add_argument('--ENV_ID', type=str, default='Hotel_Price_Offseason')
    parser.add_argument('--MODEL_TYPE', type=str, default='without_constraints') #'with_constraints,dynamic,discrimination
    parser.add_argument('--train', type=bool, default=True)
    parser.add_argument('--test', type=bool, default=True)
    parser.add_argument('--INPUT_DIM', type=int, default=200) #这是state的embedding使用的参数，嵌入的最大整数上限
    parser.add_argument('--OUTPUT_DIM', type=int, default=2) #这是state的embedding使用的参数，嵌入的每个整数输出的维度
    parser.add_argument('--BATCH', type=int, default=32)
    args = parser.parse_args()

    #随机数设置一定要写在程序的最前面，不要放在env = environment.Hotel_room_env后面
    np.random.seed(args.RANDOM_SEED)
    tf.random.set_seed(args.RANDOM_SEED)

    if args.SEASON == 1:
        ENV_ID = 'Hotel_Price_Peakseason'
    else:
        ENV_ID = 'Hotel_Price_Offseason'

    env = environment.Hotel_room_env(args.MODEL_TYPE, args.T, args.N, args.SEASON, args.RANDOM_SEED)
    env.reset()
    state_dim = env.state.shape[0] * args.OUTPUT_DIM # 状态纬度 8
    action_dim = env.N
    action_para_a = env.action_para_a
    action_para_b = env.action_para_b
    agent = AC(state_dim, action_dim, action_para_a, action_para_b)
    t0 = time.time()
    if args.train:  # train
        all_episode_reward = []
        for episode in range(args.TRAIN_EPISODES):
            env.reset()
            episode_reward = 0
            state = env.state
            for step in range(env.T):
                embed_state = embedding(state, args.INPUT_DIM, args.OUTPUT_DIM, args.RANDOM_SEED)
                action = agent.get_action(embed_state)
                state_, reward, done, _, _  = env.step(action, step)
                embed_state_ = embedding(state_, args.INPUT_DIM, args.OUTPUT_DIM, args.RANDOM_SEED)
                episode_reward += reward

                td_error = agent.critic_learn(embed_state, reward, embed_state_, done)
                agent.actor_learn(embed_state, td_error)

                state = state_
                step += 1

                if done or step >= env.T - 1:
                    break
            all_episode_reward.append(episode_reward)
            print('Training  | Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}' \
                  .format(episode + 1, args.TRAIN_EPISODES, episode_reward, (time.time() - t0)/60))
        agent.save(args.ALG_NAME, ENV_ID)
        plt.plot(all_episode_reward)
        if not os.path.exists('image'):
            os.makedirs('image')
        plt.savefig(os.path.join('image', '_'.join([args.ALG_NAME, ENV_ID])))
        if not os.path.exists('results'):
            os.makedirs('results')
        npy_name = '_'.join([args.ALG_NAME, ENV_ID]) + '.npy'
        save_rewards(os.path.join('results', npy_name), np.array(all_episode_reward))
        print('AC model saved!')
    if args.test:
        agent.load(args.ALG_NAME, ENV_ID)
        for episode in range(args.TEST_EPISODES):
            env.reset()
            episode_reward = 0
            state = env.state
            action_save = []
            state_save = []
            for step in range(env.T):
                embed_state = embedding(state, args.INPUT_DIM, args.OUTPUT_DIM, args.RANDOM_SEED)
                action = agent.get_action(embed_state)
                state_, reward, done, _, _ = env.step(action, step)
                state = state_
                episode_reward += reward
                action_save.append((np.round(action * 1000)).astype(int).tolist())
                state_save.append((np.around(state)).tolist())
                if done:
                    break
            print('Testing  | Episode: {}/{}  | Episode Reward: {:.4f} '.format(
                episode + 1, args.TEST_EPISODES, episode_reward))
            print(action_save)
            print(state_save)
    # 读取奖励数据并绘图
    npy_name = '_'.join([args.ALG_NAME, ENV_ID]) + '.npy'
    if os.path.exists('results/' + npy_name):
        all_episode_reward = load_rewards(os.path.join('results', npy_name))
        plt.plot(all_episode_reward)
        plt.show()
    else:
        print("奖励数据不存在，无法读取！")


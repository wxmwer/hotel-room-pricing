import os
import numpy as np
import matplotlib.pyplot as plt

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
        #print(f"episode reward has been loaded sucessfully from {file_path}，the shape is: {arr.shape}")
        return arr
    except FileNotFoundError:
        print(f"error: file {file_path} not found")
    except Exception as e:
        print(f"error when loading episode reward: {e}")
        return None

def draw_compare_algorithms():
    # draw Figure_compare_algorithms
    npy_name = ['PPO_Hotel_Price_Offseason_without_constraints_5_5.npy',
                'DDPG_Hotel_Price_Offseason_without_constraints_5_5.npy',
                'TD3_Hotel_Price_Offseason.npy',
                'AC_Hotel_Price_Offseason.npy']
    rewards = []
    for i in range(len(npy_name)):
        file = 'results/' + npy_name[i]
        if os.path.exists(file):

            all_episode_reward = load_rewards(file)
            rewards.append(all_episode_reward)
            print(file, 'loaded!')
        else:
            print("error when loading episode reward")

    x = np.arange(rewards[0].shape[0])
    # 遍历数组的每一行，绘制曲线
    plt.rcParams["font.family"] = ["Times New Roman", "SimHei"]  # 英文用 Times，中文用黑体
    plt.figure(figsize=(12, 6))
    plt.plot(x, rewards[0], 'r-', linewidth=1, label=f'PPO')
    plt.plot(x, rewards[1], 'g-', linewidth=1, label=f'DDPG')
    plt.plot(x, rewards[2], 'b-', linewidth=1, label=f'TD3')
    plt.plot(x, rewards[3], 'k-', linewidth=1, label=f'AC')
    # 添加图表元素
    plt.xlabel('episode', fontsize=12)
    plt.ylabel('reward (M CNY)', fontsize=12)
    plt.legend(loc='lower right', prop={"size": 12})
    plt.grid(True, linestyle='--', alpha=0.5)  # 添加网格线
    plt.tight_layout()  # 自动调整布局
    plt.show()
def draw_train_effect_offseason():
    npy_name_off = ['PPO_Hotel_Price_Offseason_without_constraints_5_5.npy',
                'PPO_Hotel_Price_Offseason_with_constraints_5_5.npy',
                'PPO_Hotel_Price_Offseason_with_constraints_5_5_adjusted.npy',
                'PPO_Hotel_Price_Offseason_dynamic_with_constraints_1_5.npy',
                'PPO_Hotel_Price_Offseason_dynamic_with_constraints_1_5_adjusted.npy',
                'PPO_Hotel_Price_Offseason_dynamic_without_constraints_5_5.npy']

    rewards = []
    for i in range(len(npy_name_off)):
        file = 'results/' + npy_name_off[i]
        if os.path.exists(file):
            all_episode_reward = load_rewards(file)
            rewards.append(all_episode_reward)
            print(file, 'loaded!')
        else:
            print("error when loading episode reward")
    rewards_1 = np.concatenate((rewards[0], np.full(500, np.nan, dtype=float)))
    rewards_2 = np.concatenate((rewards[1], rewards[2]))
    rewards_3 = np.concatenate((rewards[3], rewards[4]))
    rewards_4 = np.concatenate((rewards[5], np.full(500, np.nan, dtype=float)))
    x = np.arange(rewards_1.shape[0])
# 遍历数组的每一行，绘制曲线
    plt.rcParams["font.family"] = ["Times New Roman", "SimHei"]  # 英文用 Times，中文用黑体
    plt.figure(figsize=(12, 6))
    plt.plot(x, rewards_1, 'r-', linewidth=1, label=f'Dynamic discrimination pricing neglecting fairness (DDP-N)')
    plt.plot(x, rewards_2, 'g-', linewidth=1, label=f'Dynamic discrimination pricing considering fairness constraints (DDP-C)')
    plt.plot(x, rewards_4, 'y-', linewidth=1, label=f'Dynamic pricing neglecting fairness (DP-N)')
    plt.plot(x, rewards_3, 'b-', linewidth=1, label=f'Dynamic pricing considering fairness constraints (DP-C)')

    # 添加垂直参考线
    plt.axvline(x=2000, color='r', linestyle='--', alpha=0.5, linewidth=1)
    # 添加图表元素
    plt.xlabel('episode', fontsize=12)
    plt.ylabel('reward (M CNY)', fontsize=12)
    plt.legend(loc='lower center', prop={"size": 10})
    plt.grid(True, linestyle='--', alpha=0.5)  # 添加网格线
    plt.tight_layout()  # 自动调整布局
    plt.show()
def draw_train_effect_peakseason():
    npy_name_peak = ['PPO_Hotel_Price_Peakseason_without_constraints_5_5.npy',
                    'PPO_Hotel_Price_Peakseason_with_constraints_5_5.npy',
                    'PPO_Hotel_Price_Peakseason_with_constraints_5_5_adjusted.npy',
                    'PPO_Hotel_Price_Peakseason_dynamic_with_constraints_1_5.npy',
                    'PPO_Hotel_Price_Peakseason_dynamic_with_constraints_1_5_adjusted.npy',
                    'PPO_Hotel_Price_Peakseason_dynamic_without_constraints_5_5.npy']
    rewards = []
    for i in range(len(npy_name_peak)):
        file = 'results/' + npy_name_peak[i]
        if os.path.exists(file):
            all_episode_reward = load_rewards(file)
            rewards.append(all_episode_reward)
            print(file, 'loaded!')
        else:
            print("error when loading episode reward")
    rewards_1 = np.concatenate((rewards[0], np.full(1000, np.nan, dtype=float)))
    rewards_2 = np.concatenate((rewards[1], rewards[2]))
    rewards_3 = np.concatenate((rewards[3], rewards[4]))
    rewards_4 = np.concatenate((rewards[5], np.full(1000, np.nan, dtype=float)))
    x = np.arange(rewards_1.shape[0])
# 遍历数组的每一行，绘制曲线
    plt.rcParams["font.family"] = ["Times New Roman", "SimHei"]  # 英文用 Times，中文用黑体
    plt.figure(figsize=(12, 6))
    plt.plot(x, rewards_1, 'r-', linewidth=1, label=f'Dynamic discrimination pricing neglecting fairness (DDP-N)')
    plt.plot(x, rewards_2, 'g-', linewidth=1, label=f'Dynamic discrimination pricing considering fairness constraints (DDP-C)')
    plt.plot(x, rewards_4, 'y-', linewidth=1, label=f'Dynamic pricing neglecting fairness (DP-N)')
    plt.plot(x, rewards_3, 'b-', linewidth=1, label=f'Dynamic pricing considering fairness constraints (DP-C)')

    # 添加垂直参考线
    plt.axvline(x=4000, color='r', linestyle='--', alpha=0.5, linewidth=1)
    # 添加图表元素
    plt.xlabel('episode', fontsize=12)
    plt.ylabel('reward (M CNY)', fontsize=12)
    plt.legend(loc='lower center', prop={"size": 10})
    plt.grid(True, linestyle='--', alpha=0.5)  # 添加网格线
    plt.tight_layout()  # 自动调整布局
    plt.show()
if __name__ == '__main__':
    #draw_compare_algorithms()
    draw_train_effect_offseason()
    draw_train_effect_peakseason()
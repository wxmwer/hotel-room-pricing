"""
Deep Deterministic Policy Gradient (DDPG)
-----------------------------------------
An algorithm concurrently learns a Q-function and a policy.
It uses off-policy data and the Bellman equation to learn the Q-function,
and uses the Q-function to learn the policy.
Reference
---------
Deterministic Policy Gradient Algorithms, Silver et al. 2014
Continuous Control With Deep Reinforcement Learning, Lillicrap et al. 2016

-----------
Openai Gym Pendulum-v1, continual action space
Prerequisites
-------------
To run
------
python DDPG.py --train/test
"""

import argparse
import os
import random
import time
import h5py
import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras
import tensorlayer as tl
import os
os.environ["FOR_DISABLE_CONSOLE_CTRL_HANDLER"] = "1"  # Windows 专用
# add arguments in command  --train/test
parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=True)
parser.add_argument('--test', dest='test', action='store_true', default=True)
args = parser.parse_args()

#####################  hyper parameters  ####################

ENV_ID = 'Hotel_price'  # environment id
RANDOM_SEED = 2  # random seed, can be either an int number or None
RENDER = False  # render while training

ALG_NAME = 'DDPG'

LR_A = 0.001  # learning rate for actor
LR_C = 0.002  # learning rate for critic
GAMMA = 1  # reward discount
TAU = 0.01  # soft replacement

BATCH_SIZE = 32
VAR = 0.02  # control exploration 标准差 在无约束是使用的0.02效果较好
VAR_DISCOUNT = 0.9995

###############################  DDPG  ####################################

class ReplayBuffer:
    """
    a ring buffer for storing transitions and sampling for training
    :state: (state_dim,)
    :action: (action_dim,)
    :reward: (,), scalar
    :next_state: (state_dim,)
    :done: (,), scalar (0 and 1) or bool (True and False)
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = int((self.position + 1) % self.capacity)  # as a ring buffer

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))  # stack for each element
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)



class DDPG(object):
    """
    DDPG class
    """
    def __init__(self, action_dim, state_dim, action_para_a, action_para_b, replay_buffer):
        self.replay_buffer = replay_buffer
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.action_para_a = action_para_a
        self.action_para_b = action_para_b
        self.var = VAR
        self.var_discount = VAR_DISCOUNT

        W_init = tf.random_normal_initializer(mean=0, stddev=0.3)
        b_init = tf.constant_initializer(0.1)

        def get_actor(input_state_shape, name=''):
            """
            Build actor network
            :param input_state_shape: state
            :param name: name
            :return: act
            """
            input_layer = tl.layers.Input(input_state_shape, name='A_input')
            # layer = tl.layers.Embedding(
            #     vocabulary_size=200,  # 词汇表大小（最大整数索引 + 1）
            #     embedding_size=2,  # 嵌入向量维度
            #     name='embedding'
            # )(input_layer)
            # layer = tl.layers.Flatten(name='flatten')(layer)
            layer = tl.layers.Dense(n_units=64, act=tf.nn.tanh, W_init=W_init, b_init=b_init, name='A_l1')(input_layer)
            layer = tl.layers.Dense(n_units=64, act=tf.nn.tanh, W_init=W_init, b_init=b_init, name='A_l2')(layer)
            layer = tl.layers.Dense(n_units=action_dim, act=tf.nn.tanh, W_init=W_init, b_init=b_init, name='A_a')(layer)
            layer = tl.layers.Lambda(lambda x: action_para_a * x + action_para_b)(layer)
            return tl.models.Model(inputs=input_layer, outputs=layer, name='Actor' + name)

        def get_critic(input_state_shape, input_action_shape, name=''):
            """
            Build critic network
            :param input_state_shape: state
            :param input_action_shape: act
            :param name: name
            :return: Q value Q(s,a)
            """
            state_input = tl.layers.Input(input_state_shape, name='C_s_input')
            action_input = tl.layers.Input(input_action_shape, name='C_a_input')
            layer = tl.layers.Concat(1)([state_input, action_input])
            # layer = tl.layers.Embedding(
            #     vocabulary_size=1500,  # 词汇表大小（最大整数索引 + 1）
            #     embedding_size=2,  # 嵌入向量维度
            #     name='embedding'
            # )(layer)
            # layer = tl.layers.Flatten(name='flatten')(layer)
            #layer = tl.layers.Lambda(fn=lambda x: tf.cast(x, tf.float32), name='C_float')(layer) #int转化为float32
            layer = tl.layers.Dense(n_units=64, act=tf.nn.relu, W_init=W_init, b_init=b_init, name='C_l1')(layer)
            layer = tl.layers.Dense(n_units=64, act=tf.nn.relu, W_init=W_init, b_init=b_init, name='C_l2')(layer)
            layer = tl.layers.Dense(n_units=1, W_init=W_init, b_init=b_init, name='C_out')(layer)
            return tl.models.Model(inputs=[state_input, action_input], outputs=layer, name='Critic' + name)

        self.actor = get_actor([None, state_dim])
        self.critic = get_critic([None, state_dim], [None, action_dim])
        self.actor.train()
        self.critic.train()

        def copy_para(from_model, to_model):
            """
            Copy parameters for soft updating
            :param from_model: latest model
            :param to_model: target model
            :return: None
            """
            for i, j in zip(from_model.trainable_weights, to_model.trainable_weights):
                j.assign(i)

        self.actor_target = get_actor([None, state_dim], name='_target')
        copy_para(self.actor, self.actor_target)
        self.actor_target.eval()

        self.critic_target = get_critic([None, state_dim], [None, action_dim], name='_target')
        copy_para(self.critic, self.critic_target)
        self.critic_target.eval()
        #训练模型时，保持训练参数的移动平均值通常是有益的。 使用平均参数的评估有时会产生比最终训练值明显更好的结果。
        self.ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)  # soft replacement

        self.actor_opt = keras.optimizers.Adam(LR_A)   # tf.optimizers.Adam(LR_A)
        self.critic_opt = keras.optimizers.Adam(LR_C)  #tf.optimizers.Adam(LR_C)

    def ema_update(self):
        """
        Soft updating by exponential smoothing
        :return: None
        """
        paras = self.actor.trainable_weights + self.critic.trainable_weights
        self.ema.apply(paras)
        for i, j in zip(self.actor_target.trainable_weights + self.critic_target.trainable_weights, paras):
            i.assign(self.ema.average(j))

    def get_action(self, state, greedy=False):
        """
        Choose action
        :param s: state
        :param greedy: get action greedy or not
        :return: act
        输出的动作是三个价格，单位是百元
        """
        action = self.actor(np.array([state]))[0]
        if greedy:
            return action
        # add randomness to action selection for exploration
        action = np.random.normal(action, self.var)
        return np.clip(action, self.action_para_b - self.action_para_a, self.action_para_b + self.action_para_a).astype(np.float32)


    def learn(self):
        """
        Update parameters
        :return: None
        """
        self.var *= self.var_discount
        states, actions, rewards, states_, done = self.replay_buffer.sample(BATCH_SIZE)
        rewards = rewards[:, np.newaxis]
        done = done[:, np.newaxis]

        with tf.GradientTape() as tape:
            actions_ = self.actor_target(states_)
            #actions_ = actions_.numpy().astype(np.float64)
            #states_= states_.astype(np.float64)
            #actions_ = tf.cast(actions_, dtype=np.int64)
            q_ = self.critic_target([states_, actions_])
            target = rewards + (1 - done) * GAMMA * q_
            q_pred = self.critic([states, actions])
            td_error =  keras.losses.mean_squared_error(target, q_pred)#tf.losses.mean_squared_error(target, q_pred)
        critic_grads = tape.gradient(td_error, self.critic.trainable_weights)
        self.critic_opt.apply_gradients(zip(critic_grads, self.critic.trainable_weights))

        with tf.GradientTape() as tape:
            actions = self.actor(states)
            #actions = tf.cast(actions, dtype=np.int64)
            q = self.critic([states, actions])
            actor_loss = -tf.reduce_mean(q)  # maximize the q
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_weights)
        self.actor_opt.apply_gradients(zip(actor_grads, self.actor.trainable_weights))
        self.ema_update()


    def save(self, ALG_NAME, ENV_ID):
        """
        save trained weights
        :return: None
        """
        path = os.path.join('model', '_'.join([ALG_NAME, ENV_ID]))
        if not os.path.exists(path):
            os.makedirs(path)
        # tl.files.save_weights_to_hdf5(os.path.join(path, 'actor.hdf5'), self.actor)
        # tl.files.save_weights_to_hdf5(os.path.join(path, 'actor_target.hdf5'), self.actor_target)
        # tl.files.save_weights_to_hdf5(os.path.join(path, 'critic.hdf5'), self.critic)
        # tl.files.save_weights_to_hdf5(os.path.join(path, 'critic_target.hdf5'), self.critic_target)
        self.save_weights_to_hdf5_compatible(os.path.join(path, 'actor.hdf5'), self.actor)
        self.save_weights_to_hdf5_compatible(os.path.join(path, 'actor_target.hdf5'), self.actor_target)
        self.save_weights_to_hdf5_compatible(os.path.join(path, 'critic.hdf5'), self.critic)
        self.save_weights_to_hdf5_compatible(os.path.join(path, 'critic_target.hdf5'), self.critic_target)

    def load(self, ALG_NAME, ENV_ID):
        """
        load trained weights
        :return: None
        """
        path = os.path.join('model', '_'.join([ALG_NAME, ENV_ID]))
        # tl.files.load_hdf5_to_weights_in_order(os.path.join(path, 'actor.hdf5'), self.actor)
        # tl.files.load_hdf5_to_weights_in_order(os.path.join(path, 'actor_target.hdf5'), self.actor_target)
        # tl.files.load_hdf5_to_weights_in_order(os.path.join(path, 'critic.hdf5'), self.critic)
        # tl.files.load_hdf5_to_weights_in_order(os.path.join(path, 'critic_target.hdf5'), self.critic_target)
        self.load_hdf5_to_weights_in_order_compatible(os.path.join(path, 'actor.hdf5'), self.actor)
        self.load_hdf5_to_weights_in_order_compatible(os.path.join(path, 'actor_target.hdf5'), self.actor_target)
        self.load_hdf5_to_weights_in_order_compatible(os.path.join(path, 'critic.hdf5'), self.critic)
        self.load_hdf5_to_weights_in_order_compatible(os.path.join(path, 'critic_target.hdf5'), self.critic_target)

    def save_weights_to_hdf5_compatible(self, filepath, network):
        """兼容 h5py 3.14.0 的权重保存函数"""
        with h5py.File(filepath, 'w') as f:
            # 创建网络组
            network_group = f.create_group('network')

            # 存储每层的权重
            for i, layer in enumerate(network.all_layers):
                layer_name = layer.name if hasattr(layer, 'name') else f'layer_{i}'
                layer_group = network_group.create_group(layer_name)

                # 获取层的权重
                layer_weights = layer.all_weights

                # 存储每个权重
                for j, weight in enumerate(layer_weights):
                    weight_name = weight.name if hasattr(weight, 'name') else f'weight_{j}'

                    # 获取权重值
                    weight_value = weight.numpy()

                    # 处理字符串兼容性
                    if isinstance(weight_value, str):
                        # 在 h5py 3.x 中，字符串直接保存为 Unicode
                        layer_group.create_dataset(weight_name, data=weight_value)
                    elif isinstance(weight_value, bytes):
                        # 将字节串转换为字符串
                        try:
                            str_value = weight_value.decode('utf-8')
                            layer_group.create_dataset(weight_name, data=str_value)
                        except UnicodeDecodeError:
                            # 如果无法解码，以字节形式保存
                            layer_group.create_dataset(weight_name, data=weight_value)
                    elif isinstance(weight_value, np.ndarray) and weight_value.dtype == object:
                        # 处理对象类型的数组（可能包含字符串）
                        processed_array = []
                        for item in weight_value:
                            if isinstance(item, bytes):
                                try:
                                    processed_array.append(item.decode('utf-8'))
                                except UnicodeDecodeError:
                                    processed_array.append(item)
                            else:
                                processed_array.append(item)
                        layer_group.create_dataset(weight_name, data=np.array(processed_array, dtype=object))
                    else:
                        # 其他类型直接保存
                        layer_group.create_dataset(weight_name, data=weight_value)

    def load_hdf5_to_weights_in_order_compatible(self, filepath, network):
        """兼容 h5py 3.14.0 的权重加载函数"""
        with h5py.File(filepath, 'r') as f:
            # 获取网络组
            network_group = f['network']

            # 遍历每层
            for i, layer in enumerate(network.all_layers):
                layer_name = layer.name if hasattr(layer, 'name') else f'layer_{i}'

                if layer_name in network_group:
                    layer_group = network_group[layer_name]

                    # 获取层的权重
                    layer_weights = layer.all_weights

                    # 加载每个权重
                    for j, weight in enumerate(layer_weights):
                        weight_name = weight.name if hasattr(weight, 'name') else f'weight_{j}'

                        if weight_name in layer_group:
                            # 获取数据集
                            dataset = layer_group[weight_name]

                            # 获取权重值
                            if dataset.dtype == 'object':
                                # 处理可能的字符串或字节数据
                                data = dataset[()]
                                if isinstance(data, bytes):
                                    try:
                                        data = data.decode('utf-8')
                                    except UnicodeDecodeError:
                                        pass
                                # 如果期望的是数值权重但得到字符串，跳过
                                if isinstance(data, str) and weight.dtype != tf.string:
                                    continue
                            else:
                                data = dataset[()]

                            # 应用权重
                            try:
                                weight.assign(data)
                            except ValueError as e:
                                print(f"无法加载权重 {weight_name}: {e}")
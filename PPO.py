"""
To run
------
python PPO1.py --train/test

核心思想：PPO2的核心思想很简单，对于ratio 也就是当前policy和旧policy的偏差做clip，如果ratio偏差超过一定的范围就做clip，
clip后梯度也限制在一定的范围内，神经网络更新参数也不会太离谱。这样，在实现上，无论更新多少步都没有关系，有clip给我们挡着，不担心训练偏了。
"""
import argparse
import os
import time
import keras
import gym
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
import h5py
import tensorlayer as tl

parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=True)
parser.add_argument('--test', dest='test', action='store_false', default=True)
args = parser.parse_args()

#####################  hyper parameters  ####################

ENV_NAME = 'Hotel_price'  # environment name
ALG_NAME = 'PPO'
RANDOMSEED = 1  # random seed

EP_MAX = 1000  # total number of episodes for training
EP_LEN = 200  # total number of steps for each episode
GAMMA = 1  # reward discount
A_LR = 0.0001  # learning rate for actor
C_LR = 0.0002  # learning rate for critic
BATCH = 32  # update batchsize
A_UPDATE_STEPS = 10  # actor update steps
C_UPDATE_STEPS = 10  # critic update steps
EPS = 1e-8  # epsilon

# 注意：这里是PPO1和PPO2的相关的参数。
METHOD = [
    dict(name='kl_pen', kl_target=0.01, lam=0.5),  # KL penalty  PPO1
    dict(name='clip', epsilon=0.2),  # Clipped surrogate objective, find this is better  PPO2
][1]  # choose the method for optimization

# ppo-penalty parameters
KL_TARGET = 0.01
LAM = 0.5

# ppo-clip parameters
EPSILON = 0.2


###############################  PPO  ####################################


class PPO(object):
    '''
    PPO 类
    '''

    def __init__(self, state_dim, action_dim, action_para_a, action_para_b, method='clip'):

        self.action_para_a = action_para_a
        self.action_para_b = action_para_b
        W_init = tf.random_normal_initializer(mean=0, stddev=0.3)
        b_init = tf.constant_initializer(0.1)

        def build_critic(input_state_dim):
            input_layer = tl.layers.Input(input_state_dim, tf.float32)
            #layer = tl.layers.Dense(100, tf.nn.relu)(input_layer)
            layer = tl.layers.Dense(n_units=64, act=tf.nn.relu, W_init=W_init, b_init=b_init)(input_layer)
            layer = tl.layers.Dense(n_units=64, act=tf.nn.relu, W_init=W_init, b_init=b_init)(layer)
            output_layer = tl.layers.Dense(n_units=1, W_init=W_init, b_init=b_init)(layer)
            return tl.models.Model(input_layer, output_layer)

        def build_actor(input_state_dim, action_dim):
            ''' actor 网络，输出mu和sigma '''
            input_layer = tl.layers.Input(input_state_dim, tf.float32)
            #l1 = tl.layers.Dense(100, tf.nn.relu)(input_layer)
            layer = tl.layers.Dense(n_units=64, act=tf.nn.tanh, W_init=W_init, b_init=b_init)(input_layer)
            l1 = tl.layers.Dense(n_units=64, act=tf.nn.tanh, W_init=W_init, b_init=b_init)(layer)
            a = tl.layers.Dense(action_dim, tf.nn.tanh, W_init=W_init, b_init=b_init)(l1)
            mu = tl.layers.Lambda(lambda x: action_para_a * x + action_para_b)(a)
            #sigma = tl.layers.Dense(action_dim, act=tf.nn.softplus, W_init=W_init, b_init=b_init)(l1)  #tf.nn.softplus ln(1+e^x)
            #sigma = tl.layers.Lambda(lambda x: x/10.0)(sigma)    lambda x: clip_activation(x, 0, 0.02)
            model = tl.models.Model(input_layer, mu)
            return model

        # 构建critic网络, 输入state，输出V值
        self.critic = build_critic([None, state_dim])
        self.critic.train()

        # actor有两个: actor 和 actor_old， actor_old的主要功能是记录行为策略的版本。
        # 输入是state，输出是描述动作分布的mu和sigma
        self.actor = build_actor([None, state_dim], action_dim)
        self.actor_old = build_actor([None, state_dim], action_dim)
        # self.train()与self.eval()
        # 如果模型中有BN层(Batch Normalization）和Dropout，需要在训练时添加model.train()，在测试时添加model.eval()。
        # 其中model.train()是保证BN层用每一批数据的均值和方差，而model.eval()是保证BN用全部训练数据的均值和方差；
        # 而对于Dropout，model.train()是随机取一部分网络连接来训练更新参数，而model.eval()是利用到了所有网络连接。
        self.actor.train()
        self.actor_old.eval()

        self.actor_opt = keras.optimizers.Adam(A_LR)
        self.critic_opt = keras.optimizers.Adam(C_LR)

        self.method = method
        if method == 'penalty':
            self.kl_target = KL_TARGET
            self.lam = LAM
        elif method == 'clip':
            self.epsilon = EPSILON

        self.state_buffer, self.action_buffer = [], []
        self.reward_buffer, self.cumulative_reward_buffer = [], []
        #self.action_bound = action_bound
        self.var = 0.1
        self.var_discount = 0.9999
        self.var_old = 0.1

    def choose_action(self, s, greedy = False):
        '''
        Choose action
        :param s: state
        :return: clipped act
        '''
        s = s[np.newaxis, :].astype(np.float32)
        mu = self.actor(s)
        if greedy:
            return mu[0].numpy()
        else:
            pi = tfp.distributions.Normal(mu, self.var)  # 用mu和sigma构建正态分布
            a = tf.squeeze(pi.sample(1), axis=0)[0]  # 根据概率分布随机出动作
            a = np.clip(a, self.action_para_b - self.action_para_a, self.action_para_b + self.action_para_a)
            return a

    def store_transition(self, state, action, reward):
        """
        Store state, action, reward at each step
        """
        self.state_buffer.append(state)
        self.action_buffer.append(action)
        self.reward_buffer.append(reward)

    def a_train(self, state, action, adv):
        """ 更新策略网络(policy network) """
        self.var *= self.var_discount
        state = np.array(state, np.float32)
        action = np.array(action, np.float32)
        adv = np.array(adv, np.float32)

        with tf.GradientTape() as tape:
            # 构建两个正态分布pi，oldpi。
            mu = self.actor(state)
            pi = tfp.distributions.Normal(mu, self.var)

            mu_old = self.actor_old(state)
            oldpi = tfp.distributions.Normal(mu_old, self.var_old)

            # ratio = tf.exp(pi.log_prob(self.tfa) - oldpi.log_prob(self.tfa))
            # 在新旧两个分布下，同样输出a的概率的比值
            # 除以(oldpi.prob(tfa) + EPS)，其实就是做了import-sampling。怎么解释这里好呢
            # 本来我们是可以直接用pi.prob(tfa)去更新的，但为了能够更新多次，我们需要除以(oldpi.prob(tfa) + EPS)。
            # 在AC或者PG，我们是以1,0作为更新目标，缩小动作概率到1 or 0的差距
            # 而PPO可以想作是，以oldpi.prob(tfa)出发，不断远离（增大or缩小）的过程。
            ratio = pi.prob(action) / (oldpi.prob(action) + EPS)
            # 这个的意义和带参数更新是一样的。
            surr = ratio * adv

            # 我们还不能让两个分布差异太大。
            # PPO1
            if METHOD['name'] == 'kl_pen':
                tflam = METHOD['lam']
                kl = tfp.distributions.kl_divergence(oldpi, pi)
                kl_mean = tf.reduce_mean(kl)
                aloss = -(tf.reduce_mean(surr - tflam * kl))
            else:  # clipping method, find this is better
                aloss = -tf.reduce_mean(
                    tf.minimum(surr, tf.clip_by_value(
                        ratio, 1. - METHOD['epsilon'], 1. + METHOD['epsilon']) * adv)
                )
        a_gard = tape.gradient(aloss, self.actor.trainable_weights)
        self.actor_opt.apply_gradients(zip(a_gard, self.actor.trainable_weights))

        if METHOD['name'] == 'kl_pen':
            return kl_mean

    def update_old_pi(self):
        '''
        更新actor_old参数。
        '''
        # zip例子：a = [1,2,3]  b = [4,5,6] zipped = zip(a,b)  结果为[(1, 4), (2, 5), (3, 6)]
        for pi, oldpi in zip(self.actor.trainable_weights, self.actor_old.trainable_weights):
            oldpi.assign(pi)
        self.var_old = self.var

    def c_train(self, reward, state):
        ''' 更新Critic网络 '''
        # reward 是我们预估的 能获得的奖励，是累计奖励，即r+γV(st+1)
        reward = np.array(reward, dtype=np.float32)
        with tf.GradientTape() as tape:
            advantage = reward - self.critic(state)  # td-error
            loss = tf.reduce_mean(tf.square(advantage))
        grad = tape.gradient(loss, self.critic.trainable_weights)
        self.critic_opt.apply_gradients(zip(grad, self.critic.trainable_weights))

    def update(self):
        '''
        Update parameter with the constraint of KL divergent
        '''
        s = np.array(self.state_buffer, np.float32)
        a = np.array(self.action_buffer, np.float32)
        r = np.array(self.cumulative_reward_buffer, np.float32)

        self.update_old_pi()
        adv = (r - self.critic(s)).numpy()
        # adv = (adv - adv.mean())/(adv.std()+1e-6)  # sometimes helpful

        # update actor
        if METHOD['name'] == 'kl_pen':
            for _ in range(A_UPDATE_STEPS):
                kl = self.a_train(s, a, adv)
                if kl > 4 * METHOD['kl_target']:  # this in in google's paper
                    break
            if kl < METHOD['kl_target'] / 1.5:  # adaptive lambda, this is in OpenAI's paper
                METHOD['lam'] /= 2
            elif kl > METHOD['kl_target'] * 1.5:
                METHOD['lam'] *= 2
            METHOD['lam'] = np.clip(
                METHOD['lam'], 1e-4, 10
            )  # sometimes explode, this clipping is MorvanZhou's solution
        # PPO2 clipping method, find this is better (OpenAI's paper)
        else:
            for _ in range(A_UPDATE_STEPS):
                self.a_train(s, a, adv)

        for _ in range(C_UPDATE_STEPS):
            self.c_train(r, s)

        self.state_buffer.clear()
        self.action_buffer.clear()
        self.cumulative_reward_buffer.clear()
        self.reward_buffer.clear()

    def finish_path(self, next_state, end_period, done):
        """
        Calculate cumulative reward
        :param next_state:
        :return: None
        """
        if done:
            v_s_ = 0
        if end_period:
            v_s_ = 0
        else:
            v_s_ = self.critic(np.array([next_state], dtype=np.float32))[0, 0]
        discounted_r = []
        for r in self.reward_buffer[::-1]:  # [::-1]表示将字符或数字倒序输出
            v_s_ = r + GAMMA * v_s_
            discounted_r.append(v_s_)
        discounted_r.reverse()
        discounted_r = np.array(discounted_r)[:, np.newaxis]
        self.cumulative_reward_buffer.extend(discounted_r)
        self.reward_buffer.clear()

    def save_ckpt(self, ALG_NAME, ENV_NAME):
        path = os.path.join('model', '_'.join([ALG_NAME, ENV_NAME]))
        if not os.path.exists(path):
            os.makedirs(path)
        self.save_weights_to_hdf5_compatible(os.path.join(path, 'actor.hdf5'), self.actor)
        self.save_weights_to_hdf5_compatible(os.path.join(path, 'actor_old.hdf5'), self.actor_old)
        self.save_weights_to_hdf5_compatible(os.path.join(path, 'critic.hdf5'), self.critic)
        print('save weights success!')

    def load_ckpt(self, ALG_NAME, ENV_NAME):
        path = os.path.join('model', '_'.join([ALG_NAME, ENV_NAME]))
        self.load_hdf5_to_weights_in_order_compatible(os.path.join(path, 'actor.hdf5'), self.actor)
        self.load_hdf5_to_weights_in_order_compatible(os.path.join(path, 'actor_old.hdf5'), self.actor_old)
        self.load_hdf5_to_weights_in_order_compatible(os.path.join(path, 'critic.hdf5'), self.critic)
        print("load weight!")
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


"""
Actor-Critic
-------------
It uses TD-error as the Advantage.
To run
------
python AC_Continuous.py --train/test
"""
import h5py
import time
import matplotlib.pyplot as plt
import os
import keras
import numpy as np
import tensorflow as tf
import tensorlayer as tl
import tensorflow_probability as tfp

#####################  hyper parameters  ####################
#
# ENV_ID = 'Hotel_price'  # environment id
#
# ALG_NAME = 'AC'
# TRAIN_EPISODES = 500  # number of overall episodes for training
# TEST_EPISODES = 10  # number of overall episodes for testing
# MAX_STEPS = 200  # maximum time step in one episode
LAM = 1  # reward discount in TD error
LR_A = 0.0001  # learning rate for actor
LR_C = 0.0002  # learning rate for critic

class AC(object):
    """

    """
    def __init__(self, state_dim, action_dim, action_para_a, action_para_b):
        self.action_para_a = action_para_a
        self.action_para_b = action_para_b
        W_init = tf.random_normal_initializer(mean=0, stddev=0.3)
        b_init = tf.constant_initializer(0.1)

        def build_actor(input_state_dim, action_dim):
            # input_layer = tl.layers.Input([None, state_dim])
            input_layer = tl.layers.Input(input_state_dim, tf.float32)
            layer = tl.layers.Dense(n_units=64, act=tf.nn.tanh, W_init=W_init, b_init=b_init)(input_layer)
            layer = tl.layers.Dense(n_units=64, act=tf.nn.tanh, W_init=W_init, b_init=b_init)(layer)
            act = tl.layers.Dense(n_units=action_dim, act=tf.nn.tanh, W_init=W_init, b_init=b_init)(layer)
            mu = tl.layers.Lambda(lambda x: action_para_a * x + action_para_b)(act)
            sigma = tl.layers.Dense(action_dim, act=tf.nn.softplus, W_init=W_init, b_init=b_init)(layer)
            return tl.models.Model(inputs=input_layer, outputs=[mu, sigma])

        def build_critic(input_state_dim):
            input_layer = tl.layers.Input(input_state_dim, tf.float32)
            # input_layer = tl.layers.Input([None, state_dim], name='state')
            layer = tl.layers.Dense(n_units=64, act=tf.nn.tanh, W_init=W_init, b_init=b_init)(input_layer)
            layer = tl.layers.Dense(n_units=64, act=tf.nn.tanh, W_init=W_init, b_init=b_init)(layer)
            output_layer = tl.layers.Dense(n_units=1, W_init=W_init, b_init=b_init)(layer)
            return tl.models.Model(inputs=input_layer, outputs=output_layer)

        self.critic = build_critic([None, state_dim])
        self.critic.train()
        self.actor = build_actor([None, state_dim], action_dim)
        self.actor.train()

        self.actor_opt = keras.optimizers.Adam(LR_A)
        self.critic_opt = keras.optimizers.Adam(LR_C)

    def actor_learn(self, state, td_error):
        with tf.GradientTape() as tape:
            mu, sigma = self.actor(np.array([state], dtype=np.float32))
            pi = tfp.distributions.Normal(mu, sigma)
            action = np.clip(pi.sample(), self.action_para_b - self.action_para_a, self.action_para_b + self.action_para_a)
            log_prob = pi.log_prob(action)
            loss = - log_prob * td_error
        grad = tape.gradient(loss, self.actor.trainable_weights)
        self.actor_opt.apply_gradients(zip(grad, self.actor.trainable_weights))

    def get_action(self, state, greedy=False):
        mu, sigma = self.actor(np.array([state]))
        if greedy:
            action = mu[0]
        else:
            pi = tfp.distributions.Normal(mu, sigma)
            action = tf.squeeze(pi.sample(1), axis=0)[0]
        return np.clip(action, self.action_para_b - self.action_para_a, self.action_para_b + self.action_para_a)
    def critic_learn(self, state, reward, state_, done):
        d = 0 if done else 1
        with tf.GradientTape() as tape:
            v = self.critic(np.array([state]))
            v_ = self.critic(np.array([state_]))
            td_error = reward + d * LAM * v_ - v
            loss = tf.square(td_error)
        grads = tape.gradient(loss, self.critic.trainable_weights)
        self.critic_opt.apply_gradients(zip(grads, self.critic.trainable_weights))
        return td_error


    def save(self, ALG_NAME, ENV_ID):
        path = os.path.join('model', '_'.join([ALG_NAME, ENV_ID]))
        if not os.path.exists(path):
            os.makedirs(path)
        self.save_weights_to_hdf5_compatible(os.path.join(path, 'actor.hdf5'), self.actor)
        self.save_weights_to_hdf5_compatible(os.path.join(path, 'critic.hdf5'), self.critic)
        print('Succeed to save model')

    def load(self, ALG_NAME, ENV_ID):
        path = os.path.join('model', '_'.join([ALG_NAME, ENV_ID]))
        self.load_hdf5_to_weights_in_order_compatible(os.path.join(path, 'actor.hdf5'), self.actor)
        self.load_hdf5_to_weights_in_order_compatible(os.path.join(path, 'critic.hdf5'), self.critic)
        print('Succeed to load model')

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

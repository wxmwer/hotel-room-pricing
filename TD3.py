"""
Twin Delayed DDPG (TD3)
------------------------
DDPG suffers from problems like overestimate of Q-values and sensitivity to hyper-parameters.
Twin Delayed DDPG (TD3) is a variant of DDPG with several tricks:
* Trick One: Clipped Double-Q Learning. TD3 learns two Q-functions instead of one (hence "twin"),
and uses the smaller of the two Q-values to form the targets in the Bellman error loss functions.
* Trick Two: "Delayed" Policy Updates. TD3 updates the policy (and target networks) less frequently
than the Q-function.
* Trick Three: Target Policy Smoothing. TD3 adds noise to the target action, to make it harder for
the policy to exploit Q-function errors by smoothing out Q along changes in action.
The implementation of TD3 includes 6 networks: 2 Q-net, 2 target Q-net, 1 policy net, 1 target policy net
Actor policy in TD3 is deterministic, with Gaussian exploration noise.
Reference
---------
original paper: https://arxiv.org/pdf/1802.09477.pdf
Environment
---
Openai Gym Pendulum-v1, continuous action space
https://www.gymlibrary.dev/environments/classic_control/pendulum/
Prerequisites
---
tensorflow >=2.0.0a0
tensorflow-probability 0.6.0
tensorlayer >=2.0.0
&&
pip install box2d box2d-kengz --user
To run
-------
python TD3.py --train/test
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
import tensorflow_probability as tfp
import tensorlayer as tl
from tensorlayer.layers import Dense
from tensorlayer.models import Model

Normal = tfp.distributions.Normal
tl.logging.set_verbosity(tl.logging.DEBUG)



#####################  hyper parameters  ####################


# RL training

ENV_ID = 'Pendulum-v1'  # environment id
RANDOM_SEED = 2  # random seed
RENDER = False  # render while training

# RL training



###############################  TD3  ####################################


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


class QNetwork(Model):
    """ the network for evaluate values of state-action pairs: Q(s,a) """

    def __init__(self, num_inputs, num_actions, hidden_dim, init_w=3e-3):
        super(QNetwork, self).__init__()
        input_dim = num_inputs + num_actions
        w_init = tf.random_normal_initializer(mean=0, stddev=0.3)
        b_init = tf.constant_initializer(0.1)

        self.linear1 = Dense(n_units=hidden_dim, act=tf.nn.relu, W_init=w_init, b_init=b_init, in_channels=input_dim, name='q1')
        self.linear2 = Dense(n_units=hidden_dim, act=tf.nn.relu, W_init=w_init, b_init=b_init, in_channels=hidden_dim, name='q2')
        self.linear3 = Dense(n_units=1, W_init=w_init, b_init=b_init, in_channels=hidden_dim, name='q3')

    def forward(self, input):
        x = self.linear1(input)
        x = self.linear2(x)
        x = self.linear3(x)
        return x


class PolicyNetwork(Model):
    """ the network for generating non-determinstic (Gaussian distributed) action from the state input """

    def __init__(self, num_inputs, num_actions, hidden_dim, action_para_a, action_para_b, init_w=3e-3):
        super(PolicyNetwork, self).__init__()
        w_init = tf.random_normal_initializer(mean=0, stddev=0.3)
        b_init = tf.constant_initializer(0.1)

        self.linear1 = Dense(n_units=hidden_dim, act=tf.nn.tanh, W_init=w_init, b_init=b_init, in_channels=num_inputs, name='policy1')
        self.linear2 = Dense(n_units=hidden_dim, act=tf.nn.tanh, W_init=w_init, b_init=b_init, in_channels=hidden_dim, name='policy2')
        #self.linear3 = Dense(n_units=hidden_dim, act=tf.nn.relu, W_init=w_init, in_channels=hidden_dim, name='policy3')
        self.output_linear = Dense(
            n_units=num_actions, act=tf.nn.tanh, W_init=w_init, b_init=b_init,
            in_channels=hidden_dim, name='policy_output'
        )
        self.action_para_a = action_para_a
        self.action_para_b = action_para_b
        self.num_actions = num_actions

        self.var = 0.12
        self.var_discount = 0.99996

    def forward(self, state):
        x = self.linear1(state)
        x = self.linear2(x)
        #x = self.linear3(x)
        output = self.output_linear(x)  # unit range output [-1, 1]
        return output

    def evaluate(self, state, greedy = False):
        """
        generate action with state for calculating gradients;
        eval_noise_scale: as the trick of target policy smoothing, for generating noisy actions.
        """
        state = state.astype(np.float32)
        action = self.forward(state)
        action = self.action_para_a * action + self.action_para_b
        if greedy == False:
            pi = tfp.distributions.Normal(action, self.var)  # 用mu和sigma构建正态分布
            a = tf.squeeze(pi.sample(1), axis=0)  # 根据概率分布随机出动作
            action = np.clip(a, self.action_para_b - self.action_para_a, self.action_para_b + self.action_para_a)
            action = tf.convert_to_tensor(action)
        # add noise
        # normal = Normal(0, 1)
        # noise = normal.sample(action.shape) * eval_noise_scale
        # eval_noise_clip = 2 * eval_noise_scale
        # noise = tf.clip_by_value(noise, -eval_noise_clip, eval_noise_clip)
        # action = action + noise

        return action

    def get_action(self, state, greedy=False):
        """ generate action with state for interaction with environment """
        action = self.forward([state])
        action = self.action_para_a * action.numpy()[0] + self.action_para_b
        if greedy:
            return action
        # add noise
        # normal = Normal(0, 1)
        # noise = normal.sample(action.shape) * explore_noise_scale
        # action += noise
        # return action.numpy()
        pi = tfp.distributions.Normal(action, self.var)  # 用mu和sigma构建正态分布
        a = tf.squeeze(pi.sample(1), axis=0) # 根据概率分布随机出动作
        action = np.clip(a, self.action_para_b - self.action_para_a, self.action_para_b + self.action_para_a)
        return action

    def sample_action(self):
        """ generate random actions for exploration """
        a = tf.random.uniform([self.num_actions], -1, 1)
        return self.action_para_a * a.numpy() + self.action_para_b


class TD3:

    def __init__(
            self, state_dim, action_dim, hidden_dim, action_para_a, action_para_b, replay_buffer, policy_target_update_interval=1,
            q_lr=3e-4, policy_lr=3e-4
    ):
        self.replay_buffer = replay_buffer

        # initialize all networks
        self.q_net1 = QNetwork(state_dim, action_dim, hidden_dim)
        self.q_net2 = QNetwork(state_dim, action_dim, hidden_dim)
        self.target_q_net1 = QNetwork(state_dim, action_dim, hidden_dim)
        self.target_q_net2 = QNetwork(state_dim, action_dim, hidden_dim)
        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim, action_para_a, action_para_b)
        self.target_policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim, action_para_a, action_para_b)
        #print('Q Network (1,2): ', self.q_net1)
        #print('Policy Network: ', self.policy_net)

        # initialize weights of target networks
        self.target_q_net1 = self.target_ini(self.q_net1, self.target_q_net1)
        self.target_q_net2 = self.target_ini(self.q_net2, self.target_q_net2)
        self.target_policy_net = self.target_ini(self.policy_net, self.target_policy_net)

        # set train mode
        self.q_net1.train()
        self.q_net2.train()
        self.target_q_net1.eval()
        self.target_q_net2.eval()
        self.policy_net.train()
        self.target_policy_net.eval()

        self.update_cnt = 0
        self.policy_target_update_interval = policy_target_update_interval

        self.q_optimizer1 = keras.optimizers.Adam(q_lr)
        self.q_optimizer2 = keras.optimizers.Adam(q_lr)
        self.policy_optimizer = keras.optimizers.Adam(policy_lr)

    def target_ini(self, net, target_net):
        """ hard-copy update for initializing target networks """
        for target_param, param in zip(target_net.trainable_weights, net.trainable_weights):
            target_param.assign(param)
        return target_net

    def target_soft_update(self, net, target_net, soft_tau):
        """ soft update the target net with Polyak averaging """
        for target_param, param in zip(target_net.trainable_weights, net.trainable_weights):
            target_param.assign(  # copy weight value into target parameters
                target_param * (1.0 - soft_tau) + param * soft_tau
            )
        return target_net

    def update(self, batch_size, reward_scale=10., gamma=1, soft_tau=1e-2):
        """ update all networks in TD3 """
        self.policy_net.var *= self.policy_net.var_discount

        self.update_cnt += 1
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)

        reward = reward[:, np.newaxis]  # expand dim
        done = done[:, np.newaxis]

        new_next_action = self.target_policy_net.evaluate(next_state)  # clipped normal noise
        reward = reward_scale * (reward - np.mean(reward, axis=0)) / (
            np.std(reward, axis=0) + 1e-6
        )  # normalize with batch mean and std; plus a small number to prevent numerical problem

        # Training Q Function
        target_q_input = tf.concat([next_state, new_next_action], 1)  # the dim 0 is number of samples
        target_q_min = tf.minimum(self.target_q_net1(target_q_input), self.target_q_net2(target_q_input))

        target_q_value = reward + (1 - done) * gamma * target_q_min  # if done==1, only reward
        q_input = tf.concat([state, action], 1)  # input of q_net

        with tf.GradientTape() as q1_tape:
            predicted_q_value1 = self.q_net1(q_input)
            q_value_loss1 = tf.reduce_mean(tf.square(predicted_q_value1 - target_q_value))
        q1_grad = q1_tape.gradient(q_value_loss1, self.q_net1.trainable_weights)
        self.q_optimizer1.apply_gradients(zip(q1_grad, self.q_net1.trainable_weights))

        with tf.GradientTape() as q2_tape:
            predicted_q_value2 = self.q_net2(q_input)
            q_value_loss2 = tf.reduce_mean(tf.square(predicted_q_value2 - target_q_value))
        q2_grad = q2_tape.gradient(q_value_loss2, self.q_net2.trainable_weights)
        self.q_optimizer2.apply_gradients(zip(q2_grad, self.q_net2.trainable_weights))

        # Training Policy Function
        if self.update_cnt % self.policy_target_update_interval == 0:
            with tf.GradientTape() as p_tape:
                # 更新actor的时候，我们不需要加上noise，这里是希望actor能够寻着最大值。加上noise并没有任何意义
                new_action = self.policy_net.evaluate(state, greedy=True)  # no noise, deterministic policy gradients
                new_q_input = tf.concat([state, new_action], 1)
                # """ implementation 1 """
                # predicted_new_q_value = tf.minimum(self.q_net1(new_q_input),self.q_net2(new_q_input))
                """ implementation 2 """
                predicted_new_q_value = self.q_net1(new_q_input)
                policy_loss = -tf.reduce_mean(predicted_new_q_value)
            p_grad = p_tape.gradient(policy_loss, self.policy_net.trainable_weights)
            self.policy_optimizer.apply_gradients(zip(p_grad, self.policy_net.trainable_weights))

            # Soft update the target nets
            self.target_q_net1 = self.target_soft_update(self.q_net1, self.target_q_net1, soft_tau)
            self.target_q_net2 = self.target_soft_update(self.q_net2, self.target_q_net2, soft_tau)
            self.target_policy_net = self.target_soft_update(self.policy_net, self.target_policy_net, soft_tau)


    def save(self, ALG_NAME, ENV_NAME):
        path = os.path.join('model', '_'.join([ALG_NAME, ENV_NAME]))
        if not os.path.exists(path):
            os.makedirs(path)
        self.save_weights_to_hdf5_compatible(os.path.join(path, 'model_q_net1.hdf5'), self.q_net1)
        self.save_weights_to_hdf5_compatible(os.path.join(path, 'model_q_net2.hdf5'), self.q_net2)
        self.save_weights_to_hdf5_compatible(os.path.join(path, 'model_target_q_net1.hdf5'), self.target_q_net1)
        self.save_weights_to_hdf5_compatible(os.path.join(path, 'model_target_q_net2.hdf5'), self.target_q_net2)
        self.save_weights_to_hdf5_compatible(os.path.join(path, 'model_policy_net.hdf5'), self.policy_net)
        self.save_weights_to_hdf5_compatible(os.path.join(path, 'model_target_policy_net.hdf5'), self.target_policy_net)
        print('save TD3 weights success!')

    def load(self, ALG_NAME, ENV_NAME):
        path = os.path.join('model', '_'.join([ALG_NAME, ENV_NAME]))
        self.load_hdf5_to_weights_in_order_compatible(os.path.join(path, 'model_q_net1.hdf5'), self.q_net1)
        self.load_hdf5_to_weights_in_order_compatible(os.path.join(path, 'model_q_net2.hdf5'), self.q_net2)
        self.load_hdf5_to_weights_in_order_compatible(os.path.join(path, 'model_target_q_net1.hdf5'), self.target_q_net1)
        self.load_hdf5_to_weights_in_order_compatible(os.path.join(path, 'model_target_q_net2.hdf5'), self.target_q_net2)
        self.load_hdf5_to_weights_in_order_compatible(os.path.join(path, 'model_policy_net.hdf5'), self.policy_net)
        self.load_hdf5_to_weights_in_order_compatible(os.path.join(path, 'model_target_policy_net.hdf5'), self.target_policy_net)
        print("load TD3 weights success!")
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


# if __name__ == '__main__':
#     # initialization of env
#     env = gym.make(ENV_ID).unwrapped
#     state_dim = env.observation_space.shape[0]
#     action_dim = env.action_space.shape[0]
#     action_range = env.action_space.high  # scale action, [-action_range, action_range]
#
#     # reproducible
#     #env.seed(RANDOM_SEED)
#     env.reset(seed=RANDOM_SEED)
#     random.seed(RANDOM_SEED)
#     np.random.seed(RANDOM_SEED)
#     tf.random.set_seed(RANDOM_SEED)
#
#     # initialization of buffer
#     replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
#     # initialization of trainer
#     agent = TD3(
#         state_dim, action_dim, action_range, HIDDEN_DIM, replay_buffer, POLICY_TARGET_UPDATE_INTERVAL, Q_LR, POLICY_LR
#     )
#     t0 = time.time()
#
#     # training loop
#     if args.train:
#         frame_idx = 0
#         all_episode_reward = []
#
#         # need an extra call here to make inside functions be able to use model.forward
#         state = env.reset()[0].astype(np.float32)
#         agent.policy_net([state])
#         agent.target_policy_net([state])
#
#         for episode in range(TRAIN_EPISODES):
#             state = env.reset()[0].astype(np.float32)
#             episode_reward = 0
#
#             for step in range(MAX_STEPS):
#                 if RENDER:
#                     env.render()
#                 if frame_idx > EXPLORE_STEPS:
#                     action = agent.policy_net.get_action(state, EXPLORE_NOISE_SCALE)
#                 else:
#                     action = agent.policy_net.sample_action()
#
#                 next_state, reward, done, _, _ = env.step(action)
#                 next_state = next_state.astype(np.float32)
#                 done = 1 if done is True else 0
#
#                 replay_buffer.push(state, action, reward, next_state, done)
#                 state = next_state
#                 episode_reward += reward
#                 frame_idx += 1
#
#                 if len(replay_buffer) > BATCH_SIZE:
#                     for i in range(UPDATE_ITR):
#                         agent.update(BATCH_SIZE, EVAL_NOISE_SCALE, REWARD_SCALE)
#                 if done:
#                     break
#             if episode == 0:
#                 all_episode_reward.append(episode_reward)
#             else:
#                 all_episode_reward.append(all_episode_reward[-1] * 0.9 + episode_reward * 0.1)
#             print(
#                 'Training  | Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
#                     episode + 1, TRAIN_EPISODES, episode_reward,
#                     time.time() - t0
#                 )
#             )
#         agent.save()
#         plt.plot(all_episode_reward)
#         if not os.path.exists('image'):
#             os.makedirs('image')
#         plt.savefig(os.path.join('image', '_'.join([ALG_NAME, ENV_ID])))
#
#     if args.test:
#         agent.load()
#
#         # need an extra call here to make inside functions be able to use model.forward
#         state = env.reset()[0].astype(np.float32)
#         agent.policy_net([state])
#
#         for episode in range(TEST_EPISODES):
#             state = env.reset()[0].astype(np.float32)
#             episode_reward = 0
#             for step in range(MAX_STEPS):
#                 env.render()
#                 action = agent.policy_net.get_action(state, EXPLORE_NOISE_SCALE, greedy=True)
#                 state, reward, done, info, _ = env.step(action)
#                 state = state.astype(np.float32)
#                 episode_reward += reward
#                 if done:
#                     break
#             print(
#                 'Testing  | Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
#                     episode + 1, TEST_EPISODES, episode_reward,
#                     time.time() - t0
#                 )
#             )
#         env.close()
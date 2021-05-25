"""
This part of code is the DQN brain, which is a brain of the agent.
All decisions are made in here.
Using Tensorflow to build the neural network.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.7.3
"""

import numpy as np
import pandas as pd
import tensorflow as tf

np.random.seed(1)
tf.set_random_seed(1)


# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=1,
            replace_target_iter=300,
            memory_size=5000,
            batch_size=1000,
            sess=None,
            e_greedy_increment=None,
            output_graph=False,
            e_greedy_begin = 0.6,
            e_greedy_end = 1,
    ):
        self.e_greedy_begin = e_greedy_begin
        self.e_greedy_end = e_greedy_end
        self.n_actions = n_actions
        self.n_features = n_features
        # self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
#        self.epsilon = 0.1 if e_greedy_increment is not None else self.epsilon_max

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        # consist of [target_net, evaluate_net]
        self._build_net()

        if sess is None:
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
        else:
            self.sess = sess

        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []
        self.lost_his = []
        self.memory_save = []

    def _build_net(self):
        # ------------------ build evaluate_net ------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss
        with tf.variable_scope('eval_net'):
            # c_names(collections_names) are the collections to store variables
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 10, \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)  #[1, n_l1]

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_eval = tf.matmul(l1, w2) + b2  #[1, self.n_actions]

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))  # moving loss

        global_step = tf.train.get_or_create_global_step()
        learning_rate = tf.constant(value=0.01, shape=[], dtype=tf.float32)
        self.lr = tf.train.polynomial_decay(learning_rate, global_step,
                                                  200, end_learning_rate=0.0,
                                                  power=1.0, cycle=True)

        # summary_lr = tf.summary.scalar("lr", learning_rate)
        # train_op = tf.assign(global_step, global_step + 1)

        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # with tf.variable_scope('train'):
        #     self._train_op = tf.train.RMSPropOptimizer(learning_rate).minimize(self.loss)

        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')    # input
        with tf.variable_scope('target_net'):
            # c_names(collections_names) are the collections to store variables
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1) #[1, n_l1]

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_next = tf.matmul(l1, w2) + b2 #[1, self.n_actions]

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack((s, [a, r], s_))

        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition

        self.memory_counter += 1
        self.memory_save.append(transition)

    def save_memory(self):
        df = pd.DataFrame(self.memory_save)
        df.to_excel('memory_save2.xlsx')

    def explore(self,previous_action):
        choice_gather = []
        if previous_action == 0:
            choice_gather = [previous_action, previous_action + 1]
        elif previous_action < 21 - 1:
            choice_gather = [previous_action - 1, previous_action, previous_action + 1]
        elif previous_action == 20:
            choice_gather = [previous_action - 1, previous_action]
        elif previous_action == 21:
            choice_gather = [previous_action, previous_action + 1]
        elif previous_action < 42 - 1:
            choice_gather = [previous_action - 1, previous_action, previous_action + 1]
        elif previous_action == 41:
            choice_gather = [previous_action - 1, previous_action]
        action = np.random.choice(np.array(choice_gather))
        return action
    
    def choose_action(self, observation, episode_idx, episode_num, iteration, simulation_time, previous_action):
        flag = 0
        self.epsilon = self.e_greedy_begin + (self.e_greedy_end - self.e_greedy_begin) * 2 * (simulation_time * episode_idx + iteration) / (simulation_time * episode_num)

        sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]
        
        # # to have batch dimension when feed into tf placeholder
        # observation = observation[np.newaxis, :]
             
        if np.random.uniform() < self.epsilon:
            # choose best action
            observation = observation[np.newaxis, :]
            # print(batch_memory)
            Zero = np.zeros([1, batch_memory.shape[1]])
            for i in range(0, batch_memory.shape[0]):
                if batch_memory[i, :].any() == Zero[0].any():
                    flag = 1
            if flag:
                action = self.explore(previous_action)
            else:
                actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
                action = int(np.argmax(actions_value))
            # action = previous_action
        else:
            # choose random action
            action = self.explore(previous_action)

        return action

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('\ntarget_params_replaced\n')

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: batch_memory[:, -self.n_features:],  # fixed params
                self.s: batch_memory[:, :self.n_features],  # newest params
            })

        # change q_target w.r.t q_eval's action
        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        """
        For example in this batch I have 2 samples and 3 actions:
        q_eval =
        [[1, 2, 3],
         [4, 5, 6]]

        q_target = q_eval =
        [[1, 2, 3],
         [4, 5, 6]]

        Then change q_target with the real q_target value w.r.t the q_eval's action.
        For example in:
            sample 0, I took action 0, and the max q_target value is -1;
            sample 1, I took action 2, and the max q_target value is -2:
        q_target =
        [[-1, 2, 3],
         [4, 5, -2]]

        So the (q_target - q_eval) becomes:
        [[(-1)-(1), 0, 0],
         [0, 0, (-2)-(6)]]

        We then backpropagate this error w.r.t the corresponding action to network,
        leave other action as error=0 cause we didn't choose it.
        """

        # train eval network
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                self.q_target: q_target})
        self.cost_his.append(self.cost)
        self.lost_his.append(self.cost)

        # increasing epsilon
#        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()

    def get_loss(self,check_learning_times):
        loss = self.lost_his
        result = 0
        for i in range(0, len(loss)):
            result += (loss[i])/check_learning_times
        result = result / len(loss)
        return result

    def get_cost(self,check_learning_times):
        cost = self.cost_his
        result = 0
        for i in range(0, len(cost)):
            result += (cost[i])/check_learning_times
        result = result / len(cost)
        return result


    def refresh_cost(self):
        self.cost_his = []




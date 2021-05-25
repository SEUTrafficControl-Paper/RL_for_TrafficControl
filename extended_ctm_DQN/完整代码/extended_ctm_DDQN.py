import json
import numpy as np
import random
from RL_brain import DeepQNetwork
import math
import tensorflow as tf
import matplotlib.pyplot as plt


def parameter_initialization(parameters):
    parameters['simulation_period'] = parameters['simulation_period'] * 3600
    parameters['capacity'] = parameters['free_speed'] * parameters['cri_density']
    parameters['beta1'] = parameters['capacity'] / (parameters['jam_density'] - parameters['cri_density'])
    parameters['cri_discharge_density'] = parameters['capacity'] * (1 - parameters['capacity_drop']) / parameters[
        'free_speed']
    parameters['beta2'] = parameters['capacity'] * (1 - parameters['capacity_drop']) / (
                parameters['jam_density'] - parameters['cri_discharge_density'])
    parameters['predict_steps'] = int(parameters['prediction_horizon'] / parameters['duration_time_step'])

    return parameters


def stochastic_demand(parameters):
    num_of_step = int(parameters['simulation_period'] / parameters['duration_time_step'])
    d_peak = 0.85 * parameters['capacity'] * parameters['num_lanes'] + random.randint(0, int(
        0.05 * parameters['capacity'] * parameters['num_lanes']))
    d_offpeak = random.gauss(4000, 400)
    demand = np.zeros(num_of_step)
    for min_idx in range(120):
        if min_idx < 60:
            demand[min_idx * 6: (min_idx + 1) * 6] = random.gauss(d_peak, d_peak / 20)
        else:
            demand[min_idx * 6: (min_idx + 1) * 6] = random.gauss(d_offpeak, d_offpeak / 20)
    return demand

# def stochastic_demand(parameters):
#     num_of_step = int(parameters['simulation_period'] / parameters['duration_time_step'])
# #    d_peak = 0.85 * parameters['capacity'] * parameters['num_lanes'] + random.randint(0, int(0.05 * parameters['capacity'] * parameters['num_lanes']))
#     d_peak = 0.85 * parameters['capacity'] * parameters['num_lanes']
#     d_offpeak = random.gauss(4000, 400)
#     demand = np.zeros(num_of_step)
#     for min_idx in range(120):
#         if min_idx < 60:
#             demand[min_idx * 6: (min_idx + 1) * 6] = d_peak
# #            demand[min_idx * 6: (min_idx + 1) * 6] = random.gauss(d_peak, d_peak / 20)
#         else:
#             demand[min_idx * 6: (min_idx + 1) * 6] = random.gauss(d_offpeak, d_offpeak / 20)
#     return demand

def define_downstream_boundary(parameters):
    num_of_step = int(parameters['simulation_period'] / parameters['duration_time_step'])
    downstream_boundary = 15 * np.ones(num_of_step)
    downstream_boundary[int(386 / 2): int(406 / 2)] = 75
    return downstream_boundary


def traffic_state_initialization(parameters):
    start_time_step = 209
    num_of_step = int(parameters['simulation_period'] / parameters['duration_time_step'])
    num_vehicles_origin_cell = np.zeros((1, num_of_step))
    arrival_flow = np.zeros((1, num_of_step))
    density_matrix = np.zeros((parameters['num_cells'], num_of_step))
    speed_matrix = np.zeros((parameters['num_cells'], num_of_step))
    flow_matrix = np.zeros((parameters['num_cells'], num_of_step))
    vsl_record_all = parameters["free_speed"] * np.ones((parameters['num_cells'], num_of_step))
    control_steps = int(parameters['duration_control_step'] / parameters['duration_time_step'])
    return num_of_step, num_vehicles_origin_cell, arrival_flow, density_matrix, speed_matrix, flow_matrix, vsl_record_all, control_steps, start_time_step


def get_jam_interval(speed_column, parameter):
    jam_intervals = []
    start_idx = -1
    end_idx = -1

    for idx in range(len(speed_column)):
        if speed_column[idx] < min(parameter['VSL_values']):
            if start_idx == -1 and end_idx == -1:
                start_idx = idx
                end_idx = idx
            else:
                end_idx = idx
        else:
            if start_idx != -1 and end_idx != -1:
                jam_intervals.append([start_idx, end_idx])
                start_idx = -1
                end_idx = -1

    if start_idx != -1:
        jam_intervals.append([start_idx, end_idx])

    return jam_intervals


def get_control_flag(control_flag, speed_column, parameter):
    if min(speed_column[0: 3]) < min(parameter['VSL_values']):
        control_flag = 0
    jam_intervals = get_jam_interval(speed_column, parameter)

    if len(jam_intervals) == 0 or len(jam_intervals) > 1:
        control_flag = 0

    return control_flag


def get_vsl_interval(speed_column, parameter):
    vsl_intervals = []
    start_idx = -1
    end_idx = -1
    end_flag = speed_column.tolist().index(min(speed_column.tolist()))

    for idx in range(1, end_flag):
        if min(parameter['VSL_values']) - 5 < speed_column[idx] < max(parameter['VSL_values']) + 8:
            if start_idx == -1 and end_idx == -1:
                start_idx = idx
                end_idx = idx
            else:
                end_idx = idx
        else:
            if start_idx != -1 and end_idx != -1:
                if end_idx - start_idx > 1:
                    vsl_intervals.append([start_idx, end_idx])
                start_idx = -1
                end_idx = -1

    if start_idx != -1 and end_idx - start_idx > 1:
        vsl_intervals.append([start_idx, end_idx])

    return vsl_intervals


def get_state(density_column, speed_column, flow_column, parameter):
    vsl_intervals = get_vsl_interval(speed_column, parameter)
    jam_intervals = get_jam_interval(speed_column, parameter)
    jam_cell_num = 0
    sum_jam_speed = 0

    for idx in range(1, parameter['num_cells']):
        if speed_column[idx] < min(parameter['VSL_values']):
            sum_jam_speed += speed_column[idx]
            jam_cell_num += 1

    if jam_cell_num == 0:
        jam_speed = 0
        jam_length = 0
        jam_tail = 0
    else:
        jam_speed = sum_jam_speed / jam_cell_num
        jam_length = jam_cell_num * parameter['cell_length']
        jam_tail = jam_intervals[0][0]

    if len(vsl_intervals) > 0:
        vsl_density = sum(density_column[vsl_intervals[0][0]: vsl_intervals[0][1]]) / (
                    vsl_intervals[0][1] - vsl_intervals[0][0])
        arrival_flow = sum(flow_column[0: vsl_intervals[0][0]]) / vsl_intervals[0][0]
    else:
        vsl_density = 0
        arrival_flow = sum(flow_column[0: idx]) / idx

    jam_length = int(jam_length * 10) / 10
    return arrival_flow, vsl_density, jam_speed, jam_length, jam_tail


def effective_action(speed_column, parameters, vsl_action):
    if min(speed_column.tolist()) < min(parameters['VSL_values']):
        end_idx = speed_column.tolist().index(min(speed_column.tolist()))
    else:
        for end_idx in range(parameters['num_cells']):
            if speed_column[end_idx] < int(float(vsl_action[0])):
                break
        if end_idx == parameters['num_cells'] - 1:
            end_idx = 1
    end_idx = end_idx - 1

    start_idx = int(end_idx - float(vsl_action[1]) / parameters['cell_length'])

    return start_idx, end_idx


def get_vsl_column(action, speed_column, parameters):
    start_idx, end_idx = effective_action(speed_column, parameters, action)
    start_idx = max(1, start_idx)

    vsl_column_new = parameters["free_speed"] * np.ones((parameters['num_cells'], 1))
    vsl_column_new[start_idx: end_idx] = action[0]

    return vsl_column_new


def get_key2(value1, value2, value3, value4, value5, state1_interval, state2_interval, state3_interval):
    s1 = int(math.floor(value1 / state1_interval))
    s2 = int(math.floor(value2 / state2_interval))
    s3 = int(math.floor(value3 / state3_interval))
    s4 = value4
    s5 = value5
    key = str(s1) + '-' + str(s2) + '-' + str(s3) + '-' + str(s4) + '-' + str(s5)
    return key


def get_key2_(value1, value2, value3, value4, value5, state1_interval, state2_interval, state3_interval):
    s1 = int(math.floor(value1 / state1_interval))
    s2 = int(math.floor(value2 / state2_interval))
    s3 = int(math.floor(value3 / state3_interval))
    s4 = value4
    s5 = value5
    # key = str(s1) + '-' + str(s2) + '-' + str(s3) + '-' + str(s4) + '-' + str(s5)
    return np.array([s1, s2, s3, s4, s5])


def state_update(density_matrix, speed_matrix, flow_matrix, vsl_record_all, step_num, downstream_boundary, parameters,
                 demand):
    if demand[step_num] > parameters["capacity"] * parameters["num_lanes"]:
        demand[step_num] = parameters["capacity"] * parameters["num_lanes"]

    for cell_idx in range(parameters["num_cells"]):
        if cell_idx == 0:
            sending_flow = min(parameters["capacity"], density_matrix[cell_idx, step_num] * parameters["free_speed"])
            receiving_flow = parameters["beta1"] * (parameters["jam_density"] - density_matrix[cell_idx + 1, step_num])
            flow_matrix[cell_idx, step_num] = min(sending_flow, receiving_flow)
        elif cell_idx == 1:
            sending_flow1 = parameters["capacity"] * (1 - parameters["capacity_drop"] * (
                        density_matrix[cell_idx, step_num] - parameters["cri_density"]) / (
                                                                  parameters["jam_density"] - parameters[
                                                              "cri_density"]))
            sending_flow = min(sending_flow1, density_matrix[cell_idx, step_num] * parameters["free_speed"])
            receiving_flow = min(
                parameters["beta1"] * (parameters["jam_density"] - density_matrix[cell_idx + 1, step_num]),
                parameters["beta1"] * (parameters["jam_density"] - density_matrix[cell_idx, step_num]) + parameters[
                    "beta2"] * (density_matrix[cell_idx, step_num] - density_matrix[cell_idx + 1, step_num]))
            flow_matrix[cell_idx, step_num] = min(parameters["capacity"], min(sending_flow, receiving_flow))
        elif cell_idx == parameters["num_cells"] - 1:
            sending_flow1 = parameters["capacity"] * (1 - parameters["capacity_drop"] * (
                        density_matrix[cell_idx - 1, step_num] - parameters["cri_density"]) / (
                                                                  parameters["jam_density"] - parameters[
                                                              "cri_density"]))
            sending_flow2 = parameters["capacity"] * (1 - parameters["capacity_drop"] * (
                        density_matrix[cell_idx, step_num] - parameters["cri_density"]) / (
                                                                  parameters["jam_density"] - parameters[
                                                              "cri_density"]))
            sending_flow = min(density_matrix[cell_idx, step_num] * parameters["free_speed"],
                               min(sending_flow1, sending_flow2))
            receiving_flow = parameters["beta1"] * (parameters["jam_density"] - downstream_boundary[step_num])
            flow_matrix[cell_idx, step_num] = min(parameters["capacity"], min(sending_flow, receiving_flow))
        else:
            sending_flow1 = parameters["capacity"] * (1 - parameters["capacity_drop"] * (
                        density_matrix[cell_idx - 1, step_num] - parameters["cri_density"]) / (
                                                                  parameters["jam_density"] - parameters[
                                                              "cri_density"]))
            sending_flow2 = parameters["capacity"] * (1 - parameters["capacity_drop"] * (
                        density_matrix[cell_idx, step_num] - parameters["cri_density"]) / (
                                                                  parameters["jam_density"] - parameters[
                                                              "cri_density"]))
            sending_flow = min(density_matrix[cell_idx, step_num] * vsl_record_all[cell_idx, step_num],
                               min(sending_flow1, sending_flow2))
            receiving_flow = min(
                parameters["beta1"] * (parameters["jam_density"] - density_matrix[cell_idx + 1, step_num]),
                parameters["beta1"] * (parameters["jam_density"] - density_matrix[cell_idx, step_num]) + parameters[
                    "beta2"] * (density_matrix[cell_idx, step_num] - density_matrix[cell_idx + 1, step_num]))
            flow_matrix[cell_idx, step_num] = min(parameters["capacity"], min(sending_flow, receiving_flow))

    for cell_idx in range(parameters["num_cells"]):
        if cell_idx == 0:
            density_matrix[cell_idx, step_num + 1] = density_matrix[cell_idx, step_num] + (
                        demand[step_num] / parameters["num_lanes"] - flow_matrix[cell_idx, step_num]) * parameters[
                                                         "duration_time_step"] / parameters["cell_length"] / 3600
        else:
            density_matrix[cell_idx, step_num + 1] = density_matrix[cell_idx, step_num] + (
                        flow_matrix[cell_idx - 1, step_num] - flow_matrix[cell_idx, step_num]) * parameters[
                                                         "duration_time_step"] / parameters["cell_length"] / 3600

        if density_matrix[cell_idx, step_num] != 0:
            speed_matrix[cell_idx, step_num] = flow_matrix[cell_idx, step_num] / density_matrix[cell_idx, step_num]
        else:
            speed_matrix[cell_idx, step_num] = parameters["free_speed"]

    return density_matrix, speed_matrix, flow_matrix


def get_sars(density_matrix, speed_matrix, flow_matrix, parameters, step_num, controlled_steps, start_time_step, this_action, previous_action):

    arrival_flow, vsl_density, jam_speed, jam_length, jam_tail = get_state(density_matrix[:, step_num - controlled_steps - 1], speed_matrix[:, step_num - controlled_steps - 1], flow_matrix[:, step_num - controlled_steps - 1], parameters)
    state = get_key2_(arrival_flow, vsl_density, jam_speed, jam_length, jam_tail, 300, 2, 5)
    arrival_flow2, vsl_density2, jam_speed2, jam_length2, jam_tail2 = get_state(density_matrix[:, step_num - 1], speed_matrix[:, step_num - 1], flow_matrix[:, step_num - 1], parameters)
    next_state = get_key2_(arrival_flow2, vsl_density2, jam_speed2, jam_length2, jam_tail2, 300, 2, 5)
    if step_num == start_time_step + controlled_steps:
        reward = (min(parameters['VSL_values']) - 5 * float(state[2])) * float(state[3]) - (min(parameters['VSL_values']) - 5 * float(next_state[2])) * float(next_state[3])
    else:
        if abs(this_action - previous_action) > 1:
            reward = 0
        else:
            reward = (min(parameters['VSL_values']) - 5 * float(state[2])) * float(state[3]) - (min(parameters['VSL_values']) - 5 * float(next_state[2])) * float(next_state[3])

    return state, next_state, reward


def action_decode(action_idx, parameters):
    if action_idx < 21:
        vsl_value = 50
        vsl_length = (action_idx + 1) * parameters["cell_length"]
    else:
        vsl_value = 60
        vsl_length = (action_idx - 20) * parameters["cell_length"]

    action = [vsl_value, vsl_length]

    return action


def get_vsl_record(density_matrix, speed_matrix, flow_matrix, vsl_record_all, step_num, controlled_steps,control_step_index, idx, episode_num, num_of_step, action_indices):
    arrival_flow, vsl_density, jam_speed, jam_length, jam_tail = get_state(density_matrix[:, step_num - 1],
                                                                           speed_matrix[:, step_num - 1],
                                                                           flow_matrix[:, step_num - 1], parameters)
    state = get_key2_(arrival_flow, vsl_density, jam_speed, jam_length, jam_tail, 300, 2, 5)
    if action_indices == []:
        previous_action = np.random.choice(np.arange(42)) #随机给的初始值
    else:
        previous_action = action_indices[-1]
    action_idx = RL.choose_action(state,idx, episode_num, step_num, num_of_step, previous_action)
    action = action_decode(action_idx, parameters)
    vsl_column = get_vsl_column(action, speed_matrix[:, step_num - 1], parameters)
    vsl_record_all[:,controlled_steps * control_step_index: controlled_steps * (control_step_index + 1)] = vsl_column.reshape(len(vsl_column), 1)

    return vsl_record_all, action_idx, state

def traffic_dynamics(parameters, downstream_boundary, RL, episode_num):

    "discrete intervals: 300 veh/h: 100 veh/h/lanes, 2 veh/km/lane, 5 km/h"

    loss_list = []
    cost_list = []
    reward_list = []
    action_indices = []

    for idx in range(episode_num):
        check_learning_times = 0
        num_of_step, num_vehicles_origin_cell, arrival_flow, density_matrix, speed_matrix, flow_matrix, vsl_record_all, control_steps, start_time_step = traffic_state_initialization(parameters)
        demand = stochastic_demand(parameters)
        reward_count = 0

        for step_num in range(num_of_step-1):

            control_step_index = int((step_num + 1) / control_steps)
            controlled_steps = int(parameters['duration_control_step'] / parameters['duration_time_step'])

            if step_num >= start_time_step and (step_num + 1) % controlled_steps == 0:
                control_flag = get_control_flag(1, speed_matrix[:, step_num - 1], parameters)
                if control_flag:
                    vsl_record_all, action_idx, state = get_vsl_record(density_matrix, speed_matrix, flow_matrix, vsl_record_all, step_num, controlled_steps, control_step_index,idx, episode_num, num_of_step, action_indices)
                    action_indices.append(action_idx)
                    if control_step_index > int((start_time_step + 1) / control_steps):
                        previous_state, next_state, reward = get_sars(density_matrix, speed_matrix, flow_matrix, parameters, step_num, controlled_steps, start_time_step, action_indices[-1], action_indices[-2])
                        RL.store_transition(previous_state, action_idx, reward, next_state)
                        # print([previous_state, action_idx, reward, next_state])
                        RL.learn()
                        check_learning_times += 1
                        reward_count += reward

            density_matrix, speed_matrix, flow_matrix = state_update(density_matrix, speed_matrix, flow_matrix, vsl_record_all, step_num, downstream_boundary, parameters, demand)
          
        # print(check_learning_times)
        cost_list.append(RL.get_cost(check_learning_times))
        loss_list.append(RL.get_loss(check_learning_times))
        RL.refresh_cost()
        reward_list.append(reward_count)
        if idx % 500 == 0:
            print("this is", idx)
    return loss_list,cost_list,reward_list

def cal_increment(sum_lr_times):
    begin = 0.1
    end = 1
    increment = (begin - end) / sum_lr_times
    return increment/2   #一半仅利用

def plot_loss(loss_list):
    plt.figure(figsize=(15, 10))
    plt.subplots_adjust(hspace=0.4, wspace=0.4)

    plt.subplot(1, 1, 1)
    plt.title('Loss')

    acc_loss_list_plot = []
    for idx in range(200, len(loss_list)):
        acc_loss_list_plot.append(sum(loss_list[idx - 200: idx]) / 200)

    x = np.linspace(0, len(acc_loss_list_plot), len(acc_loss_list_plot))
    plt.scatter(x, acc_loss_list_plot, s=5)
    plt.xlabel('episode')
    plt.yticks()
    plt.ylabel('loss')
    plt.grid(True)
    plt.savefig("Loss22", dpi=1000)
    plt.close()

def plot_cost(cost_list):
    plt.figure(figsize=(15, 10))
    plt.subplots_adjust(hspace=0.4, wspace=0.4)

    plt.subplot(1, 1, 1)
    plt.title('Cost')

    acc_cost_list_plot = []
    for idx in range(200, len(cost_list)):
        acc_cost_list_plot.append(sum(cost_list[idx - 200: idx]) / 200)

    x = np.linspace(0, len(acc_cost_list_plot), len(acc_cost_list_plot))
    plt.scatter(x, acc_cost_list_plot, s=5)
    plt.xlabel('episode')
    plt.yticks()
    plt.ylabel('cost')
    plt.grid(True)
    plt.savefig("Cost22", dpi=1000)
    plt.close()

def plot_reward(reward_list):
    plt.figure(figsize=(15, 10))
    plt.subplots_adjust(hspace=0.4, wspace=0.4)

    plt.subplot(1, 1, 1)
    plt.title('acc_Reward')

    acc_reward_list_plot = []
    for idx in range(100, len(reward_list)):
        acc_reward_list_plot.append(sum(reward_list[idx - 100: idx]) / 100)

    x = np.linspace(0, len(acc_reward_list_plot), len(acc_reward_list_plot))
    plt.scatter(x, acc_reward_list_plot, s=5)
    plt.xlabel('episode')
    plt.yticks()
    plt.ylabel('reward')
    plt.grid(True)
    plt.savefig("Reward22", dpi=1000)
    plt.close()


if __name__ == "__main__":
    with open('parameter_mpc.json', 'r') as json_file:
        parameters = json.load(json_file)
    parameters = parameter_initialization(parameters)
    episode_num = 10000

    sess = tf.Session()
    with tf.variable_scope('Double_DQN'):
        RL = DeepQNetwork(
            n_actions=42, n_features=5, memory_size=3000,
             sess=sess)  # annealing steps: 0.98/e_increment
    sess.run(tf.global_variables_initializer())
    downstream_boundary = define_downstream_boundary(parameters)
    loss_list,cost_list,reward_list = traffic_dynamics(parameters, downstream_boundary, RL, episode_num)
    # print(Q_table)
    plot_loss(loss_list)
    plot_cost(cost_list)
    plot_reward(reward_list)
    RL.save_memory()

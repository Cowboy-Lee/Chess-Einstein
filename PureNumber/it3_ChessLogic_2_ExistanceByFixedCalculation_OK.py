#
#
# 电脑学习爱恩斯坦棋的规则。
#
#   —— 单层结构输入，用numpy直接算出棋子是否存在的结果（没有学习）。
#
#   ———— 成功！！
#
#

''' 需要显示loss, 随机率(违规率)，readout_action值, 梯度，'''

import random
import tensorflow as tf
from collections import deque
import numpy as np
import time
from PureNumber.Einstein_PureNumber import *
# import marshal  #用于序列化
import pickle   #用于序列化
import os


CHECKPOINT_PATH = "nothing/"
SUMMARY_PATH = "Summary/it3_ChessLogic2_ExistanceByFixedNet/"
GAME = 'EinsteinChessLogic' # the name of the game being played for log files
'''
6种行为，前三种表示往左（右）走，后三种表示往上（下）走
每组三个行为分别表示比骰子小的数、骰子数、比骰子大的数
'''
ACTIONS = 6 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVE = 50000 # timesteps to observe before training
EXPLORE = 2000000 # frames over which to anneal epsilon
''' 随机选择行为的概率 '''
# FINAL_EPSILON = 0.0001 # final value of epsilon
MAX_EPSILON = 0.5
MIN_EPSILON = 0.0001
INITIAL_EPSILON = MAX_EPSILON # starting value of epsilon
STEPS_TO_CHANGE_EPSILON = 1000
''' 玩家行为记录库（用于决策）的大小 '''
REPLAY_MEMORY = 100000
BATCH = 256 # size of minibatch
FRAME_PER_ACTION = 1
''' 共有6!=720种开局 '''
STARTSELECTIONSCOUNT = 720
WinRecords=deque()
LoseRecords=deque()
''' 输赢记录（用于统计）的数量 '''
WIN_LOSE_RECORDS_COUNT = 100000
PLAYER_RED = 1
PLAYER_BLUE = -1

def weight_variable(shape, Name):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial, name=Name)

def bias_variable(shape, Name):
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial, name=Name)

def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

def createNetwork():
    # network weights
    W_conv1 = weight_variable([2, 2, 2, 32], "policy_L1_conv_W")
    b_conv1 = bias_variable([32], "policy_L1_conv_B")

    W_conv2 = weight_variable([2, 2, 32, 64], "policy_L2_conv_W")
    b_conv2 = bias_variable([64], "policy_L2_conv_B")

    W_conv3 = weight_variable([2, 2, 64, 128], "policy_L2_conv_W")
    b_conv3 = bias_variable([128], "policy_L2_conv_B")

    W_fc1 = weight_variable([25*128, 128], "policy_L3_fc_W")
    b_fc1 = bias_variable([128], "policy_L3_fc_B")

    W_fc2 = weight_variable([128, 2], "policy_L4_fc_W")
    b_fc2 = bias_variable([2], "policy_L4_fc_B")

    '''
    14层：
    0~5，表示红方每个棋子的位置以及它的右边、下边有没有子的层，
    6~11，表示蓝方每个棋子的位置以及它的左边、上边有没有子的层，
    0~11层，空白记为0，红方的子记为正数，蓝方的子记为负数。
    12层，表示得到的骰子点数（1~6），整层数值一样。
    13层，表示轮到谁走。全PLAYER_RED表示红方，全PLAYER_BLUE表示蓝方。
    '''
    # input layer
    s = tf.placeholder("float", [None, 5, 5, 2])

    h_conv1 = tf.nn.relu(conv2d(s, W_conv1, 1) + b_conv1)

    h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2, 1) + b_conv2)

    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)

    h_conv3_flat = tf.reshape(h_conv3, [-1, 25*128])

    h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

    # h_fc1_drop = tf.nn.dropout(h_fc1, 0.9)

    # readout layer
    readout = tf.matmul(h_fc1, W_fc2, name="outLayer") + b_fc2

    return s, readout

def trainNetwork(s, readout, sess):

    t = 0

    # # open up a game state to communicate with emulator
    game_state = GameState()

    startTime = time.time()
    print_first_time = True
    sumPredict = 0

    while "Einstein_ChessLogic" != "Einstein":
        s_t, av = game_state.GetRandomGame()
        # s_t是原始数据，2层
        # s_t[:, :, 0] 表示 gameboard
        # s_t[:, :, 1] 表示 dice
        diff = s_t[:, :, 1] - s_t[:, :, 0]
        pos_diff = np.maximum(diff, np.zeros((5,5)))
        neg_diff = np.maximum(-diff, np.zeros((5,5)))
        abs_diff = pos_diff + neg_diff
        mini = np.min(abs_diff)
        if (av[0]==1 and mini==0) or (av[0]==0 and mini!=0):
            sumPredict += 1

        t += 1

        if t%10 == 0:
            print("t = %d,  rate = %.6f" % (t, sumPredict / t * 100))
        if t>=3000:
            exit()



if __name__ == '__main__':
    sess = tf.InteractiveSession()
    s, readout = createNetwork()
    trainNetwork(s, readout, sess)

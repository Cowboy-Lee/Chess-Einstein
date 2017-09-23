#
#
# 电脑自己玩爱恩斯坦棋。
#
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



GAME = 'Einstein' # the name of the game being played for log files
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

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

def createNetwork():
    # network weights
    W_conv1 = weight_variable([2, 2, 14, 32])
    b_conv1 = bias_variable([32])

    W_conv2 = weight_variable([2, 2, 32, 64])
    b_conv2 = bias_variable([64])

    W_conv3 = weight_variable([2, 2, 64, 128])
    b_conv3 = bias_variable([128])

    W_conv4 = weight_variable([2, 2, 128, 256])
    b_conv4 = bias_variable([256])

    W_fc1 = weight_variable([25*256, 128])
    b_fc1 = bias_variable([128])

    W_fc2 = weight_variable([128, ACTIONS])
    b_fc2 = bias_variable([ACTIONS])

    '''
    14层：
    0~5，表示红方每个棋子的位置以及它的右边、下边有没有子的层，
    6~11，表示蓝方每个棋子的位置以及它的左边、上边有没有子的层，
    0~11层，空白记为0，红方的子记为正数，蓝方的子记为负数。
    12层，表示得到的骰子点数（1~6），整层数值一样。
    13层，表示轮到谁走。全PLAYER_RED表示红方，全PLAYER_BLUE表示蓝方。
    '''
    # input layer
    s = tf.placeholder("float", [None, 5, 5, 14])

    h_conv1 = tf.nn.relu(conv2d(s, W_conv1, 1) + b_conv1)

    h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2, 1) + b_conv2)

    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)

    h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4, 1) + b_conv4)

    h_conv3_flat = tf.reshape(h_conv4, [-1, 25*256])

    h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

    # readout layer
    readout = tf.matmul(h_fc1, W_fc2, name="outLayer") + b_fc2

    return s, readout

def trainNetwork(s, readout, sess):
    # define the cost function
    a = tf.placeholder("float", [None, ACTIONS])
    y = tf.placeholder("float", [None])
    readout_action = tf.reduce_sum(tf.multiply(readout, a), reduction_indices=1)
    loss = tf.reduce_mean(tf.square(y - readout_action))
    opt = tf.train.AdamOptimizer(1e-3, name="loss-optimizer")
    train_step = opt.minimize(loss)
    gradient = opt.compute_gradients(loss)


    # start training
    epsilon = INITIAL_EPSILON
    eps_step_counter = 0
    eps_error_counter = 0
    t = 0

    tfv_loss = tf.Variable(0.)
    tf.summary.scalar("loss", tfv_loss)
    tfv_epsilong = tf.Variable(0.)
    tf.summary.scalar("epsilon", tfv_epsilong)
    tf.summary.histogram("action", readout_action)
    tf.summary.histogram("gradient0", gradient[0])
    tf.summary.histogram("gradient1", gradient[1])
    tf.summary.histogram("gradient2", gradient[2])
    tf.summary.histogram("gradient3", gradient[3])
    tf.summary.histogram("gradient4", gradient[4])
    tf.summary.histogram("gradient5", gradient[5])
    tf.summary.histogram("gradient6", gradient[6])
    tf.summary.histogram("gradient7", gradient[7])
    tf.summary.histogram("gradient8", gradient[8])
    tf.summary.histogram("gradient9", gradient[9])
    tf.summary.histogram("gradient10", gradient[10])
    tf.summary.histogram("gradient11", gradient[11])

    # tfv_score = tf.Variable(0.)
    # tf.summary.scalar("score", tfv_score)

    summary_vars = [tfv_loss, tfv_epsilong, readout_action,
                    gradient[0], gradient[1], gradient[2], gradient[3],
                    gradient[4], gradient[5], gradient[6], gradient[7],
                    gradient[8], gradient[9], gradient[10], gradient[11],
                    ]
    summary_placeholders = [tf.placeholder("float"), tf.placeholder("float")]
    assign_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in range(len(summary_placeholders))]
    summary_op = tf.summary.merge_all()

    writer = tf.summary.FileWriter("Summary/", sess.graph)


    # # open up a game state to communicate with emulator
    game_state = GameState()
    game_state_eval = GameState()

    # store the previous observations in replay memory
    D_red = deque()
    D_blue = deque()

    # get the first state by doing nothing and preprocess the image to 80x80x4
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    '''从720种开局中选择一种'''
    red_start = random.randint(0,719)
    blue_start = random.randint(0,719)
    start_player = random.randint(0,1)*2-1
    s_eval = game_state_eval.InitializeGame(red_start, blue_start, start_player, draw=True)
    terminal_eval = False
    s_t = game_state.InitializeGame(red_start, blue_start, start_player)
    current_player = start_player

    # saving and loading networks
    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())
    checkpoint = tf.train.get_checkpoint_state("Checkpoints/")
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        # 保留下面这一句，在其它程序中将会很有用的呢
        t = int(checkpoint.model_checkpoint_path.split('/')[-1].split('-')[-1])
        fle1 = open(os.path.join('Checkpoints/D_red.data'), 'rb')
        D_red = pickle.load(fle1)
        fle1.close()
        fle1 = open(os.path.join('Checkpoints/D_blue.data'), 'rb')
        D_blue = pickle.load(fle1)
        fle1.close()
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights")

    startTime = time.time()
    lastEvalTime = time.time()
    print_first_time = True
    while "Einstein" != "2048":
        ''' 演示 '''
        now = time.time()
        if now - lastEvalTime>1:
            lastEvalTime = now
            if terminal_eval:
                s_eval = game_state_eval.InitializeGame(random.randint(0, 719), random.randint(0, 719), random.randint(0, 1) * 2 - 1, draw=True)
                terminal_eval = False
            else:
                readout_eval = readout.eval(feed_dict={s:[s_eval]})[0]
                action_eval = np.argmax(readout_eval)
                a_eval = np.zeros([ACTIONS])
                a_eval[action_eval] = 1
                s_eval, r_eval, terminal_eval = game_state_eval.step_in_mind(a_t, draw=True)

        '''按照 epsilon greedily 法选择一个行为'''
        readout_t = readout.eval(feed_dict={s : [s_t]})[0]
        a_t = np.zeros([ACTIONS])
        isMyChoise = True
        if random.random() <= epsilon:
            # print("----------Random Action----------")
            action_index = random.randrange(ACTIONS)
            a_t[action_index] = 1
            isMyChoise=False
        else:
            action_index = np.argmax(readout_t)
            if print_first_time:
                print("default action:", action_index)
                print_first_time = False
            a_t[action_index] = 1
        isIndex_1_4 = (action_index==1 or action_index==4)

        # run the selected action and observe next state and reward
        s_t1, r_t, terminal = game_state.step_in_mind(a_t)
        if terminal:
            # print(u"====   %s %s ===="%(
            #     u"红方" if current_player==PLAYER_RED else u"蓝方",
            #     u"***胜***" if r_t==1 else u"败"))
            if r_t == 1:
                print(u"====   %s 胜 ===="%(u"红方" if current_player==PLAYER_RED else u"蓝方"))
            if r_t == -1 and isMyChoise:
                eps_error_counter += 1

        ''' 额外的调试信息 '''
        # if (not isIndex_1_4) and r_t>0:
        #     print("action is not (1 or 4) and survive! ***********************************")

        ''' 调整epsilon '''
        if isMyChoise:
            eps_step_counter += 1
            if eps_step_counter % STEPS_TO_CHANGE_EPSILON == 0:
                epsilon = min(max( eps_error_counter / STEPS_TO_CHANGE_EPSILON, MIN_EPSILON), MAX_EPSILON)
                eps_step_counter=0
                eps_error_counter=0

        ''' store the transition in D '''
        D = (D_red if current_player==PLAYER_RED else D_blue)
        D.append([s_t, a_t, r_t, s_t1, terminal])
        if len(D) > REPLAY_MEMORY:
            D.popleft()
        if terminal and r_t==1:
            ''' 
            如果是玩家主动赢棋，上一个玩家的结局也要修改 
            如果玩家主动输棋（例如走法违规），则上一个玩家的结局不用改
            '''
            D = (D_blue if current_player == PLAYER_RED else D_red)
            D[-1][2] = -1
            D[-1][4] = True
            ''' 如果玩家主动赢棋，那么要记录结果 '''
            if current_player== PLAYER_RED:
                WinRecords.append(red_start)
                LoseRecords.append(blue_start)
            else:
                WinRecords.append(blue_start)
                LoseRecords.append(red_start)
            if len(WinRecords) > WIN_LOSE_RECORDS_COUNT:
                WinRecords.popleft()
                LoseRecords.popleft()

        # only train if done observing
        if t > OBSERVE:
            # sample a minibatch to train on
            D = (D_red if current_player == PLAYER_RED else D_blue)
            minibatch = random.sample(D, BATCH)

            # get the batch variables
            s_j_batch = [d[0] for d in minibatch]
            a_batch = [d[1] for d in minibatch]
            r_batch = [d[2] for d in minibatch]
            s_j1_batch = [d[3] for d in minibatch]

            y_batch = []
            readout_j1_batch = readout.eval(feed_dict = {s : s_j1_batch})
            '''
            改进：让结局影响到之前每一步？
            改进：terminal的形势要长期记录？
            '''
            for i in range(0, len(minibatch)):
                tmp_terminal = minibatch[i][4]
                # if terminal, only equals reward
                if tmp_terminal:
                    y_batch.append(r_batch[i])
                else:
                    y_batch.append(r_batch[i] + GAMMA * np.max(readout_j1_batch[i]))

            # perform gradient step
            _, trained_loss = sess.run([train_step, loss], feed_dict = {
                y : y_batch,
                a : a_batch,
                s : s_j_batch}
            )

            # print result
            if t%100==0:
                now = time.time()
                print("t = %d\t| time=%.1f minute(s)\t| eps=%.3f\t| loss=%.5f" % (t, (now-startTime)/60.0, epsilon, trained_loss))
            if t%10000==0:
                stats = [trained_loss, epsilon]
                sess.run(assign_ops[0], {summary_placeholders[0]: stats[0]})
                sess.run(assign_ops[1], {summary_placeholders[1]: stats[1]})
                s_op = sess.run(summary_op, feed_dict={s:s_j_batch, a:a_batch, y:y_batch})
                writer.add_summary(s_op, float(t))
                sum_score = 0


        else: # t <= OBSERVE
            if t%100==0:
                print("observing: %.1f%%" % (t*100/OBSERVE))

        t += 1
        if terminal:
            red_start = random.randint(0, 719)
            blue_start = random.randint(0, 719)
            start_player = random.randint(0, 1) * 2 - 1
            ''' 要记得 s_t 里的最后两层应该包含下一次骰子值和下一次的玩家信息 '''
            s_t = game_state.InitializeGame(red_start, blue_start, start_player)
            current_player = start_player
        else:
            current_player = -current_player
            s_t = s_t1

        # save progress every 100000 iterations
        if t % 100000 == 0:
            saver.save(sess, 'Checkpoints/' + GAME + '-dqn', global_step = t)
            fle = open(os.path.join('Checkpoints/D_red.data'), 'wb')
            pickle.dump(D_red, fle)
            fle.close()
            fle = open(os.path.join('Checkpoints/D_blue.data'), 'wb')
            pickle.dump(D_blue, fle)
            fle.close()
            print("graph saved.")

        # write info to files
        '''
        if t % 10000 <= 100:
            a_file.write(",".join([str(x) for x in readout_t]) + '\n')
            h_file.write(",".join([str(x) for x in h_fc1.eval(feed_dict={s:[s_t]})[0]]) + '\n')
            cv2.imwrite("logs_tetris/frame" + str(t) + ".png", x_t1)
        '''



if __name__ == '__main__':
    sess = tf.InteractiveSession()
    s, readout = createNetwork()
    trainNetwork(s, readout, sess)

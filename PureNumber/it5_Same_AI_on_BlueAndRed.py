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
from PureNumber.Einstein_PureNumber_InverseStep import *
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
REPLAY_MEMORY = 200000
BATCH = 256 # size of minibatch
FRAME_PER_ACTION = 1
''' 共有6!=720种开局 '''
STARTSELECTIONSCOUNT = 720
WinRecords=np.zeros((720,2), dtype='int')
LoseRecords=np.zeros((720,2), dtype='int')

PLAYER_RED = 1
PLAYER_BLUE = -1
'''记录文件目录'''
RECORD_PATH = 'Checkpoints_it5/'
'''是否需要绘图'''
SHOULD_DRAW = True

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
    W_conv1 = weight_variable([2, 2, 13, 32])
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
    s = tf.placeholder("float", [None, 5, 5, 13])

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
    t = 0

    tfv_loss = tf.Variable(0.)
    tf.summary.scalar("loss", tfv_loss)
    tfv_win = tf.Variable(0.)
    tf.summary.scalar("red_win_rate", tfv_win)

    # tfv_score = tf.Variable(0.)
    # tf.summary.scalar("score", tfv_score)

    summary_vars = [tfv_loss, tfv_win]
    summary_placeholders = [tf.placeholder("float"), tf.placeholder("float")]
    assign_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in range(len(summary_placeholders))]
    summary_op = tf.summary.merge_all()

    writer = tf.summary.FileWriter("Summary/", sess.graph)


    # # open up a game state to communicate with emulator
    game_state = GameState_InverseStep()
    game_state_eval = GameState_InverseStep()
    game_state_demo = GameState_InverseStep(draw=True)

    # store the previous observations in replay memory
    D_ = deque()

    # get the first state by doing nothing and preprocess the image to 80x80x4
    s_eval,_,_ = game_state_eval.InitializeGame(PLAYER_RED)
    terminal_eval = False
    s_demo,_,_ = game_state_demo.InitializeGame(PLAYER_RED)
    terminal_demo = False
    ''' s_t为13层结构的状态 '''
    s_t, red_start, blue_start = game_state.InitializeGame(PLAYER_RED)
    current_player = PLAYER_RED

    # saving and loading networks
    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())
    checkpoint = tf.train.get_checkpoint_state(RECORD_PATH)
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        # 保留下面这一句，在其它程序中将会很有用的呢
        t = int(checkpoint.model_checkpoint_path.split('/')[-1].split('-')[-1])
        # with open(os.path.join(RECORD_PATH + 'D_.data'), 'rb') as fle1:
        #     D_ = pickle.load(fle1)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights")

    startTime = time.time()
    lastEvalTime = time.time()
    win_red_count=0
    Q_game_results = deque()
    while "Einstein" != "2048":

        # epsilon阶级递减
        if t<EXPLORE and t%STEPS_TO_CHANGE_EPSILON==0:
            epsilon = MAX_EPSILON - (MAX_EPSILON-MIN_EPSILON)/(EXPLORE//STEPS_TO_CHANGE_EPSILON) * (t//STEPS_TO_CHANGE_EPSILON)
        elif t>EXPLORE:
            epsilon = MIN_EPSILON

        # 按照 epsilon greedily 法选择一个行为
        readout_t = readout.eval(feed_dict={s : [s_t]})[0]
        '''
        例：
        action_input_available = [1  0  1  1  0  0]  
        all_available = [0  2  3]
        readout_t = [-0.1  0.1  -0.2  -0.25  0.05  0.1]
        '''
        a_t = np.zeros([ACTIONS])
        # 以一定的概率随机走
        if random.random() <= epsilon:
            action_index = game_state.GetRandomActionIndex()
            a_t[action_index] = 1
        else:
            action_index = game_state.GetActionIndex(reference_readout=readout_t)
            a_t[action_index] = 1


        # run the selected action and observe next state and reward
        s_t1, r_t, terminal = game_state.step_in_mind(a_t)
        s_t_s, av_filter, av_lst = game_state.AvailableInputFromGameboard()


        # store the transition in D
        # D = (D_red if current_player==PLAYER_RED else D_blue)
        D_.append([s_t, a_t, r_t, s_t_s, av_filter, av_lst, terminal])
        if len(D_) > REPLAY_MEMORY:
            D_.popleft()
        if terminal:
            if current_player== PLAYER_RED:
                WinRecords[red_start][0]  += 1
                LoseRecords[blue_start][1]+= 1
            else:
                WinRecords[blue_start][1] += 1
                LoseRecords[red_start][0] += 1

        # 变更棋盘状态信息和参数
        if terminal:
            ''' 要记得 s_t 里的最后两层应该包含下一次骰子值和下一次的玩家信息 '''
            s_t, red_start, blue_start = game_state.InitializeGame(PLAYER_RED)
            current_player = PLAYER_RED
        else:
            current_player = -current_player
            s_t = s_t1

        # continue if in observation
        if len(D_) <= OBSERVE:
            if len(D_)%100==0:
                print("observing: %.1f%%" % (len(D_)*100/OBSERVE))
            continue

        # TRAIN -- train after observation
        # sample a minibatch to train on
        minibatch = random.sample(D_, BATCH)

        # get the batch variables
        s_j_batch = [d[0] for d in minibatch]
        a_batch = [d[1] for d in minibatch]
        r_batch = [d[2] for d in minibatch]
        s_j1_batch = np.array([d[3] for d in minibatch])
        s_j1_batch = s_j1_batch.reshape((-1, 5, 5, 13))
        av_filter_batch = [d[4] for d in minibatch]
        av_lst_batch = [d[5] for d in minibatch]
        terminals = [d[6] for d in minibatch]

        # print(s_j1_batch.shape)
        readout_j1_batch = readout.eval(feed_dict = {s : s_j1_batch})
        # print(readout_j1_batch.shape)
        readout_j1_batch = readout_j1_batch.reshape(-1, 6, 6)
        y_batch = []
        ''' 
        改进：让结局影响到之前每一步？
        改进：terminal的形势要长期记录？
        '''
        for i in range(0, len(minibatch)):
            tmp_terminal = terminals[i]
            # if terminal, only equals reward
            if tmp_terminal:
                y_batch.append(r_batch[i])
            else:
                tmp_sm = 0
                for idx in range(6):
                    tmp_sm += readout_j1_batch[i][idx][av_lst_batch[i][idx][np.argmax(readout_j1_batch[i][idx][av_filter_batch[i][idx]==1])]]
                y_batch.append(r_batch[i] - GAMMA * tmp_sm/6)

        # perform gradient step
        _, trained_loss = sess.run([train_step, loss], feed_dict = {
            y : y_batch,
            a : a_batch,
            s : s_j_batch}
        )

        t += 1

        # print result
        if t%100==0:
            now = time.time()
            if len(Q_game_results)>0:
                print("t = %d\t| time=%.1f minute(s)\t| eps=%.4f | loss=%.5f | red_win(1000 games)=%.1f%%" % (t, (now-startTime)/60.0, epsilon, trained_loss, (win_red_count/len(Q_game_results)*100)))
        # add summary every 10000 iterations
        if t%10000==0:
            stats = [trained_loss, (win_red_count/len(Q_game_results) if len(Q_game_results)>0 else 0)]
            sess.run(assign_ops[0], {summary_placeholders[0]: stats[0]})
            sess.run(assign_ops[1], {summary_placeholders[1]: stats[1]})
            s_op = sess.run(summary_op, feed_dict={s:s_j_batch, a:a_batch, y:y_batch})
            writer.add_summary(s_op, float(t))
            sum_score = 0
        # save progress every 100000 iterations
        if t % 100000 == 0:
            saver.save(sess, RECORD_PATH + GAME + '-dqn', global_step = t)
            with open(os.path.join(RECORD_PATH + 'win_lose.txt'), 'w') as fle:
                for (_x, _y) in WinRecords:
                    fle.write("%d\t%d\n" % (_x, _y))
                for (_x, _y) in LoseRecords:
                    fle.write("%d\t%d\n" % (_x, _y))
            # with open(os.path.join(RECORD_PATH + 'D_.data'), 'wb') as fle:
            #     pickle.dump(D_, fle)
            print("graph,  and win_lose saved.")



        # evaluation
        if terminal_eval:
            # print(u"====   %s 胜 ===="%(
            #     u"红方" if game_state_eval.player==PLAYER_RED else u"蓝方"))
            winner = (1 if game_state_eval.player==PLAYER_RED else 0)
            Q_game_results.append(winner)
            win_red_count+=winner
            if len(Q_game_results)>1000:
                winner = Q_game_results.popleft()
                win_red_count-=winner
            s_eval,_,_ = game_state_eval.InitializeGame(PLAYER_RED)
            terminal_eval = False
        else:
            a_eval = np.zeros([ACTIONS])
            if game_state_eval.player == PLAYER_BLUE:
                a_eval[game_state_eval.GetRandomActionIndex()] = 1
            else:
                readout_eval = readout.eval(feed_dict={s:[s_eval]})[0]
                a_eval[game_state_eval.GetActionIndex(readout_eval)] = 1
            s_eval, r_eval, terminal_eval = game_state_eval.step_in_mind(a_eval)

        # 演示
        now = time.time()
        if now - lastEvalTime>1.5:
            '''每1.5秒演示一次'''
            lastEvalTime = now
            if terminal_demo:
                s_demo,_,_ = game_state_demo.InitializeGame(random.randint(0, 1) * 2 - 1)
                terminal_demo = False
            else:
                a_demo = np.zeros([ACTIONS])
                if game_state_demo.player == PLAYER_BLUE:
                    a_demo[game_state_demo.GetRandomActionIndex()] = 1
                else:
                    readout_demo = readout.eval(feed_dict={s:[s_demo]})[0]
                    a_demo[game_state_demo.GetActionIndex(readout_demo)] = 1
                s_demo, _, terminal_demo = game_state_demo.step_in_mind(a_demo)


if __name__ == '__main__':
    sess = tf.InteractiveSession()
    s, readout = createNetwork()
    trainNetwork(s, readout, sess)

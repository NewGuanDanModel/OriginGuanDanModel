import os
import pickle
import time
from argparse import ArgumentParser
from multiprocessing import Process

from math import exp
from typing import List
import random

import numpy as np
import tensorflow as tf
import zmq
from pyarrow import deserialize, serialize
from tensorflow.keras.backend import set_session

from model import GDModel

parser = ArgumentParser()
parser.add_argument('--observation_space', type=int, default=(567, ),
                    help='The YAML configuration file')
parser.add_argument('--action_space', type=int, default=(5, 216),
                    help='The YAML configuration file')
parser.add_argument('--model', type=str, default='../model/',
                    help='The YAML configuration file')

class Player():
    def __init__(self, index : int, args) -> None:
        # Set 'allow_growth'
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        set_session(tf.Session(config=config))

        # 数据初始化
        self.args = args
        self.init_time = time.time()
        self.index = index

        # 模型初始化
        self.model  = GDModel(args.observation_space, (5, 216))
        # self.model  = GDModel((567, ), (5, 216))
        if self.index % 2 == 1:
            with open('./train20/penalty_training-24500.ckpt', 'rb') as g:
                new_weights = pickle.load(g)
            self.model.set_weights(new_weights)
        else:
            with open('./train20/penalty_training-24500.ckpt', 'rb') as g:
            #with open('./q_network.ckpt', 'rb') as g:
                new_weights = pickle.load(g)
            self.model.set_weights(new_weights)
    
    def sample(self, state) -> int:
        output = self.model.forward(state['x_batch'])
        size = len(output)
        if size == 1:
            return 0
        for i in range(size):
            output[i] += state['addition'][i] + state['penalty'][i]
        return np.argmax(output)
    
    def find_top_k_elements_indices(self, array, k : int) -> List[int]:
        res = []
        for i in range(max(k, 1)):
            res.append(np.argmax(array))
            array[i] = -1e10
        return res

def run_one_player(index, args):
    player = Player(index,args)

    # 初始化zmq
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f'tcp://*:{6000+index}')

    action_index = 0
    while True:
        state = deserialize(socket.recv())
        action_index = player.sample(state)
        socket.send(serialize(action_index).to_buffer())


def main():
    # 参数传递
    args, _ = parser.parse_known_args()

    def exit_wrapper(index, *x, **kw):
        """Exit all actors on KeyboardInterrupt (Ctrl-C)"""
        try:
            run_one_player(index, *x, **kw)
        except KeyboardInterrupt:
            if index == 0:
                for _i, _p in enumerate(players):
                    if _i != index:
                        _p.terminate()

    players = []
    for i in [0, 1, 2, 3]:
        print(f'start{i}')
        p = Process(target=exit_wrapper, args=(i, args))
        p.start()
        time.sleep(0.5)
        players.append(p)

    for player in players:
        player.join()

if __name__ == '__main__':
    main()

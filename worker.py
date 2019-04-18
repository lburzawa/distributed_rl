from vizdoom import *
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from math import ceil
import time
import copy
import os
import shutil
import argparse
from torchvision.transforms import ToTensor
from torchvision.transforms import Resize
from torchvision.utils import save_image
from PIL import Image
import cv2

class Worker:
    def __init__(self, config_file_path, resolution, frame_repeat, actions):
        self.config_file_path = config_file_path
        self.resolution = tuple(resolution)
        self.frame_repeat = frame_repeat
        self.engine = self.initialize_vizdoom()
        self.frame = self.preprocess(self.engine.get_state())
        self.actions = actions
        self.reward = 0.0
        self.finished = 0
        self.scores = []

    def reset(self):
        self.engine.new_episode()
        self.frame = self.preprocess(self.engine.get_state())
        self.finished = 0
        self.scores = []
        
    def initialize_vizdoom(self):        
        game = DoomGame()
        game.load_config(self.config_file_path)
        game.set_window_visible(False)
        game.set_mode(Mode.PLAYER)
        game.set_screen_format(ScreenFormat.CRCGCB)
        #game.set_screen_format(ScreenFormat.GRAY8)
        game.set_screen_resolution(ScreenResolution.RES_160X120)
        game.set_depth_buffer_enabled(False)
        game.init()
        return game

    def preprocess(self, state):
        img = state.screen_buffer
        img = np.moveaxis(img, [0,1,2], [2,0,1])
        img = cv2.resize(img, self.resolution)
        img = ToTensor() (img)
        return img

    '''
    def step(self, action):
        if self.finished:
            self.scores.append(self.engine.get_total_reward())
            self.engine.new_episode()
        self.engine.make_action(self.actions[action], self.frame_repeat)
        self.reward = self.engine.get_last_reward()
        self.finished = self.engine.is_episode_finished()
        self.frame_prev = self.frame
        if not self.finished:
            self.frame = self.preprocess(self.engine.get_state())
    '''     

    def shutdown(self):
        self.engine.close()
        

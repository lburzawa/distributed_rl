from vizdoom import *
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.transforms import ToTensor
from torchvision.transforms import Resize
from torchvision.utils import save_image
from PIL import Image
import cv2
import shutil
import math
from argparse import ArgumentParser
import os
import itertools as it
import random
from random import sample, randint
from time import time, sleep
from tqdm import trange
from model import DoomNet
from worker import Worker
from multiprocessing.pool import ThreadPool
import torch.distributed as dist

parser = ArgumentParser()
_ = parser.add_argument
_('--scenario', type = str, default = './scenarios/my_way_home.cfg', help = 'set path to the scenario')
_('--save_dir', type = str, default = './save', help = 'Save directory')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['MASTER_ADDR'] = '128.46.90.148'
os.environ['MASTER_PORT'] = '29500'
torch.backends.cudnn.benchmark=True
#torch.backends.cudnn.deteministic=True
random.seed(0)
torch.manual_seed(0)

learning_rate = 0.0001
discount_factor = 0.99
epochs = 100
training_steps_per_epoch = 10000
seq_len = 20
sequences_per_epoch = training_steps_per_epoch // seq_len
frame_repeat = 4
resolution = [42, 42]
reward_scaling = 1.0
value_loss_scaling = 0.5
entropy_loss_scaling = 0.01
max_grad = 40.0
num_workers = 20
config_file_path = args.scenario
load_model = False
model_dir = args.save_dir
model_loadfile = "./save/model.pth"
model_savefile = os.path.join(model_dir, "model.pth")

def prep_frames_batch(workers):
    output = torch.FloatTensor(len(workers), 3, workers[0].resolution[0], workers[0].resolution[1])
    for i in range(len(workers)):
        output[i] = workers[i].frame.clone()
    output = output.cuda()
    return output

def set_action(workers, actions):
    for i in range(len(workers)):
        workers[i].action = actions[i].item()
    return workers

def step(worker):
    worker.reward = worker.engine.make_action(worker.actions[worker.action], worker.frame_repeat)
    worker.finished = worker.engine.is_episode_finished()
    if worker.finished:
        worker.scores.append(worker.engine.get_total_reward())
        worker.engine.new_episode()
    worker.frame = worker.preprocess(worker.engine.get_state())

'''
def perform_action(workers, actions):
    for i in range(len(workers)):
        workers[i].step(actions[i].item())
    return workers
'''

def prep_rewards_batch(workers):
    output = torch.FloatTensor(len(workers))
    for i in range(len(workers)):
        output[i] = workers[i].reward
    output = output.cuda()
    return output

def prep_finished(workers, hidden):
    output = torch.FloatTensor(len(workers))
    mask = torch.ones(len(workers), 256)
    for i in range(len(workers)):
        if workers[i].finished:
            output[i] = 0
            mask[i] = torch.zeros(256)
        else:
            output[i] = 1
    output = output.cuda()
    mask = mask.cuda()
    hidden[0] = hidden[0] * mask
    hidden[1] = hidden[1] * mask
    return output, hidden
            
def get_scores(workers):
    scores = []
    for i in range(len(workers)):
        scores.extend(workers[i].scores)
        workers[i].scores = []
    scores = np.float64(scores)
    return workers, scores

def shutdown_games(workers):
    for i in range(len(workers)):
        workers[i].shutdown()
    return workers

def average_gradients(model):
    size = dist.get_world_size()
    for param in model.parameters():
        dist.all_reduce(param.grad, op = dist.ReduceOp.SUM)
        param.grad /= size

if __name__ == '__main__':

    if os.path.isdir(model_dir):
        shutil.rmtree(model_dir)
    os.makedirs(model_dir)
    of = open(os.path.join(model_dir, 'test.txt'), 'w')

    workers = []
    workers_test = []
    for i in range(num_workers):
        workers.append(Worker(config_file_path, resolution, frame_repeat))
        workers_test.append(Worker(config_file_path, resolution, frame_repeat))  

    #n = game.get_available_buttons_size()
    #actions = [list(a) for a in it.product([0, 1], repeat=n)]
    actions = [[0, 0, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0]]

    if load_model:
        print("Loading model from: ", model_loadfile)
        model = DoomNet(len(actions))
        my_sd = torch.load(model_loadfile)
        model.load_state_dict(my_sd)
    else:
        model = DoomNet(len(actions))
    model = model.cuda()
    #criterion = nn.SmoothL1Loss()
    criterion = nn.MSELoss()
    nll = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

    whole_batch = torch.arange(num_workers)
    ones = torch.ones(num_workers).cuda()
    pool = ThreadPool()
    dist.init_process_group('nccl', rank = 0, world_size = 2)

    print("Starting the training!")
    start_time = time()
    forward_time = 0.0
    backward_time = 0.0
    test_time = 0.0
    hidden = model.init_hidden(num_workers)

    for epoch in range(epochs):
        print("\nEpoch %d\n-------" % (epoch))
        loss_value_total = 0.0
        loss_policy_total = 0.0
        loss_entropy_total = 0.0    

        print("Training...")
        model.train()
        for learning_step in trange(sequences_per_epoch, leave=False):
            loss = 0.0
            probs_list=[]
            log_probs_list=[]
            entropy_list = []
            value_list=[]
            reward_list=[]
            unfinished_list=[]
            forward_start_time = time()
            for t in range(seq_len):
                inp = prep_frames_batch(workers)
                (policy, value, hidden) = model(inp, hidden)
                probs = F.softmax(policy, 1)
                log_probs = F.log_softmax(policy, 1)
                a = probs.multinomial(num_samples=1).detach().squeeze(1)
                probs_list.append(probs[whole_batch, a])
                log_probs_list.append(log_probs[whole_batch, a])
                entropy_list.append(-(probs * log_probs).sum(1)) 
                value_list.append(value.squeeze(1))
                workers = set_action(workers, a)
                pool.map(step, workers)
                reward = prep_rewards_batch(workers) * reward_scaling                                                                               
                reward_list.append(reward)           
                unfinished, hidden = prep_finished(workers, hidden)
                unfinished_list.append(unfinished)

            inp = prep_frames_batch(workers)
            (_, value, _) = model(inp, hidden)
            value_list.append(value.squeeze(1))
            forward_time += (time() - forward_start_time)

            backward_start_time = time()
            R = value_list[t+1].detach()
            gae = torch.zeros(num_workers).cuda()
            for t in reversed(range(seq_len)):
                R = R * unfinished_list[t]
                R = reward_list[t] + discount_factor * R
                delta_t = reward_list[t] + discount_factor * value_list[t+1].detach() * unfinished_list[t] - value_list[t].detach()
                gae = gae * unfinished_list[t]
                gae = discount_factor * gae + delta_t             
                loss_policy = (-log_probs_list[t] * gae.detach()).mean()
                loss_value = criterion(value_list[t].unsqueeze(1), R.unsqueeze(1))
                loss_entropy = (-entropy_list[t]).mean()
                loss += loss_policy + value_loss_scaling * loss_value + entropy_loss_scaling * loss_entropy
                loss_policy_total += loss_policy.item()
                loss_value_total += loss_value.item()
                loss_entropy_total += loss_entropy.item()

            optimizer.zero_grad()
            loss.backward()
            average_gradients(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad)
            optimizer.step()
            for i in range(len(hidden)):
                hidden[i] = hidden[i].detach()
            backward_time += (time() - backward_start_time)
    
        workers, train_scores = get_scores(workers)
        total_steps = (epoch + 1) * training_steps_per_epoch * num_workers
        print("Results: mean: {:.2f}, std: {:.2f}, min: {:.2f}, max: {:.2f}, count: {:d}".format(train_scores.mean(), train_scores.std(), train_scores.min(), train_scores.max(), train_scores.shape[0]))
        print('Loss_policy: {:f}, loss_value: {:f}, loss_entropy: {:f}'.format(loss_policy_total/training_steps_per_epoch, loss_value_total/training_steps_per_epoch, loss_entropy_total/training_steps_per_epoch))

        print("\nTesting...")
        test_start_time = time()
        for worker in workers_test:
            worker.reset()
        with torch.no_grad():
            model.eval()
            hidden_test = model.init_hidden(num_workers)
            for learning_step in trange(50, leave=False):
                for t in range(seq_len):
                    inp = prep_frames_batch(workers_test)
                    (policy, value, hidden_test) = model(inp, hidden_test)
                    _, a = torch.max(policy, 1)
                    workers_test = set_action(workers_test, a)
                    pool.map(step, workers_test)           
                    unfinished, hidden_test = prep_finished(workers_test, hidden_test)
        test_time += (time() - test_start_time)
        workers_test, test_scores = get_scores(workers_test)
        print("Results: mean: {:.2f}, std: {:.2f}, min: {:.2f}, max: {:.2f}, count: {:d}".format(test_scores.mean(), test_scores.std(), test_scores.min(), test_scores.max(), test_scores.shape[0]))

        torch.save(model.state_dict(), model_savefile)
        total_time = time() - start_time
        print("Total training steps: {:d}, Total elapsed time: {:.2f} minutes, Time per step: {:.2f} ms".format(total_steps, total_time / 60.0, (total_time / total_steps) * 1000.0))
        print("Forward time: {:.2f} ms, Backward time: {:.2f} ms, Test time: {:.2f} ms".format((forward_time / total_steps) * 1000.0, (backward_time / total_steps) * 1000.0, (test_time / total_steps) * 1000.0))
        of.write('{:d},{:f},{:f}\n'.format(total_steps, total_time / 60.0, test_scores.mean()))
        of.flush()

    workers = shutdown_games(workers)
    print("======================================")
    print("Training finished.")



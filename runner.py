from tqdm import tqdm
from agent import Agent
from replay_buffer import Buffer
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import csv
class Runner:
    def __init__(self, args, env):
        self.args = args
        self.noise = args.noise_rate
        self.epsilon = args.epsilon
        self.episode_limit = args.max_episode_len
        self.env = env
        self.agent=Agent(args,0)
        self.buffer = Buffer(args)
        self.episode_score=[]
        self.save_path = self.args.save_dir + '/' + self.args.scenario_name
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.f = open("returns.txt",'a')
        self.writer = csv.writer(self.f)
        self.episode_length=300


    def run(self):
        for episode in tqdm(range(1,self.episode_limit+1)):
            state= self.env.reset()
            self.agent.noise.reset()
            score=0
            time_step = 0
            while True:
                action = self.agent.select_action(state,self.epsilon)
                #self.env.render()
                next_state, reward, done, _ = self.env.step(action)
                self.buffer.store_episode(state, action, reward,next_state,done)
                state = next_state
                score += reward
                if(done==True) or time_step==1000:
                    self.episode_length=time_step
                    break
                if self.buffer.current_size >= self.args.batch_size and time_step%self.args.learn_rate==0:
                    experiences = self.buffer.sample(self.args.batch_size)
                    self.agent.learn(experiences)
                time_step += 1
            self.noise = max(0.01, self.noise-((self.args.noise_rate-0.01)/self.episode_limit))
            self.epsilon = max(0.01, self.epsilon - 0.0000050)
            self.episode_score.append(score)
            self.writer.writerow([self.episode_score[episode-1],self.episode_length])
            if episode>0 and episode%100==0:
                self.agent.policy.save_model()
            if episode>0 and episode%50==0:
                self.f.flush()
    
           

    def evaluate(self):
           for episode in tqdm(range(1,self.episode_limit+1)):
            state= self.env.reset()
            self.agent.noise.reset()
            score=0
            time_step = 0
            while True:
                action = self.agent.select_action(state,0)
                self.env.render()
                next_state, reward, done, _ = self.env.step(action)
                self.buffer.store_episode(state, action, reward,next_state,done)
                state = next_state
                score += reward
                if(done==True) or time_step==1000:
                    self.episode_length=time_step
                    break
                time_step += 1
            self.noise = max(0.01, self.noise-((self.args.noise_rate-0.01)/self.episode_limit))
            self.epsilon = max(0.01, self.epsilon - 0.000015)
            self.episode_score.append(score)
            self.writer.writerow([self.episode_score[episode-1],self.episode_length])
            if episode>0 and episode%50==0:
                self.f.flush()
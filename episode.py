""" Contains the Episodes for Navigation. """
import random
import torch
import time
import sys
from constants import GOAL_SUCCESS_REWARD, STEP_PENALTY, BASIC_ACTIONS, FOUND_SUCCESS_REWARD
from environment import Environment
from utils.net_util import gpuify


class Episode:
    """ Episode for Navigation. """
    def __init__(self, args, gpu_id, rank, strict_done=False):
        super(Episode, self).__init__()

        self._env = None

        self.gpu_id = gpu_id
        self.strict_done = strict_done
        self.task_data = None
        self.glove_embedding = None
        self.object_seen=[]

        self.seed = args.seed + rank
        random.seed(self.seed)

        with open('./datasets/objects/int_objects.txt') as f:
            int_objects = [s.strip() for s in f.readlines()]
        with open('./datasets/objects/rec_objects.txt') as f:
            rec_objects = [s.strip() for s in f.readlines()]
        
        self.objects = int_objects + rec_objects

        self.actions_list = [{'action':a} for a in BASIC_ACTIONS]
        self.actions_taken = []

    @property
    def environment(self):
        return self._env

    def state_for_agent(self):
        return self.environment.current_frame

    def step(self, action_as_int):
        action = self.actions_list[action_as_int]
        self.actions_taken.append(action)
        return self.action_step(action)

    def action_step(self, action):
        self.environment.step(action)
        reward, terminal, action_was_successful = self.judge(action)

        return reward, terminal, action_was_successful

    def slow_replay(self, delay=0.2):
        # Reset the episode
        self._env.reset(self.cur_scene, change_seed = False)
        
        for action in self.actions_taken:
            self.action_step(action)
            time.sleep(delay)

    """""""""
    ORIGINAL
    """""""""
    # def judge(self, action):
    #     """ Judge the last event. """
    #     # immediate reward
    #     reward = STEP_PENALTY
    #     done = False
    #     action_was_successful = self.environment.last_action_success
    #
    #     if action['action'] == 'Done':
    #         done = True
    #         objects = self._env.last_event.metadata['objects']
    #         visible_objects = [o['objectType'] for o in objects if o['visible']]
    #         if self.target in visible_objects:
    #             reward += GOAL_SUCCESS_REWARD
    #             self.success = True
    #
    #     return reward, done, action_was_successful
    #
    """"""""""""

    def judge(self, action):
        """ Judge the last event. """
        # immediate reward
        reward = STEP_PENALTY
        done = False
        #I think this just returns whether or not the agent could execute the action at all.
        action_was_successful = self.environment.last_action_success

        objects = self._env.last_event.metadata['objects']
        visible_objects = [o['objectType'] for o in objects if o['visible']]

        if action['action'] == 'Done':
            done = True
            #if he chooses Done, he gets 'one last look'
            #go through target, see if we found them, marke as true if we do
            for i,v in enumerate(self.target):
                if v in visible_objects:
                    self.object_seen[i] = 1

            if all(self.object_seen):
                reward += GOAL_SUCCESS_REWARD
                self.success = True

        elif action['action'] == 'FOUND':
            #we want to make sure to only reward seeing a currently unseen target object
            for i,v in enumerate(self.target):
                if v in visible_objects and not(self.object_seen[i]):
                    self.object_seen[i] = 1
                    reward+= FOUND_SUCCESS_REWARD

        return reward, done, action_was_successful

    def new_episode(self, args, scene):
        
        if self._env is None:
            if args.arch == 'osx':
                local_executable_path = './datasets/builds/thor-local-OSXIntel64.app/Contents/MacOS/thor-local-OSXIntel64'
            else:
                local_executable_path = './datasets/builds/thor-local-Linux64'
            
            self._env = Environment(
                    grid_size=args.grid_size,
                    fov=args.fov,
                    local_executable_path=local_executable_path,
                    randomize_objects=args.randomize_objects,
                    seed=self.seed)
            self._env.start(scene, self.gpu_id)
        else:
            self._env.reset(scene)

        # Now two targets
        self.target = ['Tomato','Bowl']
        self.success = False
        self.cur_scene = scene
        self.actions_taken = []
        self.object_seen = []
        for i in range(len(self.target)):
            self.object_seen += [0]
        
        return True

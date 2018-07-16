#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 23:46:02 2018

@author: lihaoruo
"""

import threading
import numpy as np
import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.signal
import gym
from atari_wrappers import wrap_deepmind
from time import sleep

GLOBAL_STEP = 0

def update_target_graph(from_scope,to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var,to_var in zip(from_vars,to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder

def process_frame(image):
    image = np.reshape(image,[np.prod(image.shape)]) / 255.0
    return image

def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

class PPO_Network():
    def __init__(self,s_size,a_size,scope,trainer):
        with tf.variable_scope(scope):
            self.inputs = tf.placeholder(shape=[None,s_size],dtype=tf.float32)
            self.imageIn = tf.reshape(self.inputs,shape=[-1,84,84,1])
            self.policy, self.value = self.PPO_net(self.imageIn, 'ac', trainable=True)
            self.oldpolicy, self.oldvalue = self.PPO_net(self.imageIn, 'oldac', trainable=False)
            
            if scope != 'global':
                self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
                self.actions_onehot = tf.one_hot(self.actions,a_size,dtype=tf.float32)
                self.target_v = tf.placeholder(shape=[None],dtype=tf.float32)
                self.advantages = tf.placeholder(shape=[None],dtype=tf.float32)
                self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, [1])
                self.responsible_outputsold = tf.reduce_sum(self.oldpolicy * self.actions_onehot, [1])
                
                ratio = self.responsible_outputs / self.responsible_outputsold
                clip = ratio * self.advantages
                self.lclip = tf.minimum(clip, tf.clip_by_value(ratio, 0.2, 1.8) * self.advantages)

                self.entropy = tf.reduce_mean(self.policy * tf.log(self.policy))
                self.policy_loss = - tf.reduce_mean(self.lclip)
                
                value_reshape = tf.reshape(self.value,[-1])
                value_oldresh = tf.reshape(self.oldvalue,[-1])
                value_loss1 = 0.5 * tf.reduce_mean(tf.square(self.target_v - value_reshape))
                self.value_clip = value_oldresh + tf.clip_by_value(value_reshape - value_oldresh, -0.8, 0.8)
                value_loss2 = 0.5 * tf.reduce_mean(tf.square(self.target_v - self.value_clip))
                self.value_loss = tf.maximum(value_loss1, value_loss2)
                
                self.loss = 0.5 * self.value_loss + self.policy_loss + self.entropy * 0.01
                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope+'/ac')
                local_varsold = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope+'/oldac')
                self.gradients = tf.gradients(self.loss,local_vars)
                grads = self.gradients
                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global/ac')
                self.apply_grads = trainer.apply_gradients(zip(grads,global_vars))
                
                self.soft_replace = [[tf.assign(ta, ea)] for ta, ea in zip(local_varsold, local_vars)]

    def PPO_net(self, s, scope, trainable):
        with tf.variable_scope(scope):
            self.conv1 = slim.conv2d(activation_fn=tf.nn.relu,
                        inputs=s,num_outputs=32,
                        kernel_size=[8,8],stride=[4,4],padding='VALID',trainable=trainable)
            self.conv2 = slim.conv2d(activation_fn=tf.nn.relu,
                        inputs=self.conv1,num_outputs=64,
                        kernel_size=[4,4],stride=[2,2],padding='VALID',trainable=trainable)
            self.conv3 = slim.conv2d(activation_fn=tf.nn.relu,
                        inputs=self.conv2,num_outputs=64,
                        kernel_size=[3,3],stride=[1,1],padding='VALID',trainable=trainable)
            hidden = slim.fully_connected(slim.flatten(self.conv3),512,activation_fn=tf.nn.relu,trainable=trainable)
            
            policy = slim.fully_connected(hidden ,a_size,
                        activation_fn=tf.nn.softmax,
                        weights_initializer=normalized_columns_initializer(0.01),
                        biases_initializer=None,
                        trainable=trainable)
            value = slim.fully_connected(hidden, 1,
                        activation_fn=None,
                        weights_initializer=normalized_columns_initializer(1.0),
                        biases_initializer=None,
                        trainable=trainable)    
        return policy, value

class Worker():
    def __init__(self,env,name,s_size,a_size,trainer,model_path,global_episodes):
        self.name = "worker_" + str(name)
        self.number = name        
        self.model_path = model_path
        self.trainer = trainer
        self.global_episodes = global_episodes
        self.episode_rewards = []

        self.local_PPO = PPO_Network(s_size,a_size,self.name,trainer)
        self.update_local_ops = update_target_graph('global',self.name)        
        self.env = env
        
    def train(self,rollout,sess,gamma,lam,bootstrap_value):
        sess.run(self.local_PPO.soft_replace)
        
        rollout           = np.array(rollout)
        observations      = rollout[:,0]
        actions           = rollout[:,1]
        rewards           = rollout[:,2]
        next_observations = rollout[:,3]
        values            = rollout[:,5]

        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(self.rewards_plus,gamma)[:-1]
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
        advantages = discount(advantages,lam)

        feed_dict = {self.local_PPO.target_v:discounted_rewards,
                     self.local_PPO.inputs:np.vstack(observations),
                     self.local_PPO.actions:actions,
                     self.local_PPO.advantages:advantages}
        
        for _ in range(update_steps):
            sess.run(self.local_PPO.apply_grads, feed_dict=feed_dict)
        
    def work(self,gamma,lam,sess,coord,saver):
        global GLOBAL_STEP
        episode_count = sess.run(self.global_episodes)
        total_steps = 0
        print ("Starting worker " + str(self.number))
        with sess.as_default(), sess.graph.as_default():                 
            while not coord.should_stop():
                episode_buffer = []
                episode_reward = 0
                d = False
                s = self.env.reset()
                s = process_frame(s)
                while not d:
                    GLOBAL_STEP += 1
                    a_dist,v = sess.run([self.local_PPO.policy,self.local_PPO.value], 
                        feed_dict={self.local_PPO.inputs:[s]})
                    a = np.random.choice(a_dist[0],p=a_dist[0])
                    a = np.argmax(a_dist == a)

                    s1, r, d, _ = self.env.step(a)
                    if d == False:
                        s1 = process_frame(s1)
                    else:
                        s1 = s
                        
                    episode_buffer.append([s,a,r,s1,d,v[0,0]])
                    episode_reward += r
                    s = s1                    
                    total_steps += 1
                    
                    if len(episode_buffer) == 10 and d != True:
                        v1 = sess.run(self.local_PPO.value, feed_dict={self.local_PPO.inputs:[s]})[0, 0]
                        self.train(episode_buffer,sess,gamma,lam,v1)
                        episode_buffer = []
                        sess.run(self.update_local_ops)
                    if d == True:
                        break
                
                self.episode_rewards.append(episode_reward)
                if episode_count % 5 == 0 and episode_count != 0:
                    if self.name == 'worker_0' and episode_count % 5 == 0:
                        print('\n episode: ', episode_count, 'global_step:', GLOBAL_STEP,\
                              'mean episode reward: ', np.mean(self.episode_rewards[-5:]))
                    if episode_count % 100 == 0 and self.name == 'worker_0':
                        saver.save(sess,self.model_path+'/ppo2-'+str(episode_count)+'.cptk')
                        print ("Saved Model")
                episode_count += 1

def get_env(task):
    env_id = task.env_id
    env = gym.make(env_id)
    env = wrap_deepmind(env)
    return env

gamma = .99 
lam = 0.97
s_size = 7056
load_model = False
model_path = './ppo2model'
TAU = 0.99
update_steps = 1

benchmark = gym.benchmark_spec('Atari40M')
task = benchmark.tasks[3]
tf.reset_default_graph()
if not os.path.exists(model_path):
    os.makedirs(model_path)
    
env = get_env(task)
a_size = env.action_space.n
global_episodes = tf.Variable(0,dtype=tf.int32,name='global_episodes',trainable=False)
trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
master_network = PPO_Network(s_size,a_size,'global',None)
num_workers = 4
workers = []

for i in range(num_workers):
    env = get_env(task)
    workers.append(Worker(env,i,s_size,a_size,trainer,model_path,global_episodes))
saver = tf.train.Saver(max_to_keep=5)

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    sess.run(tf.global_variables_initializer())
    if load_model == True:
        print ('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver.restore(sess,ckpt.model_checkpoint_path)

    worker_threads = []
    for worker in workers:
        worker_work = lambda: worker.work(gamma,lam,sess,coord,saver)
        t = threading.Thread(target=(worker_work))
        t.start()
        sleep(0.5)
        worker_threads.append(t)
    coord.join(worker_threads)
    
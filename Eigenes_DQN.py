import gym
import numpy as np
import matplotlib.pyplot as plt

#import tensorflow as tf
from collections import deque,namedtuple
import random

import keras as K

class ReplayBuffer:
    def __init__(self,batch_size:int,buffersize:int):
        self.batch_size=batch_size
        self.buffersize=buffersize
        self.buffer=deque(maxlen=int(buffersize))
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def length_Replay(self):
        return len(self.buffer)
    
    def add_Experience(self,s,a,r,done,s2):
        self.buffer.append(self.experience(s,a,r,done,s2))
        
    def get_Experiences(self):
        
        experiences = random.sample(self.buffer, k=self.batch_size)

        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.vstack([e.action for e in experiences if e is not None])
        rewards = np.vstack([e.reward for e in experiences if e is not None])
        next_states = np.vstack([e.next_state for e in experiences if e is not None])
        dones = np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)
        return (states, actions, rewards, dones, next_states)
      
class Network:
    def __init__(self,in_size,out_size,hidden_size,learning_rate):
        self.input=in_size
        self.output=out_size
        self.hidden=hidden_size     
        self.learning_rate=learning_rate
        
    def built_Network(self):
        
        in_layer=K.layers.Input(shape=(self.input,), name='states')
        net =  K.layers.Dense(self.hidden[0])(in_layer)
        net = K.layers.Activation('relu')(net)
        net = K.layers.Dropout(0)(net)
        net = K.layers.Dense(self.hidden[1])(net)
        net = K.layers.Activation('relu')(net)
        net = K.layers.Dropout(0)(net)
        out_layer=K.layers.Dense(self.output,)(net)
        self.model = K.models.Model(inputs=in_layer, outputs=out_layer)
        self.model.summary()
        self.optimizer = K.optimizers.Adam(lr=self.learning_rate)
        self.model.compile(loss='mse', optimizer=self.optimizer)
        return self.model
    
class Agent:
    def __init__(self,environement,max_episodes,learn_rate,discount_rate,epsilon_start,epsilon_end,epsilon_decay,target_update_rate,hidden_space,batch_size,buffersize):

        self.env=gym.make(environement)
        self.input_space=self.env.observation_space.shape[0]
        self.output_space=self.env.action_space.n
        self.max=max_episodes
        self.epsilon=learn_rate
        self.gamma=discount_rate
        self.eps=epsilon_start
        self.ep_end=epsilon_end
        self.ep_dif=epsilon_decay
        self.update=target_update_rate
        self.hidden_space=hidden_space
        self.batch_size=batch_size
        self.buffer_size=buffersize
        
    def get_models(self):   
        dqn=Network(self.input_space,self.output_space,self.hidden_space,self.epsilon).built_Network()
        target_dqn=Network(self.input_space,self.output_space,self.hidden_space,self.epsilon).built_Network()
        replay=ReplayBuffer(self.batch_size,self.buffer_size)
        return dqn,target_dqn,replay
        
    def get_action(self):
        
        self.state = np.reshape(self.state, [-1, self.input_space])
        
        if( np.random.uniform() <= self.eps):
             action = random.choice(np.arange(self.output_space))
        else:
            action=np.argmax(self.dqn.predict(self.state))
        return action    
    
    def learn(self):
       # if(self.times_not_learned % self.update ==0):
            
        states, actions, rewards, next_states, dones =self.replay_buffer.get_Experiences()  
        
        for exp in range(len(states)):          
            state, action, reward, next_state, done = states[exp], actions[exp], rewards[exp], next_states[exp], dones[exp]
            state = np.reshape(state, [-1, self.input_space])
            next_state = np.reshape(next_state, [-1, self.input_space])

            learned=self.dqn.predict(state)         
            if(done):
                learned[0][action]=reward
            else:
                learned[0][action]=reward+self.gamma*np.argmax(self.target_dqn.predict(next_state)[0])   
            self.dqn.fit(state,learned,epochs=1, verbose=0)
         
        self.times_not_learned+=1     
       
        if(self.done):
            self.target_dqn.set_weights(self.dqn.get_weights())
            self.times_not_learned=1
            
        return learned
        
    def makeRoutine(self):
        self.dqn,self.target_dqn,self.replay_buffer=self.get_models()
        self.times_not_learned=1
        self.rewards_and_episodes=[]
        self.rewards=[]

        for episode in range(self.max): 
            self.state=self.env.reset()
            self.done=False
            self.reward=0
            self.ep_reward=0
            t_step=0
            while not self.done:
                self.action=self.get_action()
                self.next_state,self.reward,self.done,_=self.env.step(self.action)
                self.replay_buffer.add_Experience(self.state,self.action,self.reward,self.done,self.next_state)
                if(self.replay_buffer.length_Replay()>=self.batch_size):
                    self.learned=self.learn()  
                #    print("episode: {}/{}, explore_prob: {:.2f}, total reward: {}, average reward: {:.2f},mean-Q-Target: {}, state: {}".\
                #      format(episode, self.max, self.eps, self.ep_reward,np.mean(self.rewards),np.mean(self.learned),self.state[0]))    
                self.state=self.next_state
                self.ep_reward += self.reward
                t_step+=1
            self.rewards_and_episodes.append((episode,self.ep_reward)) 
            self.rewards.append(self.ep_reward) 
            if self.eps > self.ep_end:
                self.eps *= self.ep_dif 
           # if episode % 10 == 0:
            print("episode: {}/{},steps {}, explore_prob: {:.2f}, total reward: {}, average reward: {:.2f}, state: {}".\
                      format(episode, self.max,t_step, self.eps, self.ep_reward,np.mean(self.rewards),self.state[0]))        
         
            
        out_eps, out_rewards = np.array(self.rewards_and_episodes).T

        # plot reward v.s. episode
        plt.plot(out_eps, out_rewards)
        plt.xlabel('episode')
        plt.ylabel('reward')
        plt.show() 
         
         
np.random.seed(1024)

env='CartPole-v0'
max_episodes= 800         
       # "max_episode_len": 200,      # time steps per episode, 200 for CartPole-v0
        ## NN params
hidden_size= (256,256)                  
learning_rate= 0.0001   
discount_rate= 0.99             
update_target_DQN= 1      
        ## exploration prob
epsilon_start= 1.0
epsilon_end= 0.001      
epsilon_decay= 0.9   
        ## replay buffer
buffer_size= 1e5    
batch_size= 64
        
agent=Agent(env,max_episodes,learning_rate,discount_rate,epsilon_start,epsilon_end,epsilon_decay,update_target_DQN,hidden_size,batch_size,buffer_size)
agent.makeRoutine()
                
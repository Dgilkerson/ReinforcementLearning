# Imports.
import numpy as np
import numpy.random as npr
import pygame as pg
from collections import defaultdict

from SwingyMonkey import SwingyMonkey


class Learner(object):
    '''
    This agent jumps randomly.
    '''

    def __init__(self):
        self.last_state  = None
        self.last_full_state = None
        self.last_action = None
        self.last_reward = None
        self.Qlearner = defaultdict(lambda: [0,0])
        self.learning_rate = 0.1
        self.discount = 0.95
        self.it = 0
        self.counts = defaultdict(int)
        self.epochs = 1
        self.gravity = None

    def reset(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.epochs += 1
        self.gravity = None

    def discretize(self, state):
        t_to_m_vert = int(np.round(float(state['monkey']['top'] - state['tree']['top']) / 33))
        t_to_m_hor = int(np.round(float(state['tree']['dist']) / 150))
        m_vel = (state['monkey']['vel'] > 15)
        m_bot = (state['monkey']['bot'] > 75)
        return (t_to_m_vert, t_to_m_hor, m_vel, m_bot)

    # def alpha(self, t):
    #     return 10. / (10 + t)

    # def epsilon(self, n_epochs):
    #     return 0 if n_epochs > 50 else 0.05
    
    def get_gravity(self, drops):
        
        if drops[0] > drops[1]:
            if drops[0] - drops[1] > 2.5:
                #print('returning Inferred:', 4)
                return 4
            else:
                #print('returning Inferred:', 1)
                return 1
        else:
            return None

    def action_callback(self, state):
        '''
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        '''

        # You might do some learning here based on the current state and the last state.

        # You'll need to select and action and return it.
        # Return 0 to swing and 1 to jump.

        # new_action = npr.rand() < 0.1
        # new_state  = state

        # self.last_action = new_action
        # self.last_state  = new_state

        # return self.last_action
        
        self.it += 1
        state_disc = self.discretize(state)

        # epsilon greedy approach:
        # eps = self.epsilon(self.epochs)
        # if np.random.rand() > eps:
        #     action = np.argmax(self.Qlearner[state_disc])
        # else:
        #     action = np.random.rand() > 0.5

        # non epsilon greedy:
        action = np.argmax(self.Qlearner[state_disc])

        if self.last_state != None:
            if self.gravity is None and self.last_action == 0:
                drops = [self.last_full_state['monkey']['top'], state['monkey']['top']]
                self.gravity = self.get_gravity(drops)
            self.counts[(self.last_state, self.last_action)] += 1
            # learning rate is either function of times we've been in state or constant
            # lr = self.alpha(self.counts[(self.last_state, self.last_action)])
            lr = self.learning_rate
            self.Qlearner[self.last_state][self.last_action] += lr * (self.last_reward + self.discount * max(self.Qlearner[state_disc]) - self.Qlearner[self.last_state][self.last_action])

        self.last_action = action
        self.last_full_state = state
        self.last_state = state_disc
        # if self.it % 100 == 0:
        #     print(self.Qlearner)

        return action

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''

        self.last_reward = reward


def run_games(learner, hist, iters = 100, t_len = 100):
    '''
    Driver function to simulate learning by having the agent play a sequence of games.
    '''
    for ii in range(iters):
        # Make a new monkey object.
        swing = SwingyMonkey(sound=False,                  # Don't play sounds.
                             text="Epoch %d" % (ii),       # Display the epoch on screen.
                             tick_length = t_len,          # Make game ticks super fast.
                             action_callback=learner.action_callback,
                             reward_callback=learner.reward_callback)

        # Loop until you hit something.
        #print('Epoch Gravity:', swing.gravity)
        while swing.game_loop():
            pass

        # Save score history.
        hist.append(swing.score)
        print(f'it: {ii}')
        print(f'best score: {max(hist)}')
        print(f'average score: {sum(hist)/len(hist)}')
        # Reset the state of the learner.
        learner.reset()
    # print(f'best score: {max(hist)}')
    # print(f'average score: {mean(hist)}')
    pg.quit()
    return


if __name__ == '__main__':

	# Select agent.
	agent = Learner()

	# Empty list to save history.
	hist = []

	# Run games.
	run_games(agent, hist, 100, 10)

	# Save history.
	np.save('hist-1',np.array(hist))



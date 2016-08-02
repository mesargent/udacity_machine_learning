import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

#added numpy import
import numpy as np

#References: https://discussions.udacity.com/t/qtable-content-example/178397/11

        

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.state = None
        # Q Table to store state-action pairs and their rewards, a dictionary with states as keys and dictionaries
        # of action-rewards as values
        self.q_table = {}


    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        
        inputs = self.env.sense(self)
        env_state = self.env.agent_states[self]
        loc = env_state['location']
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        print "inputs {}".format(inputs)
        print "next waypoint {}".format(self.next_waypoint)


        # create a new state only if self doesn't have one
        if not self.state: 
            self.state = self.create_state()    
        
        # add state to Q table with initial rewards if it isn't in it already
        if self.state not in self.q_table:
            self.q_table[self.state] = {'right': 0, 'left': 0, 'forward': 0, None: 0}

        # TODO: Select action according to your policy

        # Code for random actions
        # self.actions = [None, 'forward', 'left', 'right']
        
        # rand = random.Random()
        # action = self.actions[rand.randint(0,3)]

        #choose action with max reward
        action = self.max_action(self.state)
        print "Action {}".format(action)
        

        # TODO: Learn policy based on state, action, reward
        #set alpha
        alpha = float(1/(self.env.t + 1))

        #set gamma
        gamma = .1

        reward = self.env.act(self, action)   

        #should reflect values for next state, since env.act was called
        next_state = self.create_state() 

        if next_state not in self.q_table:
            self.q_table[next_state] = {'right': 0, 'left': 0, 'forward': 0, None: 0}

        # The Q function
        self.q_table[self.state][action] = alpha * (self.q_table[self.state][action] + reward) + (1 - alpha) * (self.q_table[self.state][action] + reward + gamma * self.max_reward(next_state))
        new_reward = self.q_table[self.state][action]
        #print "Q Table: {}".format(self.q_table)

        print "LearningAgent.update(): loc = {}, deadline = {}, inputs = {}, action = {}, reward = {}".format(loc, deadline, inputs, action, new_reward)  # [debug]


    # to get highest reward
    def max_reward(self, state):
        rewards = self.q_table[state]
        return max(rewards.values())

    # get max key for actions, if there is more than one action with max value, choose one randomly
    # http://stackoverflow.com/a/5466625/399741
    def max_action(self, state):
        rewards = self.q_table[state]
        print "state {}".format(state)
        print "rewards {}".format(rewards)
        max_value = max(rewards.values())
        max_actions = []
        for action, reward in rewards.iteritems():
            if reward == max_value:
                max_actions.append(action)
        print "max actions {}".format(max_actions)
        rand = random.Random()
        return max_actions[rand.randint(0,len(max_actions)-1)]

    # Helper function for creating states from environment info
    #http://stackoverflow.com/a/1600806/399741
    def create_state(self):
        inputs = self.env.sense(self)
        env_state = self.env.agent_states[self]
        loc = env_state['location']
        return (loc, inputs['light'], inputs['oncoming'], inputs['left'], inputs['right'])
        
        
    
           

def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=False)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.5, display=True)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()

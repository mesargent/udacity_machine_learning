import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

#added numpy import
import numpy as np

#References: https://discussions.udacity.com/t/qtable-content-example/178397/11
#https://discussions.udacity.com/t/i-dont-know-if-this-idea-is-a-kind-of-cheating/170894/12

        

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint

        # TODO: Initialize any additional variables here
        self.state = None

        # Q Table to store state-action pairs and their rewards, a dictionary with states as keys and dictionaries
        self.q_table = {}
        
        #for testing purposes to see the percentage of successes
        self.trials = 0 
        self.successes = 0

         #To determine average reward for evaluation purposes
        self.total_reward_for_trial = 0
        self.total_updates = 0

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required

    def update(self, t):
             
        # To make sure self.state is not None, ensuring that this isn't the first time update has been run
        # Q learning only occurs on second update call and later so that previous state is always available
        # https://discussions.udacity.com/t/next-state-action-pair/44902/2?u=markesargent
        if not self.state:
            self.state = self.create_state()
            return

        # Gather inputs (for print out at end of method)
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator     
        inputs = self.env.sense(self)
        env_state = self.env.agent_states[self]
        loc = env_state['location']
        deadline = self.env.get_deadline(self)

       
        
        # This will the state before updating
        prev_state = self.create_state()

        ##################### TODO: Update state ######################################
        self.state = self.create_state()

        ##################### TODO: Select action according to your policy ############
        actions = [None, 'left', 'right', 'forward']
        epsilon = .1 - (self.env.t *.001)
        #epsilon = .2
      
        # choose action with max reward with prob epsilon. Start off more random, get more greedy
        rand = random.Random()
        rand_val = rand.random()

        #do max action with 1 - epsilon probability
        if rand_val > epsilon:
            action = self.max_action(prev_state)
        else:
            action = actions[rand.randint(0,len(actions)-1)]
        
        ##################### TODO: Learn policy based on state, action, reward ########

        # gets reward for action, advances self.state to new state
        reward = self.env.act(self, action) 

        #set alphan and gamma
        alpha = .5
        #gamma = 1.0/(self.env.t + 1) #to avoid div by zero
        gamma = 1

        # The Q function, sets value for previous state using this state as the next state
        self.q_table[prev_state][action] = self.q_table[prev_state][action] +  alpha * (reward + gamma * self.max_reward(self.state) - self.q_table[prev_state][action])

        #for printout
        new_reward = self.q_table[prev_state][action]
        self.total_reward_for_trial += reward
        self.total_updates +=1    

        print "LearningAgent.update(): loc = {0:}, deadline = {1:}, inputs = {2:}, action = {3:}, reward = {4:.2f}".format(loc, deadline, inputs, action, reward)  # [debug]

        #for reporting number of successes
        # if self.env.done:
        #     print "Average reward: {}".format(self.total_reward_for_trial/self.total_updates)
        #     if env_state['location'] == env_state['destination']:
        #         self.successes +=1
        #     print "Successes: {}".format(self.successes)
        



    # Helper function to get highest reward
    def max_reward(self, state):
        rewards = self.q_table[state]
        return max(rewards.values())

    # Helper function to get max key for actions, if there is more than one action with max value, choose one randomly
    # http://stackoverflow.com/a/5466625/399741
    def max_action(self, state):
        rewards = self.q_table[state]
        rand = random.Random()
        # print "state {}".format(state)
        # print "rewards {}".format(rewards)
        max_value = max(rewards.values())
        max_actions = []
        for action, reward in rewards.iteritems():
            if reward == max_value:
                max_actions.append(action)
        #print "max actions {}".format(max_actions)
        return max_actions[rand.randint(0,len(max_actions)-1)]

    # Helper function for creating states from environment info
    #http://stackoverflow.com/a/1600806/399741
    def create_state(self):
        inputs = self.env.sense(self)
        env_state = self.env.agent_states[self]
        loc = env_state['location']
        state = (inputs['light'], inputs['oncoming'], inputs['left'], self.next_waypoint)
        if state not in self.q_table:
            self.q_table[state] = {'right': 0, 'left': 0, 'forward': 0, None: 0}
        return state
        
    


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trial

    # Now simulate it
    sim = Simulator(e, update_delay=0.5, display=True)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()

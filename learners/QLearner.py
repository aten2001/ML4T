import numpy as np


class QLearner(object):

    def __init__(self, num_states=100, num_actions=4, alpha=0.2, gamma=0.9, rar=0.5, radr=0.99, dyna=0):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna

        self.Q = np.zeros(shape=(num_states, num_actions))
        self.s = 0
        self.a = 0

        if dyna:
            self.experience = []

    def querysetstate(self, s):
        """  		   	  			    		  		  		    	 		 		   		 		  
        @summary: Update the state without updating the Q-table  		   	  			    		  		  		    	 		 		   		 		  
        @param s: The new state  		   	  			    		  		  		    	 		 		   		 		  
        @returns: The selected action  		   	  			    		  		  		    	 		 		   		 		  
        """
        if np.random.random() < self.rar:
            action = np.random.randint(0, self.num_actions - 1)
        else:
            action = np.argmax(self.Q[s, :])

        self.s = s
        self.a = action

        return action

    def query(self, s_prime, r):
        """  		   	  			    		  		  		    	 		 		   		 		  
        @summary: Update the Q table and return an action  		   	  			    		  		  		    	 		 		   		 		  
        @param s_prime: The new state  		   	  			    		  		  		    	 		 		   		 		  
        @param r: The reward
        @returns: The selected action  		   	  			    		  		  		    	 		 		   		 		  
        """
        alpha, gamma, s, a = self.alpha, self.gamma, self.s, self.a

        self.Q[s, a] = (1 - alpha) * self.Q[s, a] + alpha * (r + gamma * self.Q[s_prime, np.argmax(self.Q[s_prime, :])])

        if np.random.random() < self.rar:
            action = np.random.randint(0, self.num_actions - 1)
        else:
            action = np.argmax(self.Q[s_prime, :])

        self.rar *= self.radr
        self.s = s_prime
        self.a = action

        if self.dyna:
            self.experience.append([s, a, s_prime, r])

            for i in range(self.dyna):
                j = np.random.randint(len(self.experience))
                s, a, s_prime, r = self.experience[j]

                self.Q[s, a] = (1 - alpha) * self.Q[s, a] + alpha * (
                        r + gamma * self.Q[s_prime, np.argmax(self.Q[s_prime, :])])

        return action

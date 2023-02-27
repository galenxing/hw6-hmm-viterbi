import copy
import numpy as np
class ViterbiAlgorithm:
    """ViterbiAlgorithm Class
    """    

    def __init__(self, hmm_object):
        """Initializes ViterbiAlgorithm to calcuale the best hidden state sequence.

        Args:
            hmm_object (_type_): _description_
        """              
        self.hmm_object = hmm_object

    def best_hidden_state_sequence(self, decode_observation_states: np.ndarray) -> np.ndarray:
        """Returns the best hidden state sequence given a list of observations

        Args:
            decode_observation_states (np.ndarray): a list of observations in order to decode the hidden state for

        Returns:
            np.ndarray: an array with the hidden states most likely for generating the observations
        """        
        # list of our observations
        Y = decode_observation_states
        
        # initialize variables
        o = self.hmm_object.observation_states
        s = self.hmm_object.hidden_states
        pi = self.hmm_object.prior_probabilities
        a = self.hmm_object.transition_probabilities
        b = self.hmm_object.emission_probabilities

        obs_dict = self.hmm_object.observation_states_dict
        state_dict = self.hmm_object.hidden_states_dict

        # k is the number of states
        # t is the number of observations we have
        k = len(s)
        t = len(Y)
        
        # T1 to keep track of the probabilities in each prior stats
        # T2 to track the pointer to the best prior state
        T1 = np.zeros((k,t))
        T2 = np.zeros((k,t))

        # initialize hidden probabilities for each state at t0
        for i in range(k):
            obs_idx = obs_dict[Y[0]]
            T1[i,0] = pi[i] * b[i,obs_idx]
            T2[i,0] = 0

        # iterate through all the states and observations
        for j in range(1,t):
            for i in range(k):
                obs_idx = obs_dict[Y[j]]
                T1[i,j] = np.max(T1[:, j-1] * a[:,i] * b[i,obs_idx])
                T2[i,j] = np.argmax(T1[:, j-1] * a[:,i] * b[i,obs_idx])

        z = np.zeros(t)
        x = [''] * t

        z[t-1] = np.argmax(T1[:,t-1])
        x[t-1] = s[int(z[t-1])]
        
        # backtrack from last obs
        for j in range(t-1, 0, -1):
            i = int(z[j])
            z[j-1] = T2[i,j]
            x[j-1] = s[int(z[j-1])]
            
        return x
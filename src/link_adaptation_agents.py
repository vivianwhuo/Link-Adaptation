"""
## Link Adaptation Agents
This package contains the base and derived classes for link adaptation agents. Currently supported agents (excluding base classes) are:
1. `ThompsonSamplingBandit`: Thompson sampling (TS)-based bandit agent
2. `OuterLoopLinkAdaptation`: Outer loop link adaptation (OLLA) 
3. `TrackingThompsonSamplingBandit`: TS-based bandit agent with constant discounting factor
4. `UnimodalThompsonsampling`: (TODO: add description)
5. `DiscountedThompsonSamplingBandit`: model-free TS-based bandit agent which only discounts the pulled arm
"""

import numpy as np
from cvxopt import matrix, solvers
from .channel_quality_index import estimate_sinr_from_cqi, determine_bler_at_sinr

class BaseConstrainedBandit():
    '''
    Base Constrained Bandit. Our implementation accepts `nrof_cqi` as an extra parameter.

    Parameters
    ----------
    nrof_rates: Number of bandit arms (K)
    nrof_cqi: (TODO: add description)
    packet_sizes: Reward value for each arm (r_k) if successful
    target_success_prob: Target success probability
    window_size: Window size for sliding window bandit. Events outside the window are discarded

    Notes
    -----
    Original implementation: https://github.com/vidits-kth/bayesla-link-adaptation/blob/3dc0de63f98e7f94f6790f834064d3a63d867c04/source.py#L230
    '''
    def __init__(self,
                 nrof_rates,
                 nrof_cqi,
                 packet_sizes,
                 target_bler):
        
        solvers.options['show_progress'] = False
        solvers.options['glpk'] = dict(msg_lev='GLP_MSG_OFF')

        self.nrof_rates = nrof_rates
        self.packet_sizes = packet_sizes
        
        self.target_success_prob = 1.0 - target_bler
        
        self.nrof_cqi = nrof_cqi
        
        self.ack_count  = np.zeros( ( nrof_rates, nrof_cqi ) )
        self.nack_count = np.zeros( ( nrof_rates, nrof_cqi ) )

    def act(self, cqi): # Implemented in child classes
        """Determines which arm to be pulled."""
        pass

    def update(self, rate_index, cqi, ack):  
        """Updates the bandit."""
        if ack:
            self.ack_count[ rate_index, cqi ] += 1
        else:
            self.nack_count[ rate_index, cqi ] += 1

class BaseModelFreeBandit():
    def __init__(self, 
                 nrof_rates,
                 packet_sizes):
        
        solvers.options['show_progress'] = False
        solvers.options['glpk'] = dict(msg_lev='GLP_MSG_OFF')

        self.nrof_rates = nrof_rates
        
        self.packet_sizes = packet_sizes
        
        self.ack_count  = np.zeros(nrof_rates)
        self.nack_count = np.zeros(nrof_rates)

    def act(self, cqi): # Implemented in child classes
        """Determines which arm to be pulled."""
        pass

    def update(self, rate_index, cqi, ack):
        """
        Updates the bandit. CQI is unused. It is only for compatibility with the other bandits.
        """
        if ack:
            self.ack_count[rate_index] += 1
        else:
            self.nack_count[rate_index] += 1

class ThompsonSamplingBandit(BaseConstrainedBandit):
    '''
    Thompson Sampling Bandit. Provides:
    (i) Unimodal Thompson sampling (UTS)
    (ii) Constrained Thompson sampling (Con-TS)

    Same as `BaseConstrainedBandit`, our implementation accepts `nrof_cqi` as an extra parameter.
    
    Parameters
    ----------
    nrof_rates: Number of bandit arms (K)
    nrof_cqi: (TODO: add description)
    packet_sizes: Reward value for each arm (r_k) if successful
    target_bler: (TODO: add description)
    prior_bler: (TODO: add description)
    prior_weight: (TODO: add description)

    Notes
    -----
    Original implementation: https://github.com/vidits-kth/bayesla-link-adaptation/blob/3dc0de63f98e7f94f6790f834064d3a63d867c04/source.py#L273
    '''
    def __init__(self, 
                 nrof_rates, 
                 nrof_cqi,
                 packet_sizes, 
                 target_bler,
                 prior_bler=[],
                 prior_weight=100):

        super().__init__(nrof_rates, nrof_cqi, packet_sizes, target_bler)

        # Exploit prior knowledge
        if not prior_bler == []:  
            for cqi in range( prior_bler.shape[1] ):
                for rate_index in range(self.nrof_rates):                    
                    prior_mu = 1.0 - prior_bler[rate_index, cqi]
                    self.ack_count[rate_index, cqi] = int( prior_weight * ( prior_mu  ) )
                    self.nack_count[rate_index, cqi] = int( prior_weight * ( 1.0 - prior_mu ) )

    def act(self, cqi):
        """
        Determines which arm to be pulled.

        Sample a success probability from beta distribution Beta(a, b)
        where `a = 1 + self.ack_count[cqi, rate_index]`
        and   `b = 1 + self.nack_count[cqi, rate_index]`.
        """
        sampled_success_prob = [ np.random.beta(1 + self.ack_count[ rate_index, cqi  ], 
                                                1 + self.nack_count[ rate_index, cqi ] ) 
                                for rate_index in range(self.nrof_rates)]
        
        expected_rewards = [( s * rew) for s, rew in zip(sampled_success_prob, self.packet_sizes)]

        return np.argmax(expected_rewards)

class OuterLoopLinkAdaptation(BaseConstrainedBandit):
    '''
    Outer Loop Link Adaptation: Bandit-like interface for OLLA.
    
    Parameters
    ----------
    nrof_rates: Number of bandit arms (K)
    nrof_cqi: (TODO: add description)
    packet_sizes: Reward value for each arm (r_k) if successful
    awgn_data: AWGN data
    target_bler: (TODO: add description)
    olla_step_size: (TODO: add description)
    
    Notes
    -----
    Original implementation: https://github.com/vidits-kth/bayesla-link-adaptation/blob/3dc0de63f98e7f94f6790f834064d3a63d867c04/source.py#L312
    '''
    def __init__(self, 
                 nrof_rates,
                 nrof_cqi,
                 packet_sizes, 
                 awgn_data,
                 target_bler,
                 olla_step_size = 0.1):
        
        super().__init__(nrof_rates, nrof_cqi, packet_sizes, target_bler)
        
        self.awgn_data = awgn_data

        self.sinr_offset = 0.0
        self.olla_step_size = olla_step_size

    def update(self, rate_index, cqi, ack):
        if ack:
            self.sinr_offset +=  self.olla_step_size
        else:
            self.sinr_offset -= self.target_success_prob / (1.0 - self.target_success_prob) * self.olla_step_size 

    def act(self, cqi):
        estimated_sinr = estimate_sinr_from_cqi(cqi, self.awgn_data )
        adjusted_sinr = estimated_sinr + self.sinr_offset

        bler_at_snr = determine_bler_at_sinr(adjusted_sinr, self.awgn_data)

        expected_rewards = [( (1.0 - bler) * rew) for bler, rew in zip( bler_at_snr, self.packet_sizes)]

        return np.argmax(expected_rewards)

class TrackingThompsonSamplingBandit(BaseConstrainedBandit):
    def __init__(self, 
                 nrof_rates, 
                 nrof_cqi,
                 packet_sizes, 
                 target_bler,
                 prior_bler=[],
                 prior_weight=100,
                 discount = 1):
        """Setting `discount = 1` is equivalent to the ordinary Thompson sampling."""
        
        super().__init__(nrof_rates, nrof_cqi,packet_sizes, target_bler)
        
        # Exploit prior knowledge
        if not prior_bler == []:  
            for cqi in range( prior_bler.shape[1] ):
                for rate_index in range(self.nrof_rates):                    
                    prior_mu = 1.0 - prior_bler[rate_index, cqi]
                    self.ack_count[rate_index, cqi] = int( prior_weight * ( prior_mu  ) )
                    self.nack_count[rate_index, cqi] = int( prior_weight * ( 1.0 - prior_mu ) )

    def act(self, cqi):
        
        # Sample a success probability from beta distribution Beta(a, b)
        # where a = 1 + self.ack_count[ cqi, rate_index ]
        # and   b = 1 + self.nack_count[ cqi, rate_index ]
        sampled_success_prob = [ np.random.beta(1 + self.ack_count[ rate_index, cqi  ], 
                                                1 + self.nack_count[ rate_index, cqi ] ) 
                                for rate_index in range(self.nrof_rates)]

        expected_rewards = [( s * rew) for s, rew in zip(sampled_success_prob, self.packet_sizes)]

        return np.argmax(expected_rewards)

    def update(self, rate_index, cqi, ack):
        for r in range(self.nrof_rates):
            for c in range(self.nrof_cqi):
                if r == rate_index and c == cqi:
                    if ack:
                        self.ack_count[r,c] = self.discount * self.ack_count[r,c] + 1
                        self.nack_count[r,c] = self.discount * self.nack_count[r,c]
                    else:
                        self.ack_count[r,c] = self.discount * self.ack_count[r,c]
                        self.nack_count[r,c] = self.discount * self.nack_count[r,c] + 1
                else:
                    self.ack_count[r,c] = self.discount * self.ack_count[r,c]
                    self.nack_count[r,c] = self.discount * self.nack_count[r,c]

class UnimodalThompsonsampling(BaseConstrainedBandit):
    def __init__(self, 
                 nrof_rates, 
                 packet_sizes, 
                 target_bler,
                 prior_bler=[],
                 prior_weight=100):
        
        super().__init__(nrof_rates, packet_sizes, target_bler)
        
        # Exploit prior knowledge
        if not prior_bler == []:  
            for cqi in range( prior_bler.shape[1] ):
                for rate_index in range(self.nrof_rates):                    
                    prior_mu = 1.0 - prior_bler[rate_index, cqi]
                    self.ack_count[rate_index, cqi] = int( prior_weight * ( prior_mu  ) )
                    self.nack_count[rate_index, cqi] = int( prior_weight * ( 1.0 - prior_mu ) )

    def act(self, cqi):
        
        # Sample a success probability from beta distribution Beta(a, b)
        # where a = 1 + self.ack_count[ cqi, rate_index ]
        # and   b = 1 + self.nack_count[ cqi, rate_index ]
        sampled_success_prob = [ np.random.beta(1 + self.ack_count[ rate_index, cqi  ], 
                                                1 + self.nack_count[ rate_index, cqi ] ) 
                                for rate_index in range(self.nrof_rates)]

        expected_rewards = [( s * rew) for s, rew in zip(sampled_success_prob, self.packet_sizes)]

        return np.argmax(expected_rewards)

class DiscountThompsonSamplingBandit(BaseModelFreeBandit):
    """The receiver only receives ACK signal, not CQI. Each MCS index is formulated as an arm."""
    def __init__(self, 
                 nrof_rates, 
                 packet_sizes,
                 discount = 1):
        
        super().__init__(nrof_rates, packet_sizes)
        self.discount = discount

    def act(self, cqi):
        """
        Determines which arm to be pulled.

        Sample a success probability from beta distribution Beta(a, b)
        where `a = 1 + self.ack_count[rate_index]`
        and   `b = 1 + self.nack_count[rate_index]`.

        Parameters
        ----------
        cqi: Unused
        """
        sampled_success_prob = [ np.random.beta(1 + self.ack_count[rate_index], 
                                                1 + self.nack_count[rate_index] ) 
                                for rate_index in range(self.nrof_rates)]
        
        expected_rewards = [(s * rew) for s, rew in zip(sampled_success_prob, self.packet_sizes)]

        return np.argmax(expected_rewards)

    def update(self, rate_index, cqi, ack):
        """
        Parameters
        ----------
        rate_index: int
            The index of the arm that is pulled.
        cqi: Unused
        ack: bool
        """
        # for r in range(self.nrof_rates):
        #     self.ack_count[r] = self.discount * self.ack_count[r]
        #     self.nack_count[r] = self.discount * self.nack_count[r]

        # self.ack_count[rate_index] += (1 if ack else 0)
        # self.nack_count[rate_index] += (0 if ack else 1)

        # only discount the arm that is pulled
        self.ack_count[rate_index] = self.discount * self.ack_count[rate_index] + (1 if ack else 0)
        self.nack_count[rate_index] = self.discount * self.nack_count[rate_index] + (0 if ack else 1)

def random_argmax(value_list):
    """
    Returns the index of the largest value in the supplied list.
    If there are multiple indices, returns any (instead of first, as in `np.argmax`) one of them randomly.
    
    Notes
    -----
    Original implementation: https://github.com/WhatIThinkAbout/BabyRobot/blob/cc6f00538ab21bbc69d94dbf0d8b2a26dcc5f11e/Multi_Armed_Bandits/PowerSocketSystem.py#L40
    """
    values = np.asarray(value_list)
    return np.argmax(np.random.random(values.shape) * (values==values.max()))

class UpperConfidenceBoundBandit(BaseConstrainedBandit):
    """
    Upper Confidence Bound (UCB1) algorithm.
    """
    def __init__(self,
                 nrof_rates,
                 nrof_cqi,
                 packet_sizes):
        
        super().__init__(nrof_rates, nrof_cqi, packet_sizes, target_bler=0)
        self.confidence_level = 2.0  # configurable
        self.num_pulls = np.zeros((nrof_rates, nrof_cqi))
        self.est_mean_rewards = np.zeros((nrof_rates, nrof_cqi))
        self.t = 0

    def act(self, cqi):
        """
        Determines which arm to be pulled.

        Play machine that maximizes mean expected reward plus uncertainty bonus.

        Parameters
        ----------
        cqi: Unused

        Notes
        -----
        UCB1 has logarithmic regret in finite-time.
        """
        # UCB1 requires playing each arm as initialization
        # if self.t < self.num_arms:
        #     # randomly pulls one unplayed arm
        #     return random_argmin(self.num_pulls)
        # else:
        #     return random_argmax(self.expected_rewards)
        return random_argmax([self.sample(r, cqi) for r in range(self.nrof_rates)])

    def update(self, rate_index, cqi, ack):
        # increment number of pulls
        self.num_pulls[rate_index, cqi] += 1

        # update mean reward estimate
        self.est_mean_rewards[rate_index, cqi] = (1 - 1.0/self.num_pulls[rate_index, cqi]) * self.est_mean_rewards[rate_index, cqi]
        + (1.0/self.num_pulls[rate_index, cqi]) * ack

    def uncertainty(self, rate_index, cqi):
        """t is the number of times the arm has been pulled."""
        if self.num_pulls[rate_index, cqi] == 0:
            return np.inf
        return self.confidence_level * (np.sqrt(2 * np.log(self.t + 1) / self.num_pulls[rate_index, cqi]))

    def sample(self, rate_index, cqi):
        """Returns UCB reward, which is the sample mean plus uncertainty of the arm."""
        return self.est_mean_rewards[rate_index, cqi] + self.uncertainty(rate_index, cqi)

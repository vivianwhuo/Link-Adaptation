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

        self.t = 0

    def act(self, cqi): # Implemented in child classes
        """Determines which arm to be pulled."""
        pass

    def update(self, rate_index, cqi, ack):  
        """Updates the bandit."""
        self.t += 1  # increment time step
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
        self.t += 1  # increment time step
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
        self.discount = discount
        
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

class UpperConfidenceBoundBandit(BaseConstrainedBandit):
    """
    Upper Confidence Bound (UCB1) algorithm.

    Attributes
    ----------
    confidence_level: float
        Confidence level for UCB1 algorithm.
    alpha: float
        Parameter for UCB1 algorithm.
    pulls: np.ndarray
        Number of pulls for each arm.
    mu: np.ndarray
        Empirical mean reward for each arm.

    """
    def __init__(self,
                 nrof_rates,
                 nrof_cqi,
                 packet_sizes,
                 confidence_level=1.0,
                 alpha=1.0):
        
        super().__init__(nrof_rates, nrof_cqi, packet_sizes, target_bler=0.1)
        self.confidence_level = confidence_level
        self.alpha = alpha
        self.pulls = np.zeros((nrof_rates, nrof_cqi))  # number of pulls for each arm

    def act(self, cqi):
        """
        Determines which arm to be pulled. Plays machine that maximizes empirical mean plus uncertainty bonus.

        Notes
        -----
        UCB1 has logarithmic regret in finite-time.
        """
        # Note: ignore cqi as arm
        estimate_success_prob = [self.ack_count[rate_index, cqi] /
                   (self.ack_count[rate_index, cqi] + self.nack_count[rate_index, cqi])
                   for rate_index in range(self.nrof_rates)]
        expected_rewards = np.array([(s * rew) for s, rew in zip(estimate_success_prob, self.packet_sizes)])
        radius = np.array([self.uncertainty(r, cqi) for r in range(self.nrof_rates)])
        f = expected_rewards + self.confidence_level * radius
        return np.argmax(f)

    def update(self, rate_index, cqi, ack):
        # increment ack, nack and number of pulls
        super().update(rate_index, cqi, ack)  # increment ack and nack
        self.pulls[rate_index, cqi] += 1

    def uncertainty(self, rate_index, cqi):
        """
        Uncertainty added to the empirical mean. Also known as padding function.
        Takes the form `b * sqrt(alpha * log(t) / n_i)`.
        n_i is the number of times the arm i has been pulled.
        """
        if self.pulls[rate_index, cqi] == 0:
            return 1  # initial confidence interval
        return np.sqrt(self.alpha * np.log(self.t + 1) / self.pulls[rate_index, cqi])

class DiscountedUCBBandit(UpperConfidenceBoundBandit):
    """
    A variant of UCB algorithm that has constant discount, where `pulls` are discounted by `gamma` in each iteration.

    Notes
    -----
    Original paper: "On Upper-Confidence Bound Policies for Non-Stationary Bandit Problems".
    """
    def __init__(self,
                 nrof_rates,
                 nrof_cqi,
                 packet_sizes,
                 confidence_level=1.0,
                 alpha=1.0,
                 gamma=0.9):
        
        super().__init__(nrof_rates, nrof_cqi, packet_sizes, confidence_level, alpha)
        
        self.mu = np.zeros((nrof_rates, nrof_cqi))  # empirical mean reward for each arm
        self.gamma = gamma  # discount factor

    def act(self, cqi):
        # ignore CQI now
        expected_rewards = np.array([(s * rew) for s, rew in zip(self.mu[:, cqi], self.packet_sizes)])
        radius = np.array([self.uncertainty(r, cqi) for r in range(self.nrof_rates)])
        f = expected_rewards + self.confidence_level * radius
        return np.argmax(f)

    def update(self, rate_index, cqi, ack):
        # discount and increment number of pulls
        prev_pulls = self.pulls[rate_index, cqi]
        self.pulls *= self.gamma
        self.pulls[rate_index] += 1

        self.mu[rate_index, cqi] = (prev_pulls / self.pulls[rate_index, cqi]) * self.mu[rate_index, cqi] * self.gamma + (1.0/self.pulls[rate_index, cqi]) * ack

    def uncertainty(self, rate_index, cqi):
        n_t_gamma = np.sum(self.pulls)
        if np.isclose(self.pulls[rate_index, cqi], 0, atol=0.001):
            return 1  # initial confidence interval
        return np.sqrt(self.alpha * np.log(n_t_gamma + 1) / self.pulls[rate_index, cqi])

class VariableDiscountedUCBBandit(UpperConfidenceBoundBandit):
    """
    A variant of UCB algorithm that has variable discount across arms.
    
    Notes
    -----
    Original paper: "A Multi-Armed Bandit Model for Non-Stationary Wireless Network Selection".
    """
    def __init__(self,
                 nrof_rates,
                 nrof_cqi,
                 packet_sizes,
                 confidence_level=2.0,
                 alpha=2.0,
                 gamma=0.9,
                 m0=5,
                 m1=5,
                 gamma_min=0.05,
                 gamma_max=0.95,
                 gamma_step=0.05):
        
        super().__init__(nrof_rates, nrof_cqi, packet_sizes, confidence_level, alpha)
        self.pulls = np.zeros((nrof_rates, nrof_cqi))  # number of pulls for each arm
        self.mu = np.zeros((nrof_rates, nrof_cqi))  # empirical mean reward for each arm
        self.m0 = m0
        self.m1 = m1
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.gamma_step = gamma_step
        # for now, same initial discount factor for all arms
        self.gammas = np.ones((nrof_rates, nrof_cqi)) * gamma

        # TODO: (low priority) improve runtime and memory efficiency.
        # current implementation for checking error sign: store all error signs for all arms
        # time: O(max(m0, m1))
        # space: O(r * c * max(m0, m1))
        self.error_signs = {}
        for r in range(self.nrof_rates):
            for c in range(self.nrof_cqi):
                # initialize as empty list
                self.error_signs[r, c] = []  # 1 if positive, -1 if negative

    def act(self, cqi):
        return np.argmax([self.sample(r, cqi) for r in range(self.nrof_rates)])

    def update(self, rate_index, cqi, ack):
        # discount and increment number of pulls
        for r in range(self.nrof_rates):
            for c in range(self.nrof_cqi):
                prev_pulls = self.pulls[r, c]
                self.pulls[r, c] = self.gammas[r, c] * prev_pulls + 1

                self.mu[r, c] = (prev_pulls / self.pulls[r, c]) * self.mu[r, c] * self.gammas[r, c] + (1.0/self.pulls[r, c]) * ack

        # compute and store the error sign for current arm
        error = ack - self.mu[rate_index, cqi]
        sign = 1 if error >= 0 else -1
        self.error_signs[rate_index, cqi].append(sign)

        if len(self.error_signs[rate_index, cqi]) > max(self.m0, self.m1):
            self.error_signs[rate_index, cqi].pop(0)  # avoid list too large

        # decrease memory
        consecutive_same_signs = 0
        for i in range(min(self.m1, len(self.error_signs[rate_index, cqi]))):
            if self.error_signs[rate_index, cqi][-i] == sign:
                consecutive_same_signs += 1
            else:
                break
        if consecutive_same_signs >= self.m1 and self.gammas[rate_index, cqi] > self.gamma_min:
            self.gammas[rate_index, cqi] -= self.gamma_step
        
        # increase memory
        consecutive_alternate_signs = 1  # note different init and range
        for i in range(min(self.m0, len(self.error_signs[rate_index, cqi])) - 1):
            if self.error_signs[rate_index, cqi][-i-1] != self.error_signs[rate_index, cqi][-i]:
                consecutive_alternate_signs += 1
            else:
                break
        if consecutive_alternate_signs >= self.m0 and self.gammas[rate_index, cqi] < self.gamma_max:
            self.gammas[rate_index, cqi] += self.gamma_step

    def uncertainty(self, rate_index, cqi):
        """
        Uncertainty added to the empirical mean. Also known as padding function.
        Takes the form `b * sqrt(alpha * log(t) / n_i)`.
        n_i is the number of times the arm i has been pulled.
        """
        if self.pulls[rate_index, cqi] == 0:
            return 1  # initial confidence interval
        return self.confidence_level * (np.sqrt(self.alpha * np.log(self.t + 1) / self.pulls[rate_index, cqi]))

    def sample(self, rate_index, cqi):
        """Returns UCB reward, which is the sample mean plus uncertainty of the arm."""
        return self.mu[rate_index, cqi] + self.uncertainty(rate_index, cqi)

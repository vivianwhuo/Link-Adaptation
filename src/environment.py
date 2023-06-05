'''
The environment package for the simulation. Default channel model is Rayleigh fading,
which assumes transmission with multipath scattering and without dominant propagation
along a line of sight (LOS).
'''

import itpp
import numpy as np
from .channel_quality_index import determine_cqi_from_sinr, determine_bler_at_sinr

def simulate_rayleigh_fading_channel(nrof_samples, avg_snr_dB, awgn_data, packet_sizes, norm_doppler = 0.01,
                                     seed = 9999, cqi_error_std = 0.0):
    """
    Orginal implementation: https://github.com/vidits-kth/bayesla-link-adaptation/blob/master/source.py
    """
    # Create a Rayleigh fading channel. The channel power is normalized to 1 by default
    channel = itpp.comm.TDL_Channel( itpp.vec('0.0'), itpp.ivec('0')) 
    channel.set_norm_doppler(norm_doppler)

    channel_coeff_itpp = itpp.cmat()
    channel.generate(nrof_samples, channel_coeff_itpp)

    channel_coeff = np.array(channel_coeff_itpp.get_col(0))
    
    avg_snr = 10 ** (0.1 * avg_snr_dB)
    instantaneous_channel_snrs = (np.absolute(channel_coeff) ** 2) * avg_snr
    
    _, nrof_rates = awgn_data['snr_vs_per'].shape
    instantaneous_blers = []
    channel_quality_indices = []
    for i in range( nrof_samples ):
        cqi_sinr_error = (itpp.random.randn() - 0.5) * cqi_error_std
        
        snr_dB = 10 * np.log10(instantaneous_channel_snrs[i])
        instantaneous_blers.append(determine_bler_at_sinr(snr_dB, awgn_data))
        channel_quality_indices.append(determine_cqi_from_sinr(snr_dB, packet_sizes, awgn_data, cqi_sinr_error)) 
    
    return (np.array(instantaneous_blers), np.array(channel_quality_indices))

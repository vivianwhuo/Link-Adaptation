'''
CQI-related functions.
'''

import numpy as np

def estimate_sinr_from_cqi(cqi, awgn_data):
    """
    Find the SINR for the given CQI to approximately achieve the given BLER target.
    
    Original implementation: https://github.com/vidits-kth/bayesla-link-adaptation/blob/3dc0de63f98e7f94f6790f834064d3a63d867c04/source.py#L14
    """
    REF_BLER_TARGET = 0.1
    REF_MCS_INDICES = [0, 1, 3, 5, 8, 9, 11, 14, 16, 20, 22, 24, 25, 26, 27, 28]

    awgn_snr_range_dB = awgn_data['snr_range_dB']
    awgn_snr_vs_bler = awgn_data['snr_vs_per']

    _, nrof_cqi = awgn_snr_vs_bler.shape

    bler = awgn_snr_vs_bler[:, REF_MCS_INDICES[ cqi ] ]

    if cqi == 0:
        return np.min(awgn_snr_range_dB)
    elif cqi == nrof_cqi - 1:
        return np.max(awgn_snr_range_dB)

    # Find the SNR indices closest to the REF_BLER_TARGET.
    # Estimate the instantaneous SNR by averaging these SNR values.
    # This assumes that the reported CQI actually had a BLER close to REF_BLER_TARGET.
    index1 = np.max(np.argwhere(REF_BLER_TARGET < bler))
    index2 = np.min(np.argwhere(REF_BLER_TARGET > bler))

    estimated_sinr_dB = (awgn_snr_range_dB[index1] + awgn_snr_range_dB[index2]) / 2.0

    return estimated_sinr_dB

def determine_cqi_from_sinr(snr_dB, packet_sizes, awgn_data, cqi_sinr_error = 0.0):
    """
    Original implementation: https://github.com/vidits-kth/bayesla-link-adaptation/blob/3dc0de63f98e7f94f6790f834064d3a63d867c04/source.py#L41
    """
    awgn_snr_range_dB = awgn_data['snr_range_dB']
    awgn_snr_vs_bler = awgn_data['snr_vs_per']

    REF_BLER_TARGET  = 0.1
    REF_MCS_INDICES = [0, 1, 3, 5, 8, 9, 11, 14, 16, 20, 22, 24, 25, 26, 27, 28]
    nrof_cqi = len( REF_MCS_INDICES )

    # Estimate the BLER for the reference MCSs used to calculate the CQI
    bler_at_snr = determine_bler_at_sinr(snr_dB + cqi_sinr_error, awgn_data)[ REF_MCS_INDICES ]
    
    # Calculate expcted throughput for all valid MCSs
    expected_tputs = np.multiply( ( 1 - bler_at_snr ), np.array( packet_sizes )[ REF_MCS_INDICES ] )
    
    # Ignore any MCSs with BLER less than REF_BLER_TARGET
    expected_tputs[ bler_at_snr > 0.1 ] = 0
    
    # The CQI is the index of the highest-throuput MCS from the reference MCSs
    cqi = 0
    if len( expected_tputs ) > 0:
        cqi = np.argmax( expected_tputs )
    
    return cqi
    

def determine_bler_at_sinr(snr_dB, awgn_data):
    """
    Original implementation: https://github.com/vidits-kth/bayesla-link-adaptation/blob/3dc0de63f98e7f94f6790f834064d3a63d867c04/source.py#L66
    """
    awgn_snr_range_dB = awgn_data['snr_range_dB']
    awgn_snr_vs_bler   = awgn_data['snr_vs_per']

    _, nrof_mcs = awgn_snr_vs_bler.shape

    bler_at_sinr = np.ndarray((nrof_mcs))

    for i in range(nrof_mcs):
        bler = awgn_snr_vs_bler[:, i]
        
        if snr_dB <= np.min(awgn_snr_range_dB):
            bler_at_sinr[i] = 1.0
        elif snr_dB >= np.max(awgn_snr_range_dB):
            bler_at_sinr[i] = 0.0
        else:
            index1 = np.max(np.argwhere(awgn_snr_range_dB < snr_dB))
            index2 = np.min(np.argwhere(awgn_snr_range_dB > snr_dB))

            bler_at_sinr[i] = (bler[index1] + bler[index2]) / 2.0

    return bler_at_sinr

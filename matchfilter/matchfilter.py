"""
Make a simple conceptual figure of a match filter
"""

import obspy
import matplotlib.pyplot as plt
import numpy as np
import bottleneck as bn


# ----------------------------- functions

def fast_normcorr(t, s):
    """
    fast normalized cc
    """
    if len(t) > len(s):  # switch t and s if t is larger than s
        t, s = s, t
    n = len(t)
    nt = (t - np.mean(t)) / (np.std(t) * n)
    sum_nt = nt.sum()
    a = bn.move_mean(s, n)[n - 1:]
    b = bn.move_std(s, n)[n - 1:]
    b *= np.sqrt((n - 1.0) / n)
    c = np.convolve(nt[::-1], s, mode="valid")
    result = (c - sum_nt * a) / b
    return result

# ----------------- Get waveform, buried waveform, and cc
# get waveform to plot
wf = obspy.read().select(component='Z')[0].data
wf = wf / np.std(wf)  # normalize

# embed in white noise
noise_with_wf = d

# get cc array
cc = fast_normcorr(wf, noise_with_wf)






fig = plt.figure(figsize=(10, 15), dpi=300)

ax = plt.subplot(3, 1, 1)





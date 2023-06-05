# Link Adaptation
This repository contains the simulation code for the industrial project "Ultra-Reliable and Low Latency Communications in 5G Networks".

## Dependencies

**Important: Some packages are OS-dependent. Linux distributions like Ubuntu and their WSL counterparts are proved to work.**

This simulation code is written in Python3. Running each of the cells in the corresponding [`Jupyter Notebooks`](https://github.com/jupyter/notebook) will execute the experiments, generate a results file, and plot the results.

The simulations make extensive use of the [`py-itpp`](https://github.com/vidits-kth/py-itpp), [`Numpy`](https://github.com/numpy/numpy), and [`Matplotlib`](https://github.com/matplotlib/matplotlib) packages.

Additionally, to speed up the generation of results, the simulations are parallezlized using the [`Ray`](http://ray.readthedocs.io/en/latest/index.html) package. It is possible to run single-threaded simulations at the cost of slowness, by commenting out the Ray-specific lines in the notebook - this is indicated in the appropriate sections of the code.

## Files  
[`link_adaptation.ipynb`](/link_adaptation.ipynb) contains the code for running the experiments and saving the results to disk.  
[`plot_results.ipynb`](/plot_results.ipynb) contains the code for plotting experiment results from the result file read from the disk.  
[`source.py`](/source.py) contains helper code for simulating a Rayleigh fading wireless channel and for different variants of multi-armed bandit (MAB) algorithms. Note that there is a bandit-like implementation for OLLA algorithm as well.
`AWGN_DATASET.npy` contains offline lookup data for mapping between instantaneous channel SNR and CQI values for each MCS.  

## References
1. Vidit Saxena and Joakim Jaldén, "Bayesian Link Adaptation under a BLER Target", In 2020 IEEE 21st International Workshop on Signal Processing Advances in Wireless Communications (SPAWC) on May 26-29, 2020.
2. L. Martínez, M. Cabrera-Bean and J. Vidal, "A Multi-Armed Bandit Model for Non-Stationary Wireless Network Selection," 2021 IEEE Globecom Workshops (GC Wkshps), Madrid, Spain, 2021, pp. 1-6, doi: 10.1109/GCWkshps52748.2021.9681963.
3. V. Raj and S. Kalyani, Taming Non-stationary Bandits: A Bayesian Approach. 2017.
4. (TODO: add references as required)
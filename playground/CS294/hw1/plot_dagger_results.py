import pickle
import numpy as np
import matplotlib.pyplot as plt

#######################################################
# Plot the number of DAgger iterations
# vs. the policy’s mean return, with error bars to show
# the standard deviation and also the expert returns as
# a comparison.
#######################################################

with open('results/Humanoid-v2' + '_expert_returns.pkl', 'rb') as fp:
    expert_returns = pickle.load(fp)

with open('results/Humanoid-v2' + '_DAgger_results.pkl', 'rb') as fp:
    dagger_results = pickle.load(fp)

dagger_means = dagger_results['means']
dagger_stds = dagger_results['stds']
dagger_train_size = dagger_results['train_size']


x_axis = np.arange(0, len(dagger_means), 1)
plt.errorbar(
    x_axis,
    dagger_means,
    dagger_stds,
    linestyle='-',
    marker='^',
    c='lightcoral')
plt.plot(expert_returns[0:len(dagger_means)], c='red')
plt.show()

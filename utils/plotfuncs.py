import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_losses(arr, legend_name, fname):
    plt.figure(figsize=(10, 10))
    sns.set_style('darkgrid')
    plt.plot(arr)
    plt.legend(legend_name)
    plt.savefig(fname)
    plt.close()

import matplotlib.pyplot as plt, mpld3
import numpy as np
import io
import matplotlib

matplotlib.use('Agg')
def getChart():
    figure = plt.figure(figsize=[6, 6])
    plt.plot([3,1,4,1,5], 'ks-', mec='w', mew=5, ms=20)
    plt.axis('off')
    plt.gca().set_position([0, 0, 1, 1])

    x = np.arange(0, 100, 0.00001)
    y = x*np.sin(2* np.pi * x)
    figdata = io.BytesIO()
    plt.savefig(figdata, format='svg')
    figdata.seek(0)
    
    return figdata


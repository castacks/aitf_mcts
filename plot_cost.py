from costmap import CostMap
from matplotlib import pyplot as plt
import numpy as np
import time
from tqdm import tqdm
import seaborn as sns

if __name__ == '__main__':
    mp3d = CostMap('./dataset/111_days/processed_data/train')
    fig = plt.figure()
    sp = fig.add_subplot(111)
    # fig.show()
    x = np.arange(-6, 6, 0.1)
    y = np.arange(-6, 6, 0.1)
    print(x.shape)
    xv, yv = np.meshgrid(x, y,sparse=False)
    arr = np.empty((120,120))
    for i in tqdm(range(len(xv))):
        for j in range(len(yv)):

            z = 0.5096 #(km)
            angle = -180 #degrees
            wind = 1 # 1 for right -1 for left
            # print(xv.shape)
            
            # print(mp3d.state_value(xv[i,j], yv[i,j], z, angle, wind))
            # plt.scatter(x= xv[i,j], y = yv[i,j], c = mp3d.state_value(xv[i,j], yv[i,j], z, angle, wind))
            arr[i][j] = mp3d.state_value(xv[i,j], yv[i,j], z, angle, wind)
    # plt.colorbar()
    xlabels = ['{:3.1f}'.format(x) for x in xv[0,:]]

    ax = sns.heatmap(arr,xticklabels=xlabels, yticklabels=xlabels)
    ax.invert_yaxis() 
    ax.set_xticks(ax.get_xticks()[::3])
    ax.set_xticklabels(xlabels[::3])
    ax.set_yticks(ax.get_yticks()[::3])
    ax.set_yticklabels(xlabels[::3])
    plt.savefig("mcts_"+ ".png")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob 
import os
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
ST=5
prob = 'Adv_1D'
# ty   = 'initial'
ty   = 'Final'
filename = f"../Data/{prob}/{prob}_*{ty}.csv"
files=glob(filename)

if not os.path.isdir(prob):
    os.mkdir(f'./{prob}')

for f0 in range(len(files)):
    print(files[f0])
    f= open(files[f0])
    numline = len(f.readlines())-1
    f.close()
    Nx=int(numline)

    df=pd.read_csv(files[f0])
    data = {col : df[col].values.reshape(Nx) for col in df.columns}
    # print(list(data.keys()))


    fig, ax = plt.subplots( 3, 1, sharex=True, sharey=False, dpi=140 )
    fig.subplots_adjust( hspace=0.1, wspace=0.0 )
    #fig.set_size_inches( 6.4, 12.8 )
    line_d, = ax[0].plot( df['x'], df['rho'],'ro', linestyle='-', markeredgecolor='k', markersize=3 )
    line_u, = ax[1].plot( df['x'], df['u'], 'bo', linestyle='-', markeredgecolor='k', markersize=3 )
    line_p, = ax[2].plot( df['x'], df['P'], 'go', linestyle='-', markeredgecolor='k', markersize=3 )
    ax[2].set_xlabel( 'x' )
    ax[0].set_ylabel( 'Density' )
    ax[1].set_ylabel( 'Velocity' )
    ax[2].set_ylabel( 'Pressure' )
    

    savename=files[f0].split('/')[-1].split('.')[0]+ '.png'
    plt.savefig(f'./{prob}/{savename}',dpi = 400)
    plt.close()
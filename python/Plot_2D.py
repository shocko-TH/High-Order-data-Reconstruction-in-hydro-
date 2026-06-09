import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from glob import glob 
import os
import imageio
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
ST=5
prob = 'Euler_2D_PPM_KHI_1'
ty   = 'Final'
filename = f"../Data/{prob}/{prob}_*{ty}.csv"

files=glob(filename)

if not os.path.isdir(prob):
    os.mkdir(f'./{prob}')
gif_path = f'./{prob}/{prob}_simulation.gif'

with imageio.get_writer(gif_path, mode='I', fps=5) as writer:
 for f0 in range(len(files)):

    print(files[f0])
    f= open(files[f0])
    numline = len(f.readlines())-1
    f.close()
    Nx=int(numline**0.5)

    df=pd.read_csv(files[f0],low_memory=False)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    data = {col : np.array(df[col].values.astype(float).reshape(Nx,Nx)) for col in df.columns}

    # fig, ax = plt.subplots(1, 3, dpi=140, figsize=(13, 4))    # fig.subplots_adjust( hspace=0.1, wspace=0.0 )
    
    # surf_d = ax[0].pcolor(data['x'], data['y'], data['rho'], cmap=cm.coolwarm)
    # ax[0].set_title('Density')

    # surf_E = ax[1].pcolor(data['x'], data['y'], data['E'], cmap=cm.coolwarm)
    # ax[1].set_title('Energy')

    # surf_P = ax[2].pcolor(data['x'], data['y'], data['P'], cmap=cm.coolwarm)
    # ax[2].set_title('Pressure')

    # time = data['t'][0][0]
    # fig.suptitle(f'{prob} | Time = {time}')

    # for a in ax:
    #     a.set_xlabel('X')
    #     a.set_ylabel('Y')
    #     plt.colorbar(a.collections[-1],ax=a)
    #     # a.view_init(elev=90, azim=-90)
    
    # fig.canvas.draw()  
        
    # image_rgba = np.asarray(fig.canvas.buffer_rgba())
    
    # image_rgb = image_rgba[:, :, :3]
    # writer.append_data(image_rgb)
    
    # plt.close(fig)
    # print(np.shape(data['mu'][::4,::4]))

    fig, ax = plt.subplots(1, 1, dpi=400, figsize=(7, 7)) 
    surf_d  = ax.pcolor(data['x'], data['y'], data['rho'], cmap=cm.coolwarm)
    
    stepx = 32
    stepy = 32
    
    quiver  = ax.quiver(data['x'][::stepx,::stepy], data['y'][::stepx,::stepy]
                        , data['mu'][::stepx,::stepy], data['mv'][::stepx,::stepy],
                        color='white',scale=10)

    ax.set_title('Density')

    time = data['t'][0][0]
    fig.suptitle(f'{prob} | Time = {time}')

    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    plt.colorbar(ax.collections[-1],ax=ax)


    savename=files[f0].split('/')[-1].split('.')[0]+ '.png'
    plt.savefig(f'./{savename}',dpi = 400)
    plt.close()

    break

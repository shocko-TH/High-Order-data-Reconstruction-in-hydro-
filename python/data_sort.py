import pandas as pd
import numpy as np
from glob import glob 
import scipy.io as sio
import os

# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
ST=5
prob = 'Euler_3D_KHI_PPM_WS'
ty   = 'Final'
filename = f"../Data/{prob}/{prob}_*{ty}.csv"

if not os.path.isdir(prob):
    os.mkdir(f'./{prob}')
gif_path = f'./{prob}/{prob}_simulation.gif'

files = glob(filename)
files.sort()
f= open(files[0])
numline = len(f.readlines())-1

tl=len(files)
Nx, Ny, Nz = int(np.cbrt(numline)),int(np.cbrt(numline)),int(np.cbrt(numline))

df_template = pd.read_csv(files[0])
all_columns = df_template.columns.tolist()

exclude_cols = ['time', 'step', 'timestep']
fields = [col for col in all_columns if col not in exclude_cols]

data = {field: [] for field in fields}
time = []  

for f0 in range(tl):
    print(f0)
    df = pd.read_csv(files[f0])
    
    for field in fields:
        field_3d = df[field].values.astype(float).reshape(Nz, Ny, Nx).transpose(2, 1, 0).copy()
        data[field].append(field_3d)

    if 'time' in df.columns:
        time.append(df['time'].iloc[0])

Data={}

for field in fields:
    Data[field] = np.array(data[field])

Data['time']=time

np.savez(f'./npy_data/{prob}.npz',**Data)


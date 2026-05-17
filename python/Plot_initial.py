import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
data=pd.read_csv('../Data/testing_1.csv')

Nx,Ny=256,256
X = data['x'].values.reshape(Nx,Ny)
Y = data['y'].values.reshape(Nx,Ny)
U = data['u0'].values.reshape(Nx, Ny)

plt.figure(figsize=(7, 6))
mesh = plt.pcolormesh(X, Y, U, shading='auto', cmap='jet')
plt.colorbar(mesh, label='Physical Field U')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('2D Hydrodynamics Simulation Initial State')
plt.savefig('fluid_density.png', dpi=300)
plt.show()
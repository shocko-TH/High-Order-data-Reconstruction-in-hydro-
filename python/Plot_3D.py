
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from glob import glob 
import os
import pyvista as pv

# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
prob = 'Euler_3D_KHI_PPM'

data=np.load(f'./npy_data/{prob}.npz')

# ==========================================================
x   = data['x'][0, :, :, :]
y   = data['y'][0, :, :, :]
z   = data['z'][0, :, :, :]
rho = data['rho'][0, :, :, :] 
u   = data['u'][0, :, :, :]
v   = data['v'][0, :, :, :]
w   = data['w'][0, :, :, :]

mask = ((x >= 0) & (y >= 0) & (z >= 0))
x = x[mask]
y = y[mask]
z = z[mask]
# x[mask] = np.nan
# y[mask] = np.nan
# z[mask] = np.nan
# rho[mask] = np.nan

time = data['time']
# Nx , Ny , Nz =np.array(np.shape(rho))
# ==========================================================
pv.OFF_SCREEN = True

grid = pv.ImageData()
grid.dimensions = np.shape(rho)

grid.point_data['Density'] = rho.ravel(order='F')

remove_bounds = [
    0.0, np.max(x),   
    np.max(y), 0.0,    
    0.0, np.max(z)    
]

clipped_grid = grid.clip_box(bounds=remove_bounds, invert=True)

pl = pv.Plotter(window_size=[1000, 1000])
pl.background_color = 'white'

opacity_map = [0.01, 0.02, 0.05, 0.1, 0.7]

vol_actor=pl.add_volume(
    clipped_grid,
    scalars='Density',
    cmap='jet',
    opacity=opacity_map,
    shade=True  
)
pl.camera.zoom(1.)
pl.camera_position = [(153, 153, 110),
                    (31.5, 31.5, 31.5),
                    (0.0, 0.0, 1.0)]


# === GIF ===
gif_path = f'./{prob}/T_{prob}.gif'
pl.open_gif(gif_path, fps=10)

n_frame=np.shape(data['x'])[0]
angle = 90.0 / n_frame



# for t in range(n_frame):
for t in [-1]:
# for t in range(10):

    if vol_actor is not None:
        pl.remove_actor(vol_actor)
    if vector_actor is not None:
        pl.remove_actor(vector_actor)

    print(t)
    rho = data['rho'][t, :, :, :] 
    rho[mask] = np.nan
    
    u0 = data['u'][t, :, :, :].copy()
    v0 = data['v'][t, :, :, :].copy()
    w0 = data['w'][t, :, :, :].copy()

    vectors = np.zeros((rho.size, 3))
    vectors[:, 0] = u0.ravel(order='F')
    vectors[:, 1] = v0.ravel(order='F')
    vectors[:, 2] = w0.ravel(order='F')

    grid = pv.ImageData()
    grid.dimensions = np.shape(rho)
    center_point = grid.center

    grid.point_data['Density'] = rho.ravel(order='F')
    grid.point_data['Velocity'] = vectors

    remove_bounds = [
        0.0, np.max(x),   
        np.max(y), 0.0,    
        0.0, np.max(z)    
    ]

    clipped_grid = grid.clip_box(bounds=remove_bounds, invert=True)
    pl.camera.Azimuth(angle)

    vol_actor=pl.add_volume(
        clipped_grid,
        scalars='Density',
        cmap='jet',
        opacity=opacity_map,
        shade=True  
    )

    arrows = clipped_grid.glyph(
        orient='Velocity', 
        scale='Velocity',  
        factor=0.1,        
        tolerance=0.03     
    )

    vector_actor = pl.add_mesh(arrows, color='black', lighting=False)

    pl.camera.zoom(1.)

    pl.add_text(f'Time : {time[t]}', name='time-label')
    pl.render()
    pl.write_frame()
    # pl.screenshot(f'./{prob}/volume_rendering_cutaway_{t}.png', transparent_background=False)

pl.close()
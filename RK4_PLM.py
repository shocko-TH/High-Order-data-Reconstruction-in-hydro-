
#---------------------------------------
# LF with RK4 solves time
#------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

#--------------------------------------------
# Parameters
#--------------------------------------------
# 網格點
Lx       = 1.0          # domain size in x
Ly       = 1.0          # domain size in y
Nx_In    = 128          # number of real cells in x
Ny_In    = 128          # number of real cells in y
nghost   = 3            # RK4 needs more ghost zones (set 3)
cfl      = 0.4          # Courant factor
gamma    = 5.0/3.0      # ideal gas gamma
end_time = 0.05         # final time

# 初始背景和blast wave條件
rho0     = 1.0          # background density
P0       = 1.0e-5       # background pressure
E_blast  = 1.0          # explosion energy per unit length
r_blast  = 0.03         # radius of initial energy injection

# derived constants
Nx = Nx_In + 2*nghost
Ny = Ny_In + 2*nghost
dx = Lx/Nx_In
dy = Ly/Ny_In

# floor values to avoid negative density/pressure
rho_floor = 1.0e-12
P_floor   = 1.0e-12


#--------------------------------------------------------------------
# compute pressure
# p = rho * v
# P = (gamma - 1) * (E - E_k)
#--------------------------------------------------------------------
def ComputePressure( U ):
    rho = U[...,0]
    px  = U[...,1]
    py  = U[...,2]
    E   = U[...,3]

    vx2 = (px/rho)**2.0
    vy2 = (py/rho)**2.0
    P   = (gamma-1.0)*( E - 0.5*rho*(vx2+vy2) )

    return np.maximum( P, P_floor )


#--------------------------------------------------------------------
# conserved variables --> primitive variables
# conserved --> 進網格點 = 出網格點
# primitive --> 確定沒有負壓跟負密度
# W = [rho, u, v, P]
#--------------------------------------------------------------------
def Conserved2Primitive( U ):
    W = np.empty_like( U )

    rho = np.maximum( U[...,0], rho_floor )
    P   = ComputePressure( U )

    W[...,0] = rho
    W[...,1] = U[...,1]/rho
    W[...,2] = U[...,2]/rho
    W[...,3] = P

    return W


#--------------------------------------------------------------------
# primitive variables --> conserved variables
#--------------------------------------------------------------------
def Primitive2Conserved( W ):
    U = np.empty_like( W )

    rho = np.maximum( W[...,0], rho_floor )
    u   = W[...,1]
    v   = W[...,2]
    P   = np.maximum( W[...,3], P_floor )

    U[...,0] = rho
    U[...,1] = rho*u
    U[...,2] = rho*v
    U[...,3] = P/(gamma-1.0) + 0.5*rho*(u*u+v*v)

    return U


#--------------------------------------------------------------------
# outflow boundary condition
# 把邊界的值複製進ghost zones，outflow的時候變化量就都是0
#--------------------------------------------------------------------
def BoundaryCondition( U ):
    # x boundary
    U[0:nghost,:,:]       = U[nghost:nghost+1,:,:]
    U[Nx-nghost:Nx,:,:]   = U[Nx-nghost-1:Nx-nghost,:,:]

    # y boundary
    U[:,0:nghost,:]       = U[:,nghost:nghost+1,:]
    U[:,Ny-nghost:Ny,:]   = U[:,Ny-nghost-1:Ny-nghost,:]


#--------------------------------------------------------------------
# compute time-step by CFL condition
# 找到最快速度的網格點，用cfl條件算dt的時長
#--------------------------------------------------------------------
def ComputeTimestep( U, t ):
    W = Conserved2Primitive( U )

    rho = W[...,0]
    u   = W[...,1]
    v   = W[...,2]
    P   = W[...,3]
    a   = np.sqrt( gamma*P/rho )     # sound speed

    max_x = np.amax( np.abs(u) + a )
    max_y = np.amax( np.abs(v) + a )

    dt_cfl = cfl/( max_x/dx + max_y/dy )
    dt_end = end_time - t

    return min( dt_cfl, dt_end )


#--------------------------------------------------------------------
# PLM-like reconstruction in 1D
#--------------------------------------------------------------------
def PLM_1D( q ):
    # q is a 1D array
    N = len(q)
    qL = np.zeros(N)
    qR = np.zeros(N)

    # PLM use j-1 & j+1
    for j in range(1, N-1):
        # 網格的左邊界和右邊界的斜率
        dqL = q[j] - q[j-1]     # 中間全部 - 左邊全部
        dqR = q[j+1] - q[j]     # 右邊全部 - 中間全部

        # van Leer limiter
        slope_LR      = dqL * dqR
        slope_limited = np.where(slope_LR > 0.0, 2.0 * slope_LR / (dqL + dqR), 0.0)

        # 用斜率和網格中心的值(q)，算出左右邊界
        # 左邊界
        qL[j] = q[j] - 0.5 * slope_limited      # 左邊界 = 原本值 - limiter
        # 右邊界
        qR[j] = q[j] + 0.5 * slope_limited      # 右邊界 = 原本值 + limiter

    # 維持ghost zone變化量是0
    # 直接複製左右ghost zone的中心值到ghost zone的左右邊界
    qL[0] = q[0]    # 第一項
    qR[0] = q[0]
    qL[N-1] = q[N-1]    # 最後一項 (q定義是np.zero(N))
    qR[N-1] = q[N-1]    

    return qL, qR


#--------------------------------------------------------------------
# PLM reconstruction in x direction
#--------------------------------------------------------------------
def DataReconstruction_PLM_x( U ):
    W  = Conserved2Primitive( U )
    WL = np.zeros_like( W )
    WR = np.zeros_like( W )

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 每一個丟進去算LR，再放進WLWR    
    # for j in range( Ny ):
    #     for v in range( 4 ):
    #         qL, qR = PLM_1D( W[:,j,v] )
    #         WL[:,j,v] = qL
    #         WR[:,j,v] = qR
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # 網格的左邊界和右邊界的斜率
    dqL = W[1:-1, :, :] - W[:-2, :, :]
    dqR = W[2:, :, :] - W[1:-1, :, :]

    # van Leer
    slope_LR = dqL * dqR
    slope_limited = np.where(slope_LR > 0.0, 2.0 * slope_LR / (dqL + dqR), 0.0)

    # 用斜率和網格中心的值，算出左右邊界
    WL[1:-1, :, :] = W[1:-1, :, :] - 0.5 * slope_limited
    WR[1:-1, :, :] = W[1:-1, :, :] + 0.5 * slope_limited

    # 維持ghost zone變化量是0
    # 直接複製左右ghost zone的中心值到ghost zone的左右邊界
    WL[0, :, :] = W[0, :, :]
    WR[0, :, :] = W[0, :, :]
    WL[-1, :, :] = W[-1, :, :]
    WR[-1, :, :] = W[-1, :, :]

    WL[...,0] = np.maximum( WL[...,0], rho_floor )
    WR[...,0] = np.maximum( WR[...,0], rho_floor )
    WL[...,3] = np.maximum( WL[...,3], P_floor )
    WR[...,3] = np.maximum( WR[...,3], P_floor )

    return Primitive2Conserved(WL), Primitive2Conserved(WR)


#--------------------------------------------------------------------
# PLM reconstruction in y direction
#--------------------------------------------------------------------
def DataReconstruction_PLM_y( U ):
    W  = Conserved2Primitive( U )
    WL = np.zeros_like( W )
    WR = np.zeros_like( W )

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # for i in range( Nx ):
    #     for v in range( 4 ):
    #         qL, qR = PLM_1D( W[i,:,v] )
    #         WL[i,:,v] = qL
    #         WR[i,:,v] = qR
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # 網格的左邊界和右邊界的斜率
    dqL = W[:, 1:-1, :] - W[:, :-2, :]
    dqR = W[:, 2:, :] - W[:, 1:-1, :]

    # van Leer
    slope_LR = dqL * dqR
    slope_limited = np.where(slope_LR > 0.0, 2.0 * slope_LR / (dqL + dqR), 0.0)

    # 用斜率和網格中心的值，算出左右邊界
    WL[:, 1:-1, :] = W[:, 1:-1, :] - 0.5 * slope_limited
    WR[:, 1:-1, :] = W[:, 1:-1, :] + 0.5 * slope_limited

    # 維持ghost zone變化量是0
    # 直接複製左右ghost zone的中心值到ghost zone的左右邊界
    WL[:, 0, :] = W[:, 0, :]
    WR[:, 0, :] = W[:, 0, :]
    WL[:, -1, :] = W[:, -1, :]
    WR[:, -1, :] = W[:, -1, :]

    WL[...,0] = np.maximum( WL[...,0], rho_floor )
    WR[...,0] = np.maximum( WR[...,0], rho_floor )
    WL[...,3] = np.maximum( WL[...,3], P_floor )
    WR[...,3] = np.maximum( WR[...,3], P_floor )

    return Primitive2Conserved(WL), Primitive2Conserved(WR)


#--------------------------------------------------------------------
# x-flux from conserved variables
# F(U)公式
#--------------------------------------------------------------------
def Flux_x( U ):
    flux = np.empty_like( U )

    rho = U[...,0]
    u   = U[...,1]/rho
    v   = U[...,2]/rho
    P   = ComputePressure( U )

    flux[...,0] = rho*u
    flux[...,1] = rho*u*u + P
    flux[...,2] = rho*u*v
    flux[...,3] = u*( U[...,3] + P )

    return flux


#--------------------------------------------------------------------
# y-flux from conserved variables
# G(U)公式
#--------------------------------------------------------------------
def Flux_y( U ):
    flux = np.empty_like( U )

    rho = U[...,0]
    u   = U[...,1]/rho
    v   = U[...,2]/rho
    P   = ComputePressure( U )

    flux[...,0] = rho*v
    flux[...,1] = rho*u*v
    flux[...,2] = rho*v*v + P
    flux[...,3] = v*( U[...,3] + P )

    return flux


#--------------------------------------------------------------------
# Rusanov flux in x direction
#--------------------------------------------------------------------
def Rusanov_x( UL, UR ):
    # 算左右邊界的flux
    FL = Flux_x( UL )
    FR = Flux_x( UR )

    # 算左右的物理量值
    WL = Conserved2Primitive( UL )
    WR = Conserved2Primitive( UR )

    # 算sound speed
    aL = np.sqrt( gamma*WL[...,3]/WL[...,0] )
    aR = np.sqrt( gamma*WR[...,3]/WR[...,0] )
    smax = np.maximum( np.abs(WL[...,1])+aL, np.abs(WR[...,1])+aR )

    # F = 1/2 (L + R) - S_max * (R - L) 中央差分含數值擴散項
    return 0.5*(FL+FR) - 0.5*smax[...,None]*(UR-UL)


#--------------------------------------------------------------------
# Rusanov flux in y direction
#--------------------------------------------------------------------
def Rusanov_y( UL, UR ):
    FL = Flux_y( UL )
    FR = Flux_y( UR )

    WL = Conserved2Primitive( UL )
    WR = Conserved2Primitive( UR )

    aL = np.sqrt( gamma*WL[...,3]/WL[...,0] )
    aR = np.sqrt( gamma*WR[...,3]/WR[...,0] )
    smax = np.maximum( np.abs(WL[...,2])+aL, np.abs(WR[...,2])+aR )

    return 0.5*(FL+FR) - 0.5*smax[...,None]*(UR-UL)


#--------------------------------------------------------------------
# Compute Right-Hand Side (RHS) for Method of Lines (MOL)
# ∂U/∂t + ∂ flux_x/∂x + ∂ flux_y/∂y = 0
# ∂U/∂t = - (∂ flux_x/∂x + ∂ flux_y/∂y) = RHS
#--------------------------------------------------------------------
def Compute_RHS( U ):
    BoundaryCondition( U )
    
    RHS = np.zeros_like( U )

    # x-direction interface fluxes
    Lx, Rx = DataReconstruction_PLM_x( U )
    flux_x = np.zeros_like( U )
    for i in range( nghost, Nx-nghost+1 ):
        # interface i-1/2 is between cell i-1 and i
        flux_x[i,:,:] = Rusanov_x( Rx[i-1,:,:], Lx[i,:,:] )

    # y-direction interface fluxes
    Ly, Ry = DataReconstruction_PLM_y( U )
    flux_y = np.zeros_like( U )
    for j in range( nghost, Ny-nghost+1 ):
        # interface j-1/2 is between cell j-1 and j
        flux_y[:,j,:] = Rusanov_y( Ry[:,j-1,:], Ly[:,j,:] )

    # U變化量 = 1/dx (流進流出量)
    # U後 = U前 + U變化量 --> RK4裡面算
    # update real cells
    RHS[nghost:Nx-nghost, nghost:Ny-nghost, :] -= (1.0/dx)*( flux_x[nghost+1:Nx-nghost+1, nghost:Ny-nghost, :]
                                                            -flux_x[nghost  :Nx-nghost,   nghost:Ny-nghost, :] )

    RHS[nghost:Nx-nghost, nghost:Ny-nghost, :] -= (1.0/dy)*( flux_y[nghost:Nx-nghost, nghost+1:Ny-nghost+1, :]
                                                            -flux_y[nghost:Nx-nghost, nghost  :Ny-nghost,   :] )
    
    # Self-gravity
    # RHS += Gravity_Source
    
    return RHS


#--------------------------------------------------------------------
# RK4 time integration step
#--------------------------------------------------------------------
def Update( U, dt ):
    # 算當前變化量
    k1 = Compute_RHS( U )
    
    # 算半步變化量
    k2 = Compute_RHS( U + 0.5 * dt * k1 )
    
    # 算修正半步變化量
    k3 = Compute_RHS( U + 0.5 * dt * k2 )
    
    # 算一步變化量
    k4 = Compute_RHS( U + dt * k3 )
    
    # 加weight算U後
    U += (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)
    
    # 加floor
    W = Conserved2Primitive( U )
    W[...,0] = np.maximum( W[...,0], rho_floor )
    W[...,3] = np.maximum( W[...,3], P_floor )
    U[:,:,:] = Primitive2Conserved( W )


#--------------------------------------------------------------------
# initialize Sedov blast wave
#--------------------------------------------------------------------
def SetInitialCondition():
    U = np.zeros( (Nx,Ny,4) )

    x = np.zeros( Nx )
    y = np.zeros( Ny )

    # 把正中心定義成(0, 0)
    # 設定成網格中心，再往左移動一半L
    for i in range( Nx ):
        x[i] = (i-nghost+0.5)*dx - 0.5*Lx

    for j in range( Ny ):
        y[j] = (j-nghost+0.5)*dy - 0.5*Ly

    # 給初始條件
    W = np.zeros( (Nx,Ny,4) )
    W[...,0] = rho0
    W[...,1] = 0.0
    W[...,2] = 0.0
    W[...,3] = P0

    # 算總共有幾個格點在blast範圍內
    # inject thermal energy into a small circular region
    count = 0
    for i in range( nghost, Nx-nghost ):
        for j in range( nghost, Ny-nghost ):
            r = np.sqrt( x[i]**2.0 + y[j]**2.0 )
            if r < r_blast:
                count += 1

    # P等於gamma-1乘以能量密度
    area = count*dx*dy
    P_add = (gamma-1.0)*E_blast/area

    # 在blast範圍內加壓力
    for i in range( nghost, Nx-nghost ):
        for j in range( nghost, Ny-nghost ):
            r = np.sqrt( x[i]**2.0 + y[j]**2.0 )
            if r < r_blast:
                W[i,j,3] += P_add

    U = Primitive2Conserved( W )
    BoundaryCondition( U )

    return U, x, y


#--------------------------------------------------------------------
# estimate shock radius roughly from density jump
# 兩倍rho0的地方設成shock radius
#--------------------------------------------------------------------
def EstimateShockRadius( U, x, y ):
    rho = U[...,0]
    Rmax = 0.0

    for i in range( nghost, Nx-nghost ):
        for j in range( nghost, Ny-nghost ):
            if rho[i,j] > 2.0*rho0:
                r = np.sqrt( x[i]**2.0 + y[j]**2.0 )
                Rmax = max( Rmax, r )

    return Rmax


#--------------------------------------------------------------------
# main
#--------------------------------------------------------------------
t = 0.0
U, x, y = SetInitialCondition()

step = 0

# store snapshots for animation
snapshots = []
snapshot_times = []
shock_radii = []
output_interval = 0.002
next_output_time = 0.0

while t < end_time:
    dt = ComputeTimestep( U[nghost:Nx-nghost,nghost:Ny-nghost,:], t )

    print( "step = %5d, t = %13.7e --> %13.7e, dt = %13.7e" % (step,t,t+dt,dt) )

    Update( U, dt )
    t += dt
    step += 1

    # save density snapshots for animation
    if t >= next_output_time or t >= end_time:
        rho_now = U[nghost:Nx-nghost,nghost:Ny-nghost,0].copy()
        Rshock  = EstimateShockRadius( U, x, y )

        snapshots.append( rho_now )
        snapshot_times.append( t )
        shock_radii.append( Rshock )

        print( "    save frame: t = %10.5e, shock radius ~ %10.5e" % (t,Rshock) )

        next_output_time += output_interval


#--------------------------------------------------------------------
# make animation: density evolution + wave front
#--------------------------------------------------------------------
import matplotlib.animation as animation

fig, ax = plt.subplots( dpi=140 )

# fixed color range makes the time evolution easier to compare
rho_min = min( np.amin(rho) for rho in snapshots )
rho_max = max( np.amax(rho) for rho in snapshots )

img = ax.imshow( np.log10(snapshots[0].T), origin='lower',
                 extent=[-0.5*Lx,0.5*Lx,-0.5*Ly,0.5*Ly],
                 aspect='equal', cmap='jet',
                 vmin=np.log10(rho_min), vmax=np.log10(rho_max) )

front = plt.Circle( (0.0,0.0), shock_radii[0],
                    fill=False, color='white', linewidth=1.5 )
ax.add_patch( front )

time_text = ax.text( 0.03, 0.95, '', color='white', transform=ax.transAxes )
radius_text = ax.text( 0.03, 0.89, '', color='white', transform=ax.transAxes )

ax.set_xlabel( 'x' )
ax.set_ylabel( 'y' )
ax.set_title( '2D Sedov blast wave density' )
plt.colorbar( img, ax=ax, label='log10 density' )


def AnimateFrame( n ):
    img.set_data( np.log10(snapshots[n].T) )
    front.set_radius( shock_radii[n] )
    time_text.set_text( 't = %.4f' % snapshot_times[n] )
    radius_text.set_text( 'wave front R = %.4f' % shock_radii[n] )
    return img, front, time_text, radius_text


ani = animation.FuncAnimation( fig, AnimateFrame,
                               frames=len(snapshots),
                               interval=80, blit=True )

# Save animation. GIF is easiest to open.
ani.save( 'sedov_density_wavefront.gif', writer='pillow', fps=12 )
print( 'Animation saved as sedov_density_wavefront.gif' )
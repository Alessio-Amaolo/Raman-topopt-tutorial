import numpy as np 
import jax.numpy as jnp 
from skimage.draw import disk

def init_domain(Nx, Ny, Mx, My, Mi, Npml, Npmlsep, nonpmlNx, nonpmlNy, circle):
    '''
    Tool to draw domains for the optimization problem.
    '''
    design_mask = np.zeros((Nx, Ny), dtype=bool) 

    if circle:
        rr1, cc1 = disk((Nx//2, Ny//2), Mx//2, shape=design_mask.shape)
        design_mask[rr1, cc1] = 1
        if Mi > 0:
            rr2, cc2 = disk((Nx//2, Ny//2), Mi, shape=design_mask.shape) 
            design_mask[rr2, cc2] = 0
    else:
        design_mask[Npml+Npmlsep:Npml+Npmlsep+Mx, Npml+Npmlsep:Npml+Npmlsep+My] = True
        if Mi > 0:
            design_mask[Npml+nonpmlNx//2 - Mi : Npml+nonpmlNx//2 + 1 + Mi,  Npml+nonpmlNy//2 - Mi : Npml+nonpmlNy//2 + 1 + Mi] = False
    ndof = np.sum(design_mask)
    return design_mask, ndof 

def getchi(rho, chi, chibkg, Nx, Ny, Dmask):
    rho = jnp.clip(rho, 0, 1)   
    chigrid = chibkg + (chi-chibkg) * rho
    bigchigrid = jnp.zeros((Nx, Ny), dtype=complex)
    bigchigrid = bigchigrid.at[Dmask].add(chigrid)
    return bigchigrid
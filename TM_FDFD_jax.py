import numpy as np
from functools import partial
import os
os.environ['JAX_PLATFORMS'] = 'cpu' # This code is a bit faster on GPU for sufficiently large systems, but this is not extensively tested for now. The designs presented in this paper were calculated on CPU.

import jax.numpy as jnp
import jax.experimental.sparse as jsp
from jax import jit

C_0 = 1.0 #dimensionless units
EPSILON_0 = 1.0
ETA_0 = 1.0

@jit
def _jt_diag(x):
    '''
    Create a sparse batched COO diagonal matrix from a 1D array
    '''
    N = x.shape[0]
    indices = jnp.stack((jnp.arange(N), jnp.arange(N)), axis=1)
    return jsp.BCOO((x, indices), shape=(N, N))

@jit
def _get_diagM_from_chigrid(omega, chigrid):
    '''
    Get the diagonal component of the Maxwell operator from the susceptibility grid. 
    '''
    return _jt_diag(-jnp.ravel(chigrid) * omega**2)

def get_TM_dipole_field(M0, wvlgth, dL, Nx, Ny, cx, cy, amp, chigrid=None):
    """
    Solves curl curl E - k^2 chi(r) E = i omega 1/dl/dl where 1/dl/dl is a representation of the delta function.
    Practically, gets the field of a TM dipole source at position (cx,cy) in a grid of size (Nx, Ny) and material distribution given by chigrid.
    
    Parameters
    ----------
    M0 : jax.experimental.BCOO sparse matrix
         Vacuum Maxwell operator from get_TM_MaxwellOp as a sparse matrix. It must be pre-computed with maxwell_operator.py
    wvlgth : real
        wavelength of interest.
    dL : float
        size of a single pixel of the finite difference grid.
    Nx : int
        Number of pixels along the x direciton.
    Ny : int
        Number of pixels along the y direction.
    cx : int
        x-coordinate of the dipole source.
    cy : int
        y-coordinate of the dipole source.
    chigrid : 2D numpy complex array, optional
        spatial distribution of material susceptibility. The default is None, corresponding to vacuum.
        If not vacuum, it will dictate 

    Returns
    -------
    Ez : 2D numpy complex array
         Field of the dipole source.
    """
    Qabs = np.inf # Quality of mode, can be passed as a parameter but just not generalized yet
    omega = (2*np.pi*C_0/wvlgth) * (1 + 1j/2/Qabs) 
    if not chigrid is None:
        M = _get_diagM_from_chigrid(omega, chigrid)
        A = jsp.BCSR.from_bcoo(jsp.BCOO.sum_duplicates(M+M0, nse=M0.nse)) # Add chi and vacuum component of the Maxwell operator
    else:
        A = jsp.BCSR.from_bcoo(M0) # Get a BCSR matrix from the BCOO matrix, which we need to pass to spsolve

    sourcegrid = jnp.zeros((Nx,Ny), dtype=complex)
    sourcegrid = sourcegrid.at[cx, cy].set(amp / dL**2)
    RHS = jnp.ravel(1j*omega*sourcegrid)
    
    tree, info = A.tree_flatten()
    data, indices, indptr = tree     
    Ez = jsp.linalg.spsolve(data, indices, indptr, RHS) 

    Ez = jnp.reshape(Ez, (Nx, Ny))
    return Ez
    
@partial(jit, static_argnums=(3, 4))
def get_TM_linesource_field(M0, wvlgth, dL, Nx, Ny, cx, amp, chigrid=None):
    '''
    The same thing as get_TM_dipole_field but it populates a row of dipoles with amplitude 1/dl. 
    '''
    Qabs = np.inf
    omega = (2*np.pi*C_0/wvlgth) * (1 + 1j/2/Qabs) 
    if not chigrid is None:
        M = _get_diagM_from_chigrid(omega, chigrid)
        A = jsp.BCSR.from_bcoo(jsp.BCOO.sum_duplicates(M+M0, nse=M0.nse))
    else:
        A = jsp.BCSR.from_bcoo(M0)
    
    sourcegrid = jnp.zeros((Nx,Ny), dtype=complex)
    sourcegrid = sourcegrid.at[cx, :].set(amp / dL)
    RHS = jnp.ravel(1j*omega*sourcegrid)
    
    tree, info = A.tree_flatten()
    data, indices, indptr = tree     
    Ez = jsp.linalg.spsolve(data, indices, indptr, RHS) 
    Ez = jnp.reshape(Ez, (Nx, Ny))
    return Ez

def get_TM_field(M0, wvlgth, Nx, Ny, sourcegrid, chigrid):
    '''
    The same thing as get_TM_dipole_field but the dipole distribution is passed as a parameter. 
    This is useful for the Raman active case, where the Raman source is anywhere the material is. 
    '''
    shape = (Nx,Ny)
    omega = (2*np.pi*C_0/wvlgth) * (1 + 1j/2/np.inf)    
    if not chigrid is None:
        M = _get_diagM_from_chigrid(omega, chigrid)
        A = jsp.BCSR.from_bcoo(jsp.BCOO.sum_duplicates(M+M0, nse=M0.nse))
    else:
        A = jsp.BCSR.from_bcoo(M0)

    RHS = 1j*omega*sourcegrid.flatten()
    
    tree, info = A.tree_flatten()
    data, indices, indptr = tree     
    Ez = jsp.linalg.spsolve(data, indices, indptr, RHS)
    Ez = jnp.reshape(Ez, (Nx, Ny))

    return Ez


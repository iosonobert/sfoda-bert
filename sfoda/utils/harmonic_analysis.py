"""
Harmonic analysis tools
(moved from  timeseries.py and uspectra.py to avoid confusion)

M.Rayson
UWA
Jan 2017
"""

import numpy as np
from scipy import linalg, optimize

from datetime import datetime
from . import othertime
import sfoda.tides.cons as tcons

import pdb

def _build_lsq_A(t,frq):
        """
        Construct matrix A
        """
        nt=t.shape[0]
        nf=frq.shape[0]
        nff=nf*2+1
        A=np.ones((nt,nff))
        for ff in range(0,nf):
            A[:,ff*2+1]=np.cos(frq[ff]*t)
            A[:,ff*2+2]=np.sin(frq[ff]*t)
            
        return A
    

def harmonic_fit_array(X, t, frq, axis=0):
    """
    Least-squares harmonic fit on an array

    X - vector [Nt] or array [Nt, (size)]
    t - floating point time vector [Nt]
    frq - frequency vector [Ncon]
        
    Return a single array, R,  where
        R[:,0] - mean
        R[:,1::2] - Real amplitude
        R[:,2::2] - Imag amplitude

    """
    # Check the inputs
    frq = np.array(frq)
    Nfrq = frq.shape[0]
    
    # Reshape the array sizes
    X = X.swapaxes(0, axis)
    sz = X.shape
    lenX = int(np.prod(sz[1:]))
    
    if not len(t) == sz[0]:
        raise 'length of t (%d) must equal dimension of X (%s)'%(len(t),sz[0])
    
    # Need to reshape so rows contain time and other dimensions are along the columns
    X = np.reshape(X,(sz[0],lenX))
    
    # Least-squares matrix approach
    A = _build_lsq_A(t,frq)

    # Do the least-squares fit...
    b = np.linalg.lstsq(A,X)
   
    # reshape the array back to its original dimensions
    output = np.reshape(b[0],(2*Nfrq+1,)+sz[1:])
    
    # Output back along the original axis
    return output.swapaxes(axis,0)
    

def harmonic_fit(dtime, X, frq, mask=None, axis=0, phsbase=None):
    """
    Least-squares harmonic fit on an array, X, with frequencies, frq. 
    
    X - vector [Nt] or array [Nt, (size)]
    dtime - datetime-like vector [Nt]
    frq - vector [Ncon]
    mask - array [(size non-time X)]
    phsbase - phase offset
    
    where, dimension with Nt should correspond to axis = axis.
    """

    if phsbase is None:
        phsbase = datetime(1900,1,1)

    ###
    # Convert the dtime to seconds since
    t = othertime.SecondsSince(dtime, basetime=phsbase)
    #t = np.asarray(t)
    
    # Reshape the array sizes
    X = X.swapaxes(0, axis)
    sz = X.shape
    lenX = int(np.prod(sz[1:]))
    
    if not len(t) == sz[0]:
        raise 'length of t (%d) must equal dimension of X (%s)'%(len(t),sz[0])
    
    X = np.reshape(X,(sz[0],lenX))
    
    if not mask is None and np.any(mask):
         mask = mask.swapaxes(0, axis)
         #mask = np.reshape(mask,(sz[0],lenX))
    #    X = X.ravel()
    #else:
    #    mask = False
    
    frq = np.array(frq)
    Nfrq = frq.shape[0]
    
    def buildA(t,frq):
        """
        Construct matrix A
        """
        nt=t.shape[0]
        nf=frq.shape[0]
        nff=nf*2+1
        A=np.ones((nt,nff))
        for ff in range(0,nf):
            A[:,ff*2+1]=np.cos(frq[ff]*t)
            A[:,ff*2+2]=np.sin(frq[ff]*t)
            
        return A
    
    def lstsqnumpy(A,y):    
        """    
        Solve the least square problem
        
        Return:
            the complex amplitude 
            the mean
        """
        N=A.shape[1]
        b = np.linalg.lstsq(A,y)
        A = b[0][1::2]
        B = b[0][2::2]
        
        return A+1j*B, b[0][0::N]

    def lstsqscipy(A,y):    
        """    
        TESTING ONLY...

        Uses scipy's least_squares function that uses
        non-least-squares methods i.e. MLE
        """
        if type(y) == np.ma.MaskedArray:
            y=y.data

        N=A.shape[1]
        def fun(x0):
            err = y - A.dot(x0[:,np.newaxis])
            return err.ravel()

        b = linalg.lstsq(A,y)
        x0=b[0].ravel()

        # Robust regression
        b = optimize.least_squares(fun, x0, loss='soft_l1', f_scale=0.2)

        A = b['x'][1::2]
        B = b['x'][2::2]
        
        return A+1j*B, b['x'][0::N]
    
    
    def phsamp(C):
        return np.abs(C), np.angle(C)
        
   
    # Non-vectorized method (~20x slower)
    # Use this on a masked array
    if np.any(mask):
        Amp = np.zeros((Nfrq,lenX))
        Phs = np.zeros((Nfrq,lenX))
        C0 = np.zeros((lenX,))
        for ii in range(0,lenX):    
            #idx = mask[ii,:]==False
            idx = mask[:,ii]==False
            if not np.any(idx):
                continue
            A = buildA(t[idx],frq)
            C, C0[ii] = lstsqnumpy(A,X[idx,ii])
            # Calculate the phase and amplitude
            am, ph= phsamp(C)
            Amp[:,ii] = am
            Phs[:,ii] = ph
    else:
        # Least-squares matrix approach
        A = buildA(t,frq)
        C, C0 = lstsqnumpy(A,X) # This works on all columns of X!!
        Amp, Phs= phsamp(C)

    ###
    # !! Do not need to do this as the time is now in units of seconds since phsbase
    # Reference the phase to some time
    #if not phsbase is None:
    #    base = othertime.SecondsSince(phsbase)
    #    phsoff = phase_offset(frq,t[0],base)
    #    phsoff = np.repeat(phsoff.reshape((phsoff.shape[0],1)),lenX,axis=1)
    #    Phs = np.mod(Phs+phsoff,2*np.pi)
    #    pdb.set_trace()
    # !!
    
            
    
    # reshape the array
    Amp = np.reshape(Amp,(Nfrq,)+sz[1:])
    Phs = np.reshape(Phs,(Nfrq,)+sz[1:])
    C0 = np.reshape(C0,sz[1:])
    
    # Output back along the original axis
    # Amplitude, phase, mean
    return Amp.swapaxes(axis,0), Phs.swapaxes(axis,0), C0#C0.swapaxes(axis,0)
    
def phase_offset(frq,start,base):
        """
        Compute a phase offset for a given fruequency
        """
        
        if isinstance(start, datetime):
            dx = start - base
            dx = dx.total_seconds()
        elif isinstance(start, np.datetime64):
            dx = (start - base)/np.timedelta64(1,'s')
        else:
            dx = start - base
        
        return np.mod(dx*np.array(frq),2*np.pi)

def phase_offset_old(frq,start,base):
        """
        Compute a phase offset for a given fruequency
        """
        
        if type(start)==datetime:
            dx = start - base
            dx = dx.total_seconds()
        else:
            dx = start -base
        
        return np.mod(dx*np.array(frq),2*np.pi)
 
def harmonic_signal(time, amp, phs, cmean, omega, phsbase=None, axis=-1):
    """
    Reconstruct a harmonic signal for any size array

    (Assumes time is along the first axis for now)
    """
    nt = time.shape[0]
    # Initialise the output arrays
    if amp.ndim>1:
        sz = amp.shape[:1]
        h=np.ones((nt,)+sz)*cmean[np.newaxis,...]
    else:
        h=np.ones((nt,))*cmean[np.newaxis,...]

    #nx = np.prod(sz)
    
    # Rebuild the time series
    #tsec=TS_harm.tsec - TS_harm.tsec[0]
    if phsbase is None:
        phsbase=datetime(1900,1,1)
        #phsbase=time[0]

    tsec = othertime.SecondsSince(time,basetime=phsbase)

    for nn,om in enumerate(omega):
        if amp.ndim>1:
            h[:] += amp[np.newaxis,...,nn] *\
                np.cos(om*tsec[:,np.newaxis] - phs[np.newaxis,...,nn])
        else:
            h[:] += amp[nn] *\
                np.cos(om*tsec[:] - phs[nn])
            
    return h

def phsamp2complex(phs,amp):
    """
    Convert polar phase-amplitude to complex form
    """
    return amp*np.cos(phs) + 1j*amp*np.sin(phs)

def complex2phsamp(C):
    """
    Convert complex amplitude to phase and amplitude
    """
    return np.angle(C), np.abs(C)
 

def getTideFreq(Fin=None):
    """
    Function removed. 

    Could pass
    """

    freq, fout = tcons.getTideFreq(Fin=Fin)

    return freq, fout
    # raise(Exception('Function moved to sfoda.tides.cons.getTideFreq - could wrap this rather than raise an error.'))
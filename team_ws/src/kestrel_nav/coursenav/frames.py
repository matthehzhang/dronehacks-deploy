"""Rigid planar transforms for the fixed VIO launch fixture."""
import numpy as np

def compose(T,p):
    T,p=np.asarray(T),np.asarray(p)
    c,s=np.cos(T[...,2]),np.sin(T[...,2])
    return np.stack([T[...,0]+c*p[...,0]-s*p[...,1],T[...,1]+s*p[...,0]+c*p[...,1],
                     (T[...,2]+p[...,2]+np.pi)%(2*np.pi)-np.pi],axis=-1)

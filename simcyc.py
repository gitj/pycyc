"""
Library for simulating cyclic spectra
"""
import pycyc
reload(pycyc)
import numpy as np
from matplotlib import pyplot as plt
import os

def runLoopTest(tau=650.0,nlag=4096,bw=1.0,noise=0,tolfact=10, niter=5,maxfun=1000,maxinitharm=20):
    filename = 'itersim1643_tau%.2fus_noise%.2f' % (tau,noise)
    blah,fbase = os.path.split(filename)
    plotdir = os.path.join(os.path.abspath(os.path.curdir),('%s_plots' % fbase))
    if not os.path.exists(plotdir):
        os.mkdir(plotdir)
    
    ht0 = makeht(tau=tau,nlag=nlag,bw=bw)
    np.save(os.path.join(plotdir,'ht0'), ht0)
    
    CS = initSim(ht0,bw=bw,noise=noise)
    CS.filename = filename
    
    CS.initProfile(maxinitharm=maxinitharm)
    
    np.savez(os.path.join(plotdir,'start'),cs=CS.cs,pp_ref=CS.pp_ref,
             pp_start = CS.data.mean(2).squeeze())
    
    
    for k in range(1,niter+1):
        CS.pp_ref = CS.pp_int.copy()
        CS.loop(make_plots=True,plotdir=plotdir,hf_prev=CS.hf_prev,maxfun=maxfun)
        np.savez(os.path.join(plotdir,('iter%d' % k)),
                 pp_ref=CS.pp_ref,pp_int=CS.pp_int, hf = CS.hf_prev,
                 ht = pycyc.freq2time(CS.hf_prev), csm = CS.modelCS(hf=CS.hf_prev),
                 )

def runTest(tau,nlag,bw=1.0,noise=0.0,tolfact=10,initprof=False):
    CS = initSim(makeht(tau=tau,nlag=nlag,bw=bw),bw=bw,noise=noise)
    CS.filename = 'sim1643_tau%.2fus_noise%.2f' % (tau,noise)
    if initprof:
        CS.initProfile()
    CS.loop(make_plots=True,tolfact=tolfact)
    return CS

def initSim(ht,prof,ref_freq,bw,rf = None,filename= None,source='fake',noise = None):
    if rf is None:
        rf = np.abs(bw)/2.0
    if ht.ndim == 1:
        ht = ht[None,:]
    CS = pycyc.CyclicSolver()
    CS.ht0 = ht
    if filename:
        CS.filename = filename
    else:
        CS.filename = 'sim%s_%.1fMHz_%.1fms' % (source,bw,1000/ref_freq)   
    CS.nchan = ht.shape[1]
    CS.nlag = CS.nchan
    CS.nphase = prof.shape[0]
    CS.nharm = CS.nphase/2 + 1
    CS.source = source
    CS.nspec = ht.shape[0]
    CS.dynamic_spectrum = np.zeros((CS.nspec,CS.nchan))
    CS.optimized_filters = np.zeros((CS.nspec,CS.nchan),dtype='complex')
    CS.nopt = 0
    
    CS.ref_freq = ref_freq
    CS.bw = bw
    CS.ref_phase = 0
    CS.rf = rf
    
    CS.hf_prev = np.ones((CS.nchan,),dtype='complex')
    CS.pp_int = np.zeros((CS.nphase)) #intrinsic profile
    
    CS.pp_ref = prof.copy()
    
    CS.ph_ref = pycyc.phase2harm(CS.pp_ref)
    CS.ph_ref = pycyc.normalize_profile(CS.ph_ref)
    CS.ph_ref[0] = 0
    CS.s0 = CS.ph_ref.copy()
    
    CS.data = np.empty((CS.nspec,1,CS.nchan,CS.nphase), dtype='complex')
    for k in range(CS.nspec):
        CS.data[k,0,:,:] = pycyc.cs2ps(CS.modelCS(ht[k,:]))
    if noise is not None:
        fact = CS.data.std()*noise
        CS.data += (np.random.randn(CS.data.shape[2],CS.data.shape[3]).astype('complex')+1j*np.random.randn(CS.data.shape[2],CS.data.shape[3]))*fact
    return CS

def makeProfile(scale,nbins=1024):
    """
    Construct a fake profile with exponentially decaying harmonics
    
    scale : harmonic at which amplitude = exp(-1)
    nbins : number of bins in the profile
    """
    ph = np.exp(-np.arange(nbins/2 + 1)/(1.0*scale))
    ph[0] = 0.0
    pp = np.fft.irfft(ph)
    return pp
    
def makeht(tau=1.0,nlag=1024,scale = 0,bw=1.0):
    t = np.arange(nlag)/bw
    ht = (np.random.randn(nlag)+1j*np.random.randn(nlag))*np.sqrt(np.exp(-t/tau))
    ht = ht/np.abs(ht[0])
    ht[0] += scale
    ht[ht.shape[0]/2:] = 0
    return ht

def makehtOne(tau=1.0,val = 0.1, nlag=1024,bw=1.0):
    ht = np.zeros((nlag,),dtype='complex')
    ht[0] = 1.0
    ht[int(tau*bw)] = val
    return ht
    

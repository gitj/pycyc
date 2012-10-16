import pycyc
reload(pycyc)
import numpy as np
from matplotlib import pyplot as plt
import os


std_prof = np.genfromtxt('/psr/gjones/1643-1224.Rcvr_800.std')[:,1]

ref_freq = 1/4.62e-3
bw = 10.119529411764706
rf = 430

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
    
    np.savez(os.path.join(plotdir,'start'),cs=CS.cs,pp_ref=CS.pp_ref)
    
    
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

def initSim(ht,prof=std_prof,ref_freq=ref_freq,bw=bw,noise = None):
    CS = pycyc.CyclicSolver()
    CS.ht0 = ht
    CS.filename = 'sim1643'
    CS.nchan = ht.shape[0]
    CS.nlag = CS.nchan
    CS.nphase = prof.shape[0]
    CS.nharm = CS.nphase/2 + 1
    CS.source = 'J1643-1224sim'
    CS.nspec = 1
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
    
    csm = CS.modelCS(ht)
    CS.data = pycyc.cs2ps(csm)[None,None,:,:]
    CS.cs = csm.copy()
    if noise is not None:
        fact = CS.data.std()*noise
        CS.data += (np.random.randn(CS.data.shape[2],CS.data.shape[3])+1j*np.random.randn(CS.data.shape[2],CS.data.shape[3]))*fact
    return CS

def makeht(tau=1.0,nlag=1024,scale = 0,bw=bw):
    t = np.arange(nlag)/bw
    ht = (np.random.randn(nlag)+1j*np.random.randn(nlag))*np.sqrt(np.exp(-t/tau))
    ht = ht/np.abs(ht[0])
    ht[0] += scale
    ht[ht.shape[0]/2:] = 0
    return ht

def makehtOne(tau=1.0,val = 0.1, nlag=1024):
    ht = np.zeros((nlag,),dtype='complex')
    ht[0] = 1.0
    ht[int(tau*bw)] = val
    return ht

def grabData():
    import glob
    dirs = glob.glob('/home/gjones/workspace/pycyc/itersim*')
    for d in dirs:
        blah,fbase = os.path.split(d)
        outdir = os.path.join('/home/gjones/workspace/pycyc/data',fbase)
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        try:
            os.link(os.path.join(d,'ht0.npy'), os.path.join(outdir,'ht0.npy'))
            os.link(os.path.join(d,'start.npz'), os.path.join(outdir,'start.npz'))  
        except:
            pass
        npzs = glob.glob(os.path.join(d,'iter*.npz'))
        for npz in npzs:
            data = np.load(npz)
            newdata = {}
            for k in data.keys():
                if k != 'csm':
                    newdata[k] = data[k]
            blah,fbase = os.path.split(npz)
            np.savez(os.path.join(outdir,fbase),**newdata)
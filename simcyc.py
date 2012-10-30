import pycyc
reload(pycyc)
import numpy as np
from matplotlib import pyplot as plt
import os


#std_prof = np.genfromtxt('1643-1224.Rcvr_800.std')[:,1]
std_prof = np.genfromtxt('/home/gjones/Average_profile_tau_5_jp_0_noise_0.5_rebinned')[:,1]

#std_12 = np.genfromtxt('1643-1224.Rcvr1_2.std')[:,1]

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
        CS.data += (np.random.randn(CS.data.shape[2],CS.data.shape[3]).astype('complex')+1j*np.random.randn(CS.data.shape[2],CS.data.shape[3]))*fact
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
    
def plotGbt(dirname='D:/data/itersim1643_tau650.00us_noise8.01_plots'):
#def plotGbt(dirname='D:/data/itersim1643_tau650.00us_noise16.00_plots'):
    f,axs = plt.subplots(nrows=4,sharex=True,figsize=(3,6))
    f.subplots_adjust(left=0.175,bottom=0.125)
    ht0 = np.load(os.path.join(dirname,'ht0.npy'))
    pmeas = np.load('profile_noise8.npy')
    start = np.load(os.path.join(dirname,'start.npz'))
    iter5 = np.load(os.path.join(dirname,'iter5.npz'))
    tp = 1000*np.linspace(0,1/ref_freq,pmeas.shape[0],endpoint=False)
    th = 1000*np.arange(ht0.shape[0])/1e6
    fr = np.linspace(0,1e3,ht0.shape[0])
    axs[0].plot(tp,np.roll(pycyc.normalize_pp(std_prof),1024),'k',label='Original profile')    
    axs[0].set_ylim(-2,10)
    axs[2].plot(tp,np.roll(pycyc.normalize_pp(pmeas),1024),'k',label='Simulated scattered profile')
    axs[2].set_ylim(-2,10)
    axs[1].plot(th,np.roll(np.abs(ht0)**2,ht0.shape[0]/2),'k',label='Simulated $|h(t)|^2$')    
    max0 = std_prof.argmax()
    max1 = iter5['pp_int'].argmax()
    
    axs[3].plot(tp,np.roll(pycyc.normalize_pp(iter5['pp_int']),1024),'k',label='Calculated intrinsic profile')
    axs[3].plot(tp,np.roll(pycyc.normalize_pp(std_prof),1024-max0+max1),'r',lw=1.5,label='Original profile')
    axs[3].set_ylim(-2,10)
    max0 = pmeas.argmax()
    max1 = start['pp_ref'].argmax()    
    axs[2].plot(tp,np.roll(pycyc.normalize_pp(start['pp_ref']),1024-max0+max1),'r',linewidth=1,label='Initial guess')
#    axs[4].plot(th,np.roll(np.abs(iter5['ht'])**2,iter5['ht'].shape[0]/2))
    axs[0].set_xlim(0,1000/ref_freq)
    
    for k in range(4):
        axs[k].yaxis.set_major_locator(plt.MaxNLocator(4))
        
    axs[3].xaxis.set_major_locator(plt.MaxNLocator(5))
    axs[3].set_xlabel('Time (ms)')

    f.savefig('panel1.pdf')
#    axs[0].legend(prop=dict(size='x-small'),loc='upper left')
#    axs[1].legend(prop=dict(size='x-small'),loc='upper left')
#    axs[2].legend(prop=dict(size='x-small'),loc='upper left')
#    axs[3].legend(prop=dict(size='x-small'),loc='upper left')
    
    f,axs = plt.subplots(nrows=2,sharex=True,figsize=(3,6))
    f.subplots_adjust(left=0.175,bottom=0.125)
    hf2 = pycyc.time2freq(np.roll(iter5['ht'],290))/86.7
    hf0 = pycyc.time2freq(ht0)
    ht2 = np.roll(iter5['ht'],290+2048)/86.7
    ht0r = np.roll(ht0,2048)
    
    axs[0].plot(th,np.abs(ht0r)**2,'k')
    axs[1].plot(th,np.abs(ht2)**2,'k')    
    axs[0].set_xlim(0,th.max())
    axs[0].text(0.05,0.95,r'$|h(t)|^2$',transform=axs[0].transAxes,ha='left',va='top',bbox=dict(fc='w'))
    axs[1].text(0.05,0.95,r'$|\hat{h}(t)|^2$',transform=axs[1].transAxes,ha='left',va='top',bbox=dict(fc='w'))

    
    for k in range(2):
        axs[k].yaxis.set_major_locator(plt.MaxNLocator(4))
        axs[k].set_ylim(0,24)
    axs[1].xaxis.set_major_locator(plt.MaxNLocator(5))
    axs[1].set_xlabel('Time (ms)')
    
    f.savefig('panel2.pdf')
    
    f,axs = plt.subplots(nrows=2,sharex=True,figsize=(3,6))
    f.subplots_adjust(left=0.175,bottom=0.125)
    axs[0].plot(fr,np.abs(hf2/hf0),'k')
    axs[0].text(0.05,0.95,r'$\left|\frac{\hat{H}(f)}{H(f)}\right|$',transform=axs[0].transAxes,ha='left',va='top',bbox=dict(fc='w'))
    axs[1].plot(fr,np.angle(hf2*np.conj(hf0)),'k')    
    axs[1].text(0.05,0.95,r'$\angle\left(\frac{\hat{H}(f)}{H(f)}\right)$',transform=axs[1].transAxes,ha='left',va='top',bbox=dict(fc='w'))
    axs[0].set_xlim(300,380)
    axs[0].set_ylim(0,5)
    axs[1].xaxis.set_major_locator(plt.MaxNLocator(5))
    axs[1].set_ylabel('radians')
    axs[1].set_xlabel('Relative RF frequency (kHz)')
    
    axs[0].yaxis.set_major_locator(plt.MaxNLocator(5))
    axs[1].yaxis.set_major_locator(plt.MaxNLocator(4))


    f.savefig('panel3.pdf')


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
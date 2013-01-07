# -*- coding: utf-8 -*-
"""
Created on Sat Jan 05 15:08:30 2013

@author: Glenn
"""

import simcyc
import pycyc
import numpy as np

from matplotlib import pyplot as plt

pp = np.fft.fftshift(simcyc.makeProfile(20))
ph = pycyc.phase2harm(pp)
"""
f = plt.figure()
ax = f.add_subplot(111)


ax.semilogx(20*np.log10(np.abs(ph)/np.abs(ph).max()))
ax.set_ylim(-55,5)
ax.set_xlim(1,300)
ax.grid(True)
ax.set_ylabel('20*log10(|p|)')
ax.set_xlabel('Harmonic')

f = plt.figure()
ax = f.add_subplot(111)

ax.plot(pp)
"""

ht0 = simcyc.makehtOne(tau=0,val=1.0)
CS = simcyc.initSim(ht0,pp,ref_freq=250.0,bw=1.0)

cs0 = CS.modelCS(ht0)

def plotCS(cs):
    f = plt.figure()
    ax = f.add_subplot(221)
    ps = np.fft.irfft(cs,axis=1)
    ax.imshow(np.real(ps),aspect='auto')
    ax.set_title('ps')
    plt.setp(ax.xaxis.get_ticklabels(),visible=False)
    ax.set_ylabel('frequency')
    
    ax = f.add_subplot(222)
    ax.imshow(np.real(cs),aspect='auto')
    ax.set_xlim(1,50)
    ax.set_title('cs')
    plt.setp(ax.xaxis.get_ticklabels(),visible=False)
    plt.setp(ax.yaxis.get_ticklabels(),visible=False)
    
    ax = f.add_subplot(223)
    cc0 = np.fft.irfft(np.fft.rfft(cs,axis=0),axis=1)
    ax.imshow(np.real(cc0),aspect='auto')
    ax.set_title('cc')
    ax.set_ylim(0,20)
    ax.set_ylabel('lag')
    ax.set_xlabel('phase')
    
    ax = f.add_subplot(224)
    ch = np.fft.rfft(cs,axis=0)
    ax.imshow(np.real(ch),aspect='auto')
    ax.set_xlim(1,50)
    ax.set_title('ch')
    ax.set_ylim(0,20)
    plt.setp(ax.yaxis.get_ticklabels(),visible=False)
    ax.set_xlabel('harmonic')
    f.tight_layout()
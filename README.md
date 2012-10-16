pycyc
=====

Python version of CyclicModelling code originally by Paul Demorest and Mark Walker

See https://github.com/demorest/Cyclic-Modelling for the original C version



This is designed to be a library of python functions to be used interactively with ipython
or in other python scripts. However, for demonstration purposes you can run this as a stand-alone script:

python2.7 pycyc.py input_cs_file.ar # This will generate an initial profile from the data itself

python2.7 pycyc.py input_cs_file.ar some_profile.txt # This will use the profile in some_profile.txt

The majority of these routines have been checked against the original Cyclic-Modelling code
and produce identical results to floating point accuracy. The results of the optimization may
not be quite as identical since Cyclic-Modelling uses the nlopt implementation of the L_BFGS solver
while this code uses scipy.optimize.fmin_l_bfgs_b

Here's an example of how I use this on kermit.

$ ipython -pylab

    import pycyc
    
    CS = pycyc.CyclicSolver(filename='/psr/gjones/2011-09-19-21:50:00.ar') # some 1713 data at 430 MHz Nipuni processed
    
    CS.initProfile(loadFile='/psr/gjones/pp_1713.npy') # start with a nice precomputed profile.
    # Note profile can be in .txt (filter_profile) format or .npy numpy.save format.
    
    # have a look at the profile:
    plot(CS.pp_int)
    
    CS.data.shape
    Out: (1, 2, 256, 512) # 1 subintegration, 2 polarizations, 256 freq channels, 512 phase bins
    
    CS.loop(isub = 0, make_plots=True,ipol=0,tolfact=20) # run the optimzation
    # Note this does the "inner loop": sets up the non-linear optimization and runs it
    # the next step will be to build the "outer loop" which uses the new guess at the intrinsic profile
    # to reoptimize the IRF. This isn't yet implemented but the machinary is all there.
    #
    # Running this with make_plots will create a <filename>_plots subdirectory with plots of various
    # data products. The plots are made each time the objective function is evaluated. Note that there
    # can be more than one objective function evaluation per solver iteration.
    #
    # tolfact can be used to reduce the stringency of the stopping criterion. This seems to be particularly useful
    # to avoid overfitting to the noise in real data
    
    # after the optimization runs, have a look at the CS data
    clf()
    imshow((np.abs(CS.cs))[:,1:],aspect='auto')
    
    # have a look at the IRF
    ht = pycyc.freq2time(CS.hf_prev)
    t = np.arange(ht.shape[0]) / CS.bw # this will be time in microseconds since CS.bw is in MHz
    subplot(211)
    plot(t,abs(ht))
    
    # compute the model CS and have a look
    
    csm = CS.modelCS(ht)
    subplot(212)
    imshow(np.log10(np.abs(csm)),aspect='auto')
    colorbar()
    
    # Now examine the effect of zeroing the "noisy" parts of the IRF
    
    ht2 = ht[:] #copy ht
    ht2[:114] = 0
    ht2[143:] = 0
    csm2 = CS.modelCS(ht2)
    figure()
    subplot(211)
    plot(t,abs(ht2))
    subplot(212)
    imshow(np.log10(np.abs(csm2)),aspect='auto')
    colorbar()
    
    
    # Try a bounded optimizaton, constraining h(t) to have support over a limited range
    # the two parameters are:
    # maxneg : int or None
    # The number of samples before the delta function which are allowed to be nonzero
    # This value must be given to turn on bounded optimization
    # maxlen : int or None
    # The maximum length of the impulse response in samples
    
    # e.g. suppose we want to limit the IRF to only have support from -1 to +10 us and CS.bw ~ 10 MHz
    # maxneg = int(1e-6 * 10e6) = 10
    # maxlen = int((1e-6 + 10e-6) * 10e6) = 110
    
    CS.loop(make_plots=True, tolfact=10, maxneg=10, maxlen = 110)
    

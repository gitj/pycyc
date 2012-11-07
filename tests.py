"""
tests.py - test routines for pycyc package

These test routines use an executable called 'test_merit' which can be built using `make tests` in the Cyclic-Modelling source directory.
This program needs to be able to import pycyc, so pycyc.py needs to be in your current directory or in your PYTHONPATH
You can run this program as:

python2.7 tests.py

Edit path below to point to where test_merit is located

test_merit generates random test vectors and tests the most important functions in the Cyclic-Modelling package. It writes out the
test vectors and the results in simple text files. This program then ingests those text files and compares them against the same
tests run on the same random test vectors and checks that the results are accurate to within numerical precision.

Currently the thresholds for passing are semi arbitrary; they are all fairly stringent and the tests pass.
"""
import pycyc
import numpy as np
import os

"""
    /*
     * PHASE 1:
     * First load hf with random data, and write it to hf.tst
     * then load pp with random data and write it to pp.tst
     * Compute ph from pp and write to ph.tst
     * Recompute pp_ref from ph to make sure the result is identical with original pp. write pp_ref to pp2pp.tst
     * using ph and hf, compute a model cs
     * Do the same for hf_prev, pp_ref, ph_ref, and cs2
     *
     * PHASE 2:
     * Now compute best fit ph_ref given hf and cs
     * Write reconstructed ph_ref to ph_reconstructed.tst
     * Then compare reconstructed ph_ref to original ph using profile_ms_difference
     *
     * PHASE 3:
     * Next compute ht from hf
     * Find rindex as max filter time
     * Transfer ht to parameter array x
     * Compute lag merit and gradient from ht, cs, and ph. Gradient stored in grad1.tst
     * Since ht came directly from hf used to generate cs, this should give good (low) merit and small gradient
     *
     * Now compute ht from hf_prev (a random transfer function unrelated to cs)
     * Compute lag merit and gradient from ht, cs, and ph. Gradient stored in grad2.tst
     * Since ht came from unrelated hf_prev, this should give poor (high) merit and large gradient
     *
     */
"""

###
# Change this to match the location where you built Cyclic-Modelling
# Remember you need to `make tests` to generate the test_merit executable
###
# Run the Cyclic-Modelling test routines
print "running test_merit..."
os.system('/home/gjones/workspace/Cyclic-Modelling/test_merit')

# the following assumes the results of test_merit will end up in your current directory.
dirname = os.path.abspath('.')

print "\n\n Now checkign results..."
# Gather the test results
fh = open(os.path.join(dirname,'params.tst'),'r')
params = eval(fh.read())
fh.close()

bw = params['bw']
ref_freq = params['ref_freq']

hf = np.loadtxt(os.path.join(dirname,'hf.tst'), delimiter=',').view('complex').squeeze()
hf_prev = np.loadtxt(os.path.join(dirname,'hf_prev.tst'), delimiter=',').view('complex').squeeze()
pp = np.loadtxt(os.path.join(dirname,'pp.tst'), delimiter=',')
pp2 = np.loadtxt(os.path.join(dirname,'pp2pp.tst'), delimiter=',')
pp_ref = np.loadtxt(os.path.join(dirname,'pp_ref.tst'), delimiter=',')
ph = np.loadtxt(os.path.join(dirname,'ph.tst'), delimiter=',').view('complex').squeeze()
ph_ref = np.loadtxt(os.path.join(dirname,'ph_ref.tst'), delimiter=',').view('complex').squeeze()
ph_recon = np.loadtxt(os.path.join(dirname,'ph_reconstructed.tst'), delimiter=',').view('complex').squeeze()
grad1 = np.loadtxt(os.path.join(dirname,'grad1.tst'), delimiter=',')
grad2 = np.loadtxt(os.path.join(dirname,'grad2.tst'), delimiter=',')

ntests = 0
nfail = 0

ntests += 1
phpperr = np.abs(ph - pycyc.phase2harm(pp)).mean()
print "(Cyclic-Modelling vs pycyc) phase2harm error:", phpperr
if phpperr > 1e-7:
    print "phase2harm does not agree"
    nfail += 1

ntests += 1
pp2pperr = np.abs(pp-pp2).mean()
print "(Cyclic-Modelling) test of harm2phase(phase2harm) == identity: ", pp2pperr
if pp2pperr > 1e-7:
    print "error in cyclic_utils.c: harm2phase is not exact inverse of phase2harm"
    nfail +=1

ntests += 1
phreconerr = np.abs(ph[1:]-ph_recon[1:]).mean() #skip zeroth harmonic since it's zeroed in ph
print "(Cyclic-Modelling) test if ph reconstructed from model cs and given hf matches original ph: ", phreconerr
if phreconerr > 1e-7:
    print "reconstructed ph does not match original ph"
    nfail +=1
    
cs, a, b, c = pycyc.make_model_cs(hf, ph, bw, ref_freq)

ph_pyrecon = pycyc.optimize_profile(cs, hf, bw, ref_freq)

ntests += 1
phreconerr = np.abs(ph[1:]-ph_pyrecon[1:]).mean() #skip zeroth harmonic since it's zeroed in ph
print "(pycyc) test if ph reconstructed from model cs and given hf matches original ph: ", phreconerr
if phreconerr > 1e-7:
    print "reconstructed ph does not match original ph"
    nfail +=1

CS = pycyc.CyclicSolver()
ht = pycyc.freq2time(hf)
CS.rindex = np.abs(ht).argmax()
print "pycyc rindex:",CS.rindex, "Cyclic-Modelling rindex:", params['rindex1']
ntests += 1
if CS.rindex != params['rindex1']:
    print "rindex1 does not agree"
    nfail +=1
CS.s0 = ph
CS.bw= bw
CS.ref_freq = ref_freq
CS.cs = cs
CS.objval = []
CS.nlag = ht.shape[0]
CS.nchan = ht.shape[0]
CS.iprint = 1
CS.make_plots = False
CS.niter = 0

x = pycyc.get_params(ht, CS.rindex)
pymerit,pygrad = pycyc.cyclic_merit_lag(x,CS)
print "pycyc merit:", pymerit, "Cyclic-Modelling merit:", params['merit1']
ntests += 1
if np.abs(pymerit - params['merit1']) > 1e-5:
    print "merits disagree"
    nfail+=1
    
ntests += 1    
grad1err = np.abs(grad1-pygrad).mean()
print "grad1 error:", grad1err
if grad1err > 1e-8:
    print "grad1 disagree"
    nfail += 1

ht2 = pycyc.freq2time(hf_prev)
CS.rindex = np.abs(ht2).argmax()
print "pycyc rindex:",CS.rindex, "Cyclic-Modelling rindex:", params['rindex2']
ntests += 1
if CS.rindex != params['rindex2']:
    print "rindex2 does not agree"
    nfail +=1


x = pycyc.get_params(ht2, CS.rindex)
pymerit,pygrad = pycyc.cyclic_merit_lag(x,CS)
print "pycyc merit:", pymerit, "Cyclic-Modelling merit:", params['merit2']
ntests += 1
if np.abs(pymerit - params['merit2']) > 1e-5:
    print "merit2 disagree"
    nfail +=1
grad2err = np.abs(grad2-pygrad).mean()
print "grad2 error:", grad2err

ntests += 1
if grad2err > 1e-8:
    print "grad2 disagree"
    nfail += 1


print "\n\nTotal tests: ", ntests, "failed: ", nfail
if nfail == 0:
    print "All tests PASSED"
else:
    print "Some test FAILED"          

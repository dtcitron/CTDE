import cPickle as pickle
import numpy as np
import h5py
from collections import Counter
import networkx as nx



# calculate the mean at each observation time
def m(data, otimes):
    means = np.array([mean(data[i,1][np.where(data[i,1,:] > 0)]) for i in range(len(otimes))])
    # set all means to 0 at all times when there are no active trajectories
    means[isnan(means)] = 0
    return means

# calculate the 
def s(data, otimes):
    stds =  np.array([np.std(data[i, 1][np.where(data[i, 1] > 0)]) for i in range(len(otimes))])
    # set all stds to 1 at all times when there are no active trajectories
    stds[isnan(stds)] = 1
    # if the standard deviation is 0, it means only 1 active trajectory
    stds[np.where(stds == 0)] = 1
    return stds

# return a list of extinction times 
def extimes(data, otimes):
    nruns = np.shape(data)[-1]
    # all times when each trajectory goes extinct
    q = np.array([numpy.where(data[:,1,i] < 0)[0] for i in range(nruns)])
    # if the trajectory never goes extinct, set its extinction time to the last otime index
    t = len(otimes)
    extinctions = np.array([q[i][0] if len(q[i]) > 0 else t for i in range(len(q))])
    return extinctions
#    return otimes[np.array([numpy.where(data[:,1,i] != -1)[0][-1] for i in range(len(otimes))])]

# count the number of extinct trajectories at each observation
def nextinct(data, otimes):
    extinctions = extimes(data, otimes)
    return np.array([len(np.where(extinctions <= tindex)[0]) for tindex in range(len(otimes))])

def data_params(data):
    """
    Given the output from one of the scripts for generating a phase 
    diagram, return two lists of all parameters alphas and R0s.  The 
    input is a single one of the outputs (such as idata or mdata) with
    the form {(alpha, R0):data}
    Inputs:
        idata : dictionary from output of sirs_diagram()
    Outputs:
        alphas: array vector of alpha=rho/gamma
        R0S   : array vector of R0<k>/gamma
    """
    alphas = np.array(sorted(list(set(np.array(data.keys()).transpose()[0]))))
    R0s = np.array(sorted(list(set(np.array(data.keys()).transpose()[1]))))
    return alphas, R0s
    
    
# Now we read ALL of the output files with the given filename as a prefix
# and output everything to a single pickled file for easy retrieval in Python
def cat_config_output_data(filename, alphas, r0s, otimes):
    pickle_data = {}
    for i in range(len(alphas)):
        for j in range(len(r0s)):
            fname = filename +"_rho_" + str(i) + "_r0_" + str(j) + ".hdf5"
            # open and read in data from single output file
            f = h5py.File(fname, "r")
            data = f['sirs'].value
            f.close()
            alpha = alphas[i]
            r0 = r0s[j]
            # count the number of extinct trajectories at each time
            number_extinct = nextinct(data, otimes)
            # Calculate the means at each observation time
            means = m(data, otimes)
            # Calculate the standard deviations at each observation time
            stds = s(data, otimes)
            # write out everything to a big dictionary file
            #print alpha, r0
            pickle_data[(alpha, r0)] = [data, number_extinct, means, stds]
    # create a pickle file
    pfname = filename + "_pickle.dat"
    f = open(pfname, "w")
    pickle.dump([pickle_data, otimes], f)
    f.close()
    

def __init__(config_filename, graph_name):
    """
    Example call from command line
    'python cat_config_output_data.py workflow_tests/config_ba_1000node g1'
    """
    f = h5py.File(config_filename , "r")
    graph_data = f[graph_name]
    filename = graph_data['filename'].value
    edgedata = graph_data['edgedata'].value
    alphas = graph_data['alphas'].value
    r0s = graph_data['R0s'].value
    otimes = graph_data['otimes'].value
    graphsize = graph_data['graphsize'].value
    kmean = graph_data['kmean'].value
    f.close()
    cat_config_output_data(filename, alphas, r0s, otimes)
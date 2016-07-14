import cPickle as pickle
import numpy as np
import h5py
from collections import Counter
#import networkx as nx
import sys

# This is similar to cat_config_output_data.py, except with degree classes


# calculate the mean at each observation time for each degree class
def m(data, otimes):
    # Returns an array with 2 indices: 
    #     Index 0 - observation time
    #     Index 1 - degree class
    # Each array entry is the mean of the degree class at that time
    means = np.array([np.array([np.mean( \
            data[j,i,1][np.where(np.sum(data,1)[j,1,:] > 0)]) \
                for i in range(data.shape[1])])\
                for j in range(data.shape[0])])
    # set all means to 0 at all times when there are no active trajectories
    means[np.isnan(means)] = 0
    return means

# calculate the standard deviation at each observation time for each degree class
def s(data, otimes):
    # Returns an array with 2 indices: 
    #     Index 0 - observation time
    #     Index 1 - degree class
    # Each array entry is the std of the degree class at that time
    stds = np.array([np.array([np.std( \
        data[j,i,1][np.where(np.sum(data,1)[j,1,:] > 0)]) \
            for i in range(data.shape[1])])\
            for j in range(data.shape[0])])
    # set all stds to 1 at all times when there are no active trajectories
    stds[np.isnan(stds)] = 1
    # if the standard deviation is 0, it means only 1 active trajectory
    stds[np.where(stds == 0)] = 1
    return stds

# calculate the TOTAL standard deviation at each observation time 
# summing over all degree classes
def s_total(data, otimes):
    # Returns an array with 2 indices: 
    #     Index 0 - observation time
    #     Index 1 - degree class
    # Each array entry is the std of the degree class at that time
    stds = np.array([np.std(np.sum(data,1)[i,1,:]\
            [np.where(np.sum(data,1)[i,1,:] > 0)]) \
            for i in range(len(otimes))])
    # set all stds to 1 at all times when there are no active trajectories
    stds[np.isnan(stds)] = 1
    # if the standard deviation is 0, it means only 1 active trajectory
    stds[np.where(stds == 0)] = 1
    return stds

# return a list of extinction times 
def extimes(data, otimes):
    nruns = np.shape(data)[-1]
    t = len(otimes)
    # all times when sum total of infecteds goes to 0 for each trajectory
    q = np.array([np.where(np.sum(data,1)[:,1,i] < 0)[0] \
                for i in range(nruns)])    
    # as long as I = 0 appears in the time series, return the time index 
    # when the trajectory first hits zero, or return the max time index
    # if the trajectory never goes extinct, 
    # set its extinction time to the last otime index
    extinctions = np.array([q[i][0] if len(q[i]) > 0 else t \
                    for i in range(len(q))])
    return extinctions

# smooth the time series
def smoothing(data, otimes):
    # CTDE includes I = -1 in cases when the trajectory has not changed since 
    #       the last time step.  We want to replace the -1's with all values
    #       that have come before
    ex = extimes(data, otimes)
    # find all trajectories that have -1 in the first time step
    # kludge: there could be trajectories with multiple -1's at the start
    #   but we're assuming this to be unlikely
    data[0,1,:][np.where(data[0,1,:] == -1)] = \
        data[1,1,:][np.where(data[0,1,:] == -1)]
    # interpolate out all I = -1
    for i in range(np.shape(data)[-1]):
        clone = data[:,1,i];
        blanks = np.where(clone == -1)[0]
        blanks = blanks[np.where(blanks < ex[i])]
        for j in blanks:
            clone[j] = clone[j - 1]
    return data
    
# count the number of extinct trajectories at each observation
def nextinct(data, otimes):
    ex = extimes(data, otimes)
    return np.array([len(np.where(ex <= tindex)[0]) \
        for tindex in range(len(otimes))])

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
            # smooth the data
            #data = smoothing(data, otimes)
            # Calculate the means at each observation time
            means = m(data, otimes)
            # Calculate the standard deviations at each observation time
            stds_total = s_total(data, otimes)
            # Calculate the standard deviations for each degree class
            stds = s(data, otimes)
            # write out everything to a big dictionary file
            #print alpha, r0
            pickle_data[(alpha, r0)] = [data, number_extinct, means, \
                                        stds, stds_total]
    # create a pickle file
    pfname = filename + "_pickle.dat"
    f = open(pfname, "w")
    pickle.dump([pickle_data, otimes], f)
    f.close()
    

def cat_outputs(config_filename, graph_name):
    """
    Example call from command line
    'python cat_config_output_data.py workflow_tests/config_ba_1000node g1'
    """
    f = h5py.File(config_filename , "r")
    graph_data = f[graph_name]
    filename = graph_data['filename'].value
    #edgedata = graph_data['edgedata'].value
    alphas = graph_data['alphas'].value
    r0s = graph_data['r0s'].value
    otimes = graph_data['otimes'].value
    #graphsize = graph_data['graphsize'].value
    #kmean = graph_data['kmean'].value
    f.close()
    cat_config_output_data(filename, alphas, r0s, otimes)
    
if __name__ == "__main__":
    config_filename = sys.argv[1]
    graph_name = sys.argv[2]
    cat_outputs(config_filename, graph_name)


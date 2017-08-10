'''
Created on 2017/05/29

'''
from __future__ import print_function
from __future__ import division
import numpy as np
from py_lsif import LSIF
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.kernel_approximation import RBFSampler, Nystroem
from sklearn.kernel_ridge import KernelRidge
import sys

if __name__ == "__main__":
    # setting of sample
    n_sample_molecule = 100000 # numer of sample molecule
    n_sample_denominator = 100000 # numer of sample denominator
    scale_molecule = 5. # scale of X
    scale_denominator = 5. # scale of Y
    
    np.random.seed(123)
    
    # generate two sample
    sample_molecule = np.random.randn(n_sample_molecule) * scale_molecule + 1.5
    sample_denominator = np.random.randn(n_sample_denominator) * scale_denominator + 1
    
    # train test splits
    n_test = int(sample_molecule.shape[0] * 0.3)
    sample_molecule_train = sample_molecule[:-n_test]
    sample_molecule_test = sample_molecule[n_test:]
    sample_denominator_train = sample_denominator[:-n_test]
    sample_denominator_test = sample_denominator[n_test:]
    
    print("start hyper parameter tuning")
    min_error = float('inf')
    for band_width in np.linspace(2.5e-0, 10e-0, 0):
        for regulation in  np.linspace(1, 5, 0):
            lsif = LSIF(band_width, regulation)
            lsif.fit(sample_molecule_train, sample_denominator_train)
            score = lsif.score(sample_molecule_test, sample_denominator_test)
            print("score %f band_width:%f regulation:%f"%(score, band_width, regulation))
            
            if min_error > score:
                min_error = score
                min_params = (band_width, regulation)
        print("")
        
    # learning
    #print(min_params, min_error)
    max_score = -1
    for gamma in np.linspace(1e-1, 100, 1):
        for regulation in  np.linspace(1e-2, 4, 20):
            sample_X = np.append(sample_molecule_train, sample_denominator_train, 0).reshape(-1, 1)
            sample_y = np.array([0] * sample_molecule_train.shape[0] + [1] * sample_denominator_train.shape[0])
            #rbf_feature = Nystroem(gamma=gamma, n_components=500)
            #X_features = rbf_feature.fit_transform(sample_X)
            X_features = sample_X
            lsif = LogisticRegression(C=regulation)
            lsif.fit(X_features, sample_y)
            
            test_X = np.append(sample_molecule_test, sample_denominator_test, 0).reshape(-1, 1)
            test_y = np.array([0] * sample_molecule_test.shape[0] + [1] * sample_denominator_test.shape[0])
            #rbs_space = rbf_feature.transform(test_X)
            rbs_space = test_X
            cur_score = lsif.score(rbs_space, test_y)
            print(gamma, regulation, cur_score)
            if max_score < cur_score:
                max_score = cur_score
                max_params = (gamma, regulation)
    
    print(max_params)
    gamma, regulation = max_params
    
    lsif = LogisticRegression(C=regulation)
    sample_X = np.append(sample_molecule, sample_denominator, 0).reshape(-1, 1)
    sample_y = np.array([0] * sample_molecule.shape[0] + [1] * sample_denominator.shape[0])
    #rbf_feature = RBFSampler(gamma=gamma, random_state=1)
    #X_features = rbf_feature.fit_transform(sample_X)
    X_features = sample_X
    lsif.fit(X_features, sample_y)
    #lsif = LSIF(min_params[0], min_params[1])
    #lsif.fit(sample_molecule, sample_denominator)
    
    # compute true ratio values
    prob_true = norm.pdf(sample_molecule, loc=1.5, scale=scale_molecule) / norm.pdf(sample_molecule, loc=1., scale=scale_denominator)
    # estimate ratio
    prob_est = lsif.predict_proba(sample_molecule.reshape(-1, 1))
    print(prob_est)
    
    plt.title('confirm accuracy')
    plt.xlabel("estimate values")
    plt.ylabel(" true values")
    plt.grid(True)
    plt.scatter(prob_est[:, 0], prob_true, alpha=0.1)
    plt.xlim(0, 3)
    plt.ylim(0, 3)
    plt.show()

'''
Created on 2017/05/29

'''
from __future__ import print_function
from __future__ import division
import numpy as np
from lsif import LSIF
from scipy.stats import norm
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # setting of sample
    n_sample_molecule = 500 # numer of sample molecule
    n_sample_denominator = 500 # numer of sample denominator
    scale_molecule = 3. # scale of X
    scale_denominator = 3. # scale of Y
    
    np.random.seed(123)
    
    # generate two sample
    sample_molecule = np.random.randn(n_sample_molecule, 1) * scale_molecule + 1.5
    sample_denominator = np.random.randn(n_sample_denominator, 1) * scale_denominator + 1
    
    # train test splits
    n_test = int(sample_molecule.shape[0] * 0.3)
    sample_molecule_train = sample_molecule[:-n_test]
    sample_molecule_test = sample_molecule[n_test:]
    sample_denominator_train = sample_denominator[:-n_test]
    sample_denominator_test = sample_denominator[n_test:]
    
    print("start hyper parameter tuning")
    min_error = float('inf')
    for band_width in np.linspace(2.5, 5, 3):
        for regulation in  np.linspace(5e-2, 2, 3):
            lsif = LSIF(band_width, regulation)
            lsif.fit(sample_molecule_train, sample_denominator_train)
            score = lsif.score(sample_molecule_test, sample_denominator_test)
            print("score %f band_width:%f regulation:%f"%(score, band_width, regulation))
            
            if min_error > score:
                min_error = score
                min_params = (band_width, regulation)
            
    # learning
    lsif = LSIF(min_params[0], min_params[1])
    lsif.fit(sample_molecule, sample_denominator)
    
    # compute true ratio values
    prob_true = norm.pdf(sample_molecule, loc=1.5, scale=scale_molecule) / norm.pdf(sample_molecule, loc=1., scale=scale_denominator)
    # estimate ratio
    prob_est = lsif.predict(sample_molecule)
    
    plt.xlabel(" estimate values")
    plt.ylabel(" true values")
    plt.grid(True)
    plt.scatter(prob_est, prob_true, alpha=0.1)
    plt.xlim(0, 2)
    plt.ylim(0, 2)
    plt.show()
import numpy as np
from main import BlockAnalysis

a = np.loadtxt('test.dat') #Load time series
blocks = BlockAnalysis(a) #Initialize class and do block transformation
stat = blocks.stat #Get error vs block size profile
bs, err = blocks.err() #Get decorrelating block size and error on the observable

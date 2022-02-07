import numpy as np
from scipy.optimize import minimize
from block_tools import *

class BlockAnalysis:

    def __init__(self, x, multi=1):
        self.multi = multi
        self.x = check(x, self.multi)
        self.stat = blocking(self.x, self.multi)

    def err(self):

        def find_n_intersect(x,stat):
                c=0
                for i,p in enumerate(stat):
                    if (x <= p[1]+p[2]) and (x >= p[1]-p[2]):
                        c += (np.pi*p[2]) * np.exp(-0.5*((x-p[1])/p[2])**2)
                return -c

        c = np.zeros(len(self.stat))
        for i,b in enumerate(self.stat):
            lower_bound = b[1]-b[2]
            upper_bound = b[1]+b[2]
            bnds = [(lower_bound, upper_bound)]
            c[i] -= minimize( fun=find_n_intersect, x0=b[1], args=self.stat[self.stat[...,0] > b[0]], bounds=bnds ).fun    
        return self.stat[...,0][np.argmax(c)], self.stat[...,1][np.argmax(c)]


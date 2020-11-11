import numpy as np
from kneed import KneeLocator

def blocker(array, multi=1):
    
    dimension = len(array)/multi
    n_blocks_try = np.arange(multi,dimension+1)
    if multi == 1:
        n_blocks = []
        block_sizes = []
    else:
        n_blocks = [multi]
        block_sizes = [dimension]
    
    for n in n_blocks_try:
        if dimension % n == 0:
            n_blocks.append(int(n*multi))
            block_sizes.append(dimension/n)
    
    return dimension*multi, np.array(n_blocks), block_sizes


def check(cv, bias, multi):
    nt = len( blocker(cv, multi=multi)[1] )
    if nt > 19:
        print ("Possible blocks transformation: "+str(nt)+"\n no lenght correction needed\n")
        return cv, bias
    else:
        replen = int(len(cv) / multi)
        for c in range(1,102):
            print ("Removing "+str(c)+" at the bottom of each replica")
            chunks_cv = np.array([])
            chunks_b = np.array([])
            for n in range(1,multi+1):
                e = replen*n
                s = e - replen
                chunks_cv = np.concatenate((cv[s:e-c],chunks_cv))
                chunks_b = np.concatenate((bias[s:e-c],chunks_b))
            nt = len( blocker(chunks_cv, multi=multi)[1] )
            print ("Possible blocks transformation: "+str(nt)+"\n")
            if nt > 19:
                break
        return chunks_cv, chunks_b


def fblocking(cv, bias, temp, multi=1):
    norm_bias = bias - np.max(bias)
    kb = 0.008314463
    kbt = kb*temp
    w = np.exp(norm_bias/kbt)
    w = w / w.sum()
    W = w.sum()
    S = (w**2).sum()
    
    bins = np.histogram(cv,bins=50,weights=w)[1]
    N, n_blocks, block_sizes = blocker(cv, multi=2)
    
    errs = []
    errs_errs = []
    for b in range(len(block_sizes)): 
        blocks_h = []
        wis = []
        Nb = n_blocks[b]
        div = Nb * (W-(S/W))
        for n in range(1,Nb+1):
            end = int( block_sizes[b] * n )
            start = int( end - block_sizes[b] )
            hi = np.histogram(cv[start:end],bins=bins,weights=w[start:end])[0]
            wi = w[start:end].sum()
            blocks_h.append( hi )
            wis.append(wi)
        blocks_h = np.array(blocks_h)
        wis = np.array(wis)
        u = np.zeros(len(bins)-1) ####
        for i in range(len(bins)-1):
            u[i] = ( (wi*blocks_h[...,i]).sum() ) / W
        
        e = np.zeros(len(bins)-1)
        for i in range(len(bins)-1):
            e[i] += (wi*(blocks_h[...,i]-u[i])**2).sum()
        e = np.sqrt(e/div)
        e = kbt * (e / u)
        errs.append(e)
        
    err_av = np.nanmean(errs, axis = 1)
    err_err = err_av/np.sqrt(2*((np.array(n_blocks)-1)))
    
    return np.flip( np.array([block_sizes, err_av, err_err]).T , axis=0  )


def optimal_block(ndata, stat, method, S=2.7):
    
    if method == "b3":
        
        err_first = stat[0,1]
        opt = (np.nan,np.nan)

        for (block_size, err, err_err) in reversed(stat):
            B3 =  block_size**3
            if B3 > ndata*(err/err_first)**4 :
                opt = (block_size, err)

        if (opt[0] > (ndata/50)):
            print( "You may not be converging. Sample more." )

        return opt[0], opt[1]
    
    
    if method == "knee_loc":
        kneedle = KneeLocator(stat[...,0], stat[...,1], S=S, curve="concave", direction="increasing")
        bs = kneedle.knee
        err = kneedle.knee_y
    
        return bs, err

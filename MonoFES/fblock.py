import numpy as np

def blocker(array, multi=1):
    
    dimension = len(array)/multi
    n_blocks_try = np.arange(2,dimension+1)
    if multi == 1:
        n_blocks = []
        block_sizes = []
    else:
        n_blocks = [multi]
        block_sizes = [dimension*2/multi]
    
    for n in n_blocks_try:
        if dimension % n == 0:
            n_blocks.append(int(n*2))
            block_sizes.append(dimension/n)
    
    return dimension*multi, np.array(n_blocks), block_sizes


def check(cv, bias, multi):
    nt = len( blocker(cv, multi=multi)[1] )
    if nt > 10:
        print ("Possible blocks transformation: "+str(nt)+"\n no lenght correction needed\n")
    else:
        replen = int(len(cv) / multi)
        for c in range(1,11):
            print ("Removing "+str(c)+" at the bottom of each replica")
            chunks_cv = np.array([])
            chunks_b = np.array([])
            for n in range(1,multi+1):
                e = replen*n
                s = e - replen
                np.concatenate((cv[s:e-c],chunks_cv))
                np.concatenate((bias[s:e-c],chunks_b))
            nt = len( blocker(chuncks, multi=multi)[1] )
            print ("Possible blocks transformation: "+str(nt)+"\n")
            if nt > 14:
                break
    return chunks_cv, chunks_bias


def fblocking(cv, bias, temp, multi=1):
    norm_bias = bias - np.max(bias)
    kb = 0.008314463
    kbt = kb*temp
    w = np.exp(norm_bias/kbt)
    w = w / w.sum()
    W = w.sum()
    S = (w**2).sum()
    
    u, bins = np.histogram(cv,bins=100,weights=w, density=True)
    N, n_blocks, block_sizes = blocker(cv, multi=2)
    
    errs = []
    errs_errs = []
    for b in range(len(block_sizes)): 
        blocks_h = []
        wis = []
        for n in range(1,n_blocks[b]+1):
            Nb = n_blocks[b]
            end = int( block_sizes[b] * n )
            start = int( end - block_sizes[b] )
            hi = np.histogram(cv[start:end],bins=bins,weights=w[start:end], density=True)[0]
            wi = w[start:end].sum()
            blocks_h.append( wi*(hi-u)**2 )
        blocks_h = np.array(blocks_h)

        e = np.sqrt( blocks_h.sum(axis=0) / (Nb*(W-S/W)) )
        errs.append(e)
        
    err_av = np.average(errs, axis = 1)
    err_err = err_av/np.sqrt(2*((np.array(n_blocks)-1)))
    
    return np.flip( np.array([block_sizes, err_av, err_err]).T , axis=0  )



def optimal_block(ndata, stat):
    err_first = stat[0,1]
    opt = (np.nan,np.nan)
    
    for (block_size, err, err_err) in reversed(stat):
        B3 =  block_size**3
        if B3 > ndata*(err/err_first)**4 :
            opt = (block_size, err)
            
    #if (opt[0] > (ndata/50)):
    #    print( "You may not be converging. Sample more." )
    
    return opt


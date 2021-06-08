import numpy as np

def blocker(array, multi=1):
    dimension = len(array)
    rep = dimension/multi
    n_blocks_try = np.arange(multi,dimension+1)
    n_blocks = []
    block_sizes = []

    for n in n_blocks_try:
        bs = dimension/n
        if (dimension % n == 0) & (rep % bs == 0):
            n_blocks.append(int(n))
            block_sizes.append(bs)

    return dimension, np.array(n_blocks), block_sizes


def check(cv, bias, multi):
    nt = len( blocker(cv, multi=multi)[1] )
    if nt > 19:
        print ("Possible blocks transformations: "+str(nt)+"\n no lenght correction needed\n")
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
            print ("Possible blocks transformations: "+str(nt)+"\n")
            if nt > 19:
                break
        return chunks_cv, chunks_b


def fblocking(cv, bias, temp, multi=1):
    norm_bias = bias - np.max(bias)
    kb = 0.008314463
    kbt = kb*temp
    w = np.exp(norm_bias/kbt)
    w = w / w.sum()

    N, n_blocks, block_sizes = blocker(cv, multi=multi)
    u, bins = np.histogram(cv,weights=w,bins=50)
    u = u/N
    
    err = np.zeros(len(block_sizes))
    err_err = np.zeros(len(block_sizes))
    for b in range(len(block_sizes)):
        Nb = n_blocks[b]
        his = np.zeros(len(bins)-1)
        for n in range(Nb):
            start = int( n*block_sizes[b] )
            end = int( start + block_sizes[b] )
            hi = np.histogram(cv[start:end], weights=w[start:end], bins=bins)[0] / len(cv[start:end])
            his += (hi-u)**2
        e = np.sqrt( his / (Nb*(Nb-1)) )
        e = kbt*e/u
        err[b] += e.mean()
        err_err[b] += err[b] / np.sqrt( 2*(Nb-1) )
    
    return np.flip( np.array([block_sizes, err, err_err]).T , axis=0  )


def optimal_block(stat, method="hline"):

    if method == "hline":
        c = np.zeros(len(stat))
        for i,b in enumerate(stat[...,1]):
            for p in stat:
                if (b <= p[1]+p[2]) and (b >= p[1]-p[2]):
                    c[i] += 1
        return stat[...,0][np.argmax(c)], stat[...,1][np.argmax(c)]

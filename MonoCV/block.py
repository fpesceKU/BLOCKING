import numpy as np


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


def check(array, multi=1):
    nt = len( blocker(array, multi=multi)[1] )
    if nt > 19:
        print ("Possible blocks transformations: "+str(nt)+"\n no lenght correction needed\n")
        return array
    else:
        replen = int(len(array) / multi)
        for c in range(1,102):
            print ("Removing "+str(c)+" at the bottom of each replica")
            chunks_array = np.array([])
            for n in range(1,multi+1):
                e = replen*n
                s = e - replen
                chunks_array = np.concatenate((array[s:e-c],chunks_array))
            nt = len( blocker(chunks_array, multi=multi)[1] )
            print ("Possible blocks transformations: "+str(nt)+"\n")
            if nt > 19:
                break
        return chunks_array

 
def blocking(array, multi=1):
    
    u = array.mean()
    N, n_blocks, block_sizes = blocker(array, multi=multi)
    
    errs = []
    errs_errs = []
    for b in range(len(block_sizes)):
        Nb = n_blocks[b]
        blocks_av = np.zeros(Nb)
        for n in range(1,Nb+1):
            end = int( block_sizes[b] * n )
            start = int( end - block_sizes[b] )
            blocks_av[n-1] = array[start:end].mean()

        err = np.sqrt( ((blocks_av - u)**2).sum() / (Nb*(Nb-1)) )
        errs.append(err)

        err_err = err/(np.sqrt(2*(Nb-1)))

        errs_errs.append(err_err)

    return np.flip( np.array([block_sizes, errs, errs_errs]).T , axis=0  )


def optimal_block(stat, method="hline"):

    if method == "hline":
        c = np.zeros(len(stat))
        for i,b in enumerate(stat[...,1]):
            for p in stat:
                if (b <= p[1]+p[2]) and (b >= p[1]-p[2]):
                    c[i] += 1
        return stat[...,0][np.argmax(c)], stat[...,1][np.argmax(c)]

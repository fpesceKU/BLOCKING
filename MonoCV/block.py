import numpy as np

def blocking(array, power2=False):
    
    if power2==True:
        log2_len_a=np.log(len(array))/np.log(2)
        log2_len_sub_a=int(np.floor(log2_len_a))
        len_sub_a=2**log2_len_sub_a
        array = array[:len_sub_a]
    
    n_blocks_try = np.arange(2,len(array)+1)
    n_blocks = []
    block_sizes = []
    dimension = len(array)

    for n in n_blocks_try:
        if dimension % n == 0:
            n_blocks.append(n)
            block_sizes.append(dimension/n)

    errs = []
    errs_errs = []
    for b in range(len(block_sizes)): 
        blocks_av = []
        for n in range(1,n_blocks[b]+1):
            end = int( block_sizes[b] * n )
            start = int( end - block_sizes[b] )
            blocks_av.append(array[start:end].mean())

        blocks_av = np.array(blocks_av)
        u = blocks_av.mean()
        N = len(blocks_av)

        err = np.sqrt( ((blocks_av - u)**2).sum() / (N*(N-1)) )
        errs.append(err)
        
        err_err = err/(np.sqrt(2*(N-1)))
        
        errs_errs.append(err_err)
    
    return np.flip( np.array([block_sizes, errs, errs_errs]).T , axis=0  )


def optimal_block(ndata, stat):
    err_first = stat[0,1]
    opt = None
    
    for (block_size, err, err_err) in reversed(stat):
        B3 =  block_size**3
        if B3 > 2*ndata*(err/err_first)**4 :
            opt = (block_size, err)
            
    if (opt == None):
        print( "You may not be converging. Sample more." )
    else:
        if opt[0] < (ndata/50):
            return opt
        else:
            print( "You may not be converging. Sample more." )
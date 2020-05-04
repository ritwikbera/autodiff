import numpy as np 
from numpy import ceil, log2
from collections import Counter

# log N space requirement vs sqrt(N)

def recursive(f, s_init, begin, end):
    if end-begin <= 1:
        yield s_init
    else:
        s = s_init
        mid = (end - begin)//2
        for _ in range(mid):
            s = f(s)   # number of steps till termination condition N(1/2+1/4....)
        for s in recursive(f, s, begin + mid, end):
            yield s
        for s in recursive(f, s_init, begin, begin + mid):
            yield s

def test(N):
    
    calls = Counter()

    def layer(s):
        calls[s] += 1
        return s+1

    # print(list(recursive(layer, layer(0), 0, N))) # doesn't include input as activation
    assert list(recursive(layer, layer(0), 0, N)) == list(reversed(range(1, N+1)))
    np.testing.assert_array_equal([num_calls <= ceil(log2(N)) for layer, num_calls in calls.items()],[True for _ in range(N)]) 


test(10) # 10 activations
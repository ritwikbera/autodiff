import numpy as np 
from numpy import ceil, sqrt, log2
from collections import Counter, defaultdict

def sqrt_space(f, s0, N):
    iters_bef_checkpoint = int(ceil(sqrt(N)))
    memory = defaultdict(float)

    s = s0 
    t = 0
    while t < N:
        s = f(s) # don't include input as an activation to be stored
        if t % iters_bef_checkpoint == 0:
            memory[t] = s
        t += 1
    
    pos = N
    k = iters_bef_checkpoint
    first_checkpoint_index = iters_bef_checkpoint

    while pos >= first_checkpoint_index: # at equality only first chunk remains to be computed
        num_steps = ((N % k) or k) if pos == N else k
        for s in reversed(step(f, memory[pos-num_steps], num_steps)):
            yield s
            
        pos -= num_steps

def step(f, checkpoint, num_steps):
    s = checkpoint 

    if num_steps == 0:
        return []

    aux_memory = [s]
    for _ in range(num_steps-1):
        s = f(s)
        aux_memory.append(s)
    return aux_memory

def test(N):
    
    calls = Counter()
    def layer(s):
        calls[s] += 1
        return s+1

    assert list(sqrt_space(layer, 0, N)) == list(reversed(range(1, N+1)))
    # print(list(sqrt_space(layer, 0, N)))
    np.testing.assert_array_equal([1 <= num_calls <=2 for layer, num_calls in calls.items()],[True for _ in range(N)]) 

test(5)




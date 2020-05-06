import numpy as np 

def f(x):
	return x

def grad_f(x):
	return 1

def product(items):
	return np.prod(items)

def analytic_gradient(fun, x, delta=1e-4):
    x = np.asfarray(x)
    grad = np.zeros(x.shape, dtype=x.dtype)

    for i, t in np.ndenumerate(x):
        x[i] = t + delta
        grad[i] = fun(x)
        x[i] = t - delta
        grad[i] -= fun(x)
        x[i] = t

    return grad / (2 * delta)

# O(N**2) time and O(1) space complexity
def naive(items):
	out = []
	for i in range(len(items)):
		aux = 1
		for j in range(len(items)):
			if i!=j:
				aux *= f(items[j]) # compute activations every time
		out.append(aux*grad_f(items[i]))

	return out



# O(N) space and time complexity
def prod_grad(items):
	items = np.vectorize(f)(items) # store all activations
	out = []
	for i in range(len(items)):
		if i == 0:
			left_prod = 1
			right_prod = np.prod(items[1:])
		else:	
			left_prod *= items[i-1]
			right_prod /= items[i]
		# print(left_prod, right_prod)
		out.append(left_prod*grad_f(items[i])*right_prod)

	return out 

if __name__ == '__main__':
	inputs = [i for i in range(1, 5)]
	assert prod_grad(inputs)==naive(inputs)
	assert (analytic_gradient(product, inputs) - prod_grad(inputs) < 1e-2).all()
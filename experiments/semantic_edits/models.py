import numpy as np
from .data import D

class Ridge:
    def __init__(self, alpha=1.):
        self.alpha = alpha
    def fit(self, x, y):
        self.mean, self.target_mean = x.mean(0), y.mean(0)
        z = x - self.mean
        self.weight = np.linalg.solve(z.T@z + self.alpha*np.eye(z.shape[1]), z.T@(y-self.target_mean))
        return self
    def __call__(self, x):
        return (x-self.mean)@self.weight+self.target_mean

class Representation:
    def __init__(self, residual_dim=0, seed=0):
        self.residual_dim, self.seed = residual_dim, seed
    def fit(self, x, c):
        self.concepts = Ridge().fit(x, c)
        self.mean = x.mean(0)
        rng = np.random.default_rng(self.seed)
        self.projection = rng.normal(size=(x.shape[1], self.residual_dim))/np.sqrt(x.shape[1])
        return self
    def __call__(self, x):
        return np.concatenate([self.concepts(x), (x-self.mean)@self.projection], axis=1)

def coordinate(z, actions):
    out = z.copy()
    for i, a in enumerate(actions):
        o, c = a[:2].argmax(), a[2:].argmax()
        out[i, 5*o:5*o+3] = 0
        out[i, 5*o+c] = 1
    return out

class LinearEdit:
    def fit(self, z, a, target, zv, av, tv):
        self.model = Ridge().fit(np.concatenate([z, a], 1), target-z)
        return self
    def __call__(self, z, a):
        return z+self.model(np.concatenate([z, a], 1))

class MLPEdit:
    """One hidden layer; full-batch Adam; validation selects checkpoint only."""
    def __init__(self, seed=0, steps=400, hidden=64):
        self.seed, self.steps, self.hidden = seed, steps, hidden
    def fit(self, z, a, target, zv, av, tv):
        x = np.concatenate([z, a], 1)
        self.mean, self.scale = x.mean(0), np.maximum(x.std(0), .1)
        x = (x-self.mean)/self.scale
        xv = (np.concatenate([zv, av], 1)-self.mean)/self.scale
        y = target-z
        rng = np.random.default_rng(self.seed)
        p = [rng.normal(size=(x.shape[1],self.hidden))/np.sqrt(x.shape[1]),
             np.zeros(self.hidden), rng.normal(size=(self.hidden,z.shape[1]))*.01,
             np.zeros(z.shape[1])]
        m, v = [np.zeros_like(t) for t in p], [np.zeros_like(t) for t in p]
        best = float('inf')
        for step in range(1,self.steps+1):
            h = np.tanh(x@p[0]+p[1])
            err = h@p[2]+p[3]-y
            g = 2*err/err.size
            gh = (g@p[2].T)*(1-h*h)
            grads = [x.T@gh, gh.sum(0), h.T@g, g.sum(0)]
            for j in range(4):
                m[j] = .9*m[j]+.1*grads[j]
                v[j] = .999*v[j]+.001*grads[j]**2
                p[j] -= .003*(m[j]/(1-.9**step))/(np.sqrt(v[j]/(1-.999**step))+1e-8)
            loss = np.mean((zv+np.tanh(xv@p[0]+p[1])@p[2]+p[3]-tv)**2)
            if loss < best:
                best = loss
                self.params = [t.copy() for t in p]
                self.best_step = step
        return self
    def __call__(self,z,a):
        x = (np.concatenate([z,a],1)-self.mean)/self.scale
        w,b,u,d = self.params
        return z+np.tanh(x@w+b)@u+d

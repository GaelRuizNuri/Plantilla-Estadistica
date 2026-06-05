from scipy.stats import chi2, norm, nbinom, beta, uniform, poisson
import numpy as np

def probabilidades_funcion_discreta(f, expresiones, dominio, *args):
    probs = []
    for expresion in expresiones:

        if isinstance(expresion, int):
            if expresion<dominio[0] or expresion>dominio[1]:
                probs.append(0)
            else:
                probs.append(f(expresion, *args))

        else:
            a, b = expresion
            if a is None: 
                n = np.arange(dominio[0], b+1)
                probs.append(np.sum(f(n,*args))) # P(X <= b)
            elif b is None: 
                n = np.arange(a, dominio[1]+1)
                probs.append(np.sum(f(n,*args))) # P(X >= a)
            else: 
                n = np.arange(a, b+1)
                probs.append(np.sum(f(n,*args))) # P(a <= X <= b)

    return probs


def probabilidades_discretas(dist, expresiones):
    probs = []
    for expresion in expresiones:
        
        if isinstance(expresion, int):
            probs.append(dist.pmf(expresion)) # P(X = k)

        else:
            a, b = expresion
            if a is None: probs.append(dist.cdf(b)) # P(X <= b)
            elif b is None: probs.append(dist.sf(a - 1)) # P(X >= a)
            else: probs.append(dist.cdf(b) - dist.cdf(a - 1)) # P(a <= X <= b)

    return probs


def probabilidades_continuas(dist, expresiones):
    probs = []

    for expresion in expresiones:
        a, b = expresion
        if a is None: probs.append(dist.cdf(b)) # P(X < b)
        elif b is None: probs.append(dist.sf(a)) # P(X > a) v P(a < X)
        else: probs.append(dist.cdf(b) - dist.cdf(a)) # P(a < X < b)

    return probs
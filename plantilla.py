from scipy.stats import chi2, norm, nbinom, beta, uniform, poisson
import numpy as np


def probabilidades_discretas(dist, expresiones):
    probs = {}
    for expresion in expresiones:
        
        if isinstance(expresion, tuple) and len(expresion) == 2:
            indice, k = expresion
            probs[indice] = dist.pmf(k) # P(X = k)

        else:
            indice, a, b = expresion

            if a is None: probs[indice] = dist.cdf(b) # P(X <= b)

            elif b is None: probs[indice] = dist.sf(a - 1) # P(X >= a)

            else: probs[indice] = dist.cdf(b) - dist.cdf(a - 1) # P(a <= X <= b)

    return probs


def probabilidades_continuas(dist, expresiones):
    probs = {}

    for expresion in expresiones:
        indice, a, b = expresion
        if a is None: probs[indice] = dist.cdf(b) # P(X < b)

        elif b is None: probs[indice] = dist.sf(a) # P(X > a) v P(a < X)

        else: probs[indice] = dist.cdf(b) - dist.cdf(a) # P(a < X < b)

    return probs


def agrupar_esp(observados, esperados, minimo=5):
    obs, esp = [], []
    fo, fe = 0, 0

    for i in range(len(observados)):
        fo += observados[i]
        fe += esperados[i]

        if fe >= minimo:
            obs.append(fo)
            esp.append(fe)
            fo, fe = 0, 0

    if fe > 0:
        if len(obs) > 0:
            obs[-1] += fo
            esp[-1] += fe
        else:
            obs.append(fo)
            esp.append(fe)

    return obs, esp


def chi_obs_bondad_ajuste(observados, probs):
    n = sum(observados)
    esperados = [n*probs[i] for i in range(len(observados))]
    observados, esperados = agrupar_esp(observados, esperados)

    chi_observado = sum((fo-fe)*(fo-fe)/fe for fo, fe in zip(observados, esperados))
    print(f'Observados     | {observados}')
    print(f'Probabilidades | {list(probs.values())}')
    print(f'Esperados      | {esperados}')
    print(f'Chi Cuadrado observado: {chi_observado}')
    return chi_observado

#Ej práctica #3 de la PPT 
expresiones = [
    (0,None,1.45),  # x < 1.45
    (1,1.45,1.55),  # 1.45 < x < 1.55
    (2,1.55,1.65),  # 1.55 < x < 1.65
    (3,1.65,1.75),  # 1.65 < x < 1.75
    (4,1.75, 1.85), # 1.75 < x < 1.85
    (5,1.85, 1.95), # 1.85 < x < 1.95
    (6,1.95, None)  # 1.95 < x
]

dist = norm(loc=1.76, scale=0.1)
chi_obs_bondad_ajuste([2,4,41,110,121,48,9], probabilidades_continuas(dist, expresiones))
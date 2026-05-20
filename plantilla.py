from scipy.stats import chi2, norm, nbinom, beta, uniform
import numpy as np



def agrupar_esp(observados, esperados):
    sortedList = sorted(zip(esperados, observados))
    observados, esperados = [o for (_,o) in sortedList], [e for (e,_) in sortedList]
    i = 0
    while esperados[i]<5:
        i+=1
        esperados[i]+=esperados[i-1]
        observados[i]+=observados[i-1]
    return observados[i::], esperados[i::]

def chi_obs(observados, f):
    esperados = [x*f(x) for x in observados]
    observados, esperados = agrupar_esp(observados, esperados)
    return sum((fo-fe)*(fo-fe)/fe for fo, fe in zip(observados, esperados))




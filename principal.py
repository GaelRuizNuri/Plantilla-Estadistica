from scipy.stats import chi2, norm, nbinom, beta, uniform, poisson
import calculos
import graficas
import numpy as np


X = [2,6,8,10,12,14,16]
Y = [1,5,8,11,15,20,25]
Yx, a,b,r = calculos.regresion_lineal_simple(X,Y);


graficas.distribucion_discreta(poisson(3),0,10,show=True, titulo = "Poisson λ = 3")
graficas.distribucion_continua(norm(0,1),-5,5, show=True, titulo = "Normal μ = 0 y  σ = 1")
graficas.bondad_ajuste([[np.random.randint(1, 7) for i in range(1000)].count(i) for i in range(1,7)], 6*[1000/6], show = True)
graficas.regresion_lineal_simple(Yx, X, Y, a, b, r, show = True);


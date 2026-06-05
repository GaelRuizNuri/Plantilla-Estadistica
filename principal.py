from scipy.stats import chi2, norm, nbinom, beta, uniform, poisson
import numpy as np
import calculos
import bondad_ajuste
import independencia
import anova
import regresion
import graficas
LONG = 40
TMENU = "MENU"
TGRAFICAS = "GRÁFICAS"


# Data
X = [2,6,8,10,12,14,16]
Y = [1,5,8,11,15,20,25]

observados = [[np.random.randint(1, 7) for _ in range(1000)].count(i) for i in range(1,7)]
esperados = 6*[1000/6]


def menu_graficar():
    while True:
        print("\n" + "="*LONG)
        print(" "*((LONG - len(TGRAFICAS)) // 2) + TGRAFICAS)
        print("="*LONG)
        print("1. Gráficar Bondad de Ajuste")
        print("2. Gráficar Regresión Lineal")
        print("3. Gráficar Distribuciones")
        print("0. Salir")
        print("-"*LONG)
        
        opcion = input("Opción: ")
        
        if opcion == '1':
            print("\n--- Gráfica Bondad de Ajuste ---")
            #graficas.bondad_ajuste(observados, esperados, show=True)
 
        elif opcion == '2':           
            print("\n--- Gráfica Regresión Lineal ---")
            #Yx, a, b, r = regresion.regresion_lineal_simple(X, Y)
            #graficas.regresion_lineal_simple(Yx, X, Y, a, b, r, show=True)
            
        elif opcion == '3':
            print("\n--- Gráfica Distribuciones ---")
            #graficas.distribucion_discreta(poisson(3),0,10,show=True, titulo = "Poisson λ = 3")
            #graficas.distribucion_continua(norm(0,1),-5,5, show=True, titulo = "Normal μ = 0 y  σ = 1")
            
        elif opcion == '0':
            break


def menu():
    while True:
        print("\n" + "="*LONG)
        print(" "*((LONG - len(TMENU)) // 2) + TMENU)
        print("="*LONG)
        print("1. Bondad de Ajuste")
        print("2. Independencia")
        print("3. ANOVA")
        print("4. Regresión Lineal")
        print("5. Gráficar")
        print("0. Salir")
        print("-"*LONG)
        
        opcion = input("Opción: ")
        
        if opcion == '1':
            print("\n--- Bondad de Ajuste ---")
            #bondad_ajuste.chi_obs_bondad_ajuste(observados, probs)
            
        elif opcion == '2':
            print("\n--- Independencia ---")
            #independencia.chi_obs_independencia(tabla_obs)
            
        elif opcion == '3':
            print("\n--- ANOVA ---")
            #anova.f_obs_anova(grupos)
            
        elif opcion == '4':
            print("\n--- Regresión Lineal ---")
            #regresion.regresion_lineal_simple(X, Y)
            
        elif opcion == '5':
            menu_graficar()
            
        elif opcion == '0':
            break

if __name__ == '__main__':
    menu()
from scipy.stats import chi2, norm, nbinom, beta, uniform, poisson
import numpy as np
import calculos
import bondad_ajuste
import independencia
import anova
import regresion
import intervalos
import graficas
LONG = 40
TMENU = "MENU"
TGRAFICAS = "GRÁFICAS"
TINTERVALOS = "INTERVALOS"



def ingresar_datos():
    value = input("Ingrese la letra 'q' para finalizar la inserción\n")
    data = []
    while (value != "q"):
        data.append(float(value))
        value = input()
    return data



def menu_graficar():
    while True:
        print("\n" + "="*LONG)
        print(" "*((LONG - len(TGRAFICAS)) // 2) + TGRAFICAS)
        print("="*LONG)
        print("1. Gráficar Bondad de Ajuste")
        print("2. Gráficar Regresión Lineal")
        print("3. Gráficar Regresión No Lineal Simple")
        print("4. Gráficar Distribuciones")
        print("5. Gráficar Puntos")
        print("0. Salir")
        print("-"*LONG)
        
        opcion = input("Opción: ")
        
        if opcion == '1':
            print("\n--- Gráfica Bondad de Ajuste ---")
            #graficas.bondad_ajuste(observados, esperados, show=True)
 
        elif opcion == '2':           
            print("\n--- Gráfica Regresión Lineal ---")
            Yx, a, b, _, _, _, r, _, _ = regresion.regresion_lineal_simple(X, Y)
            graficas.regresion_lineal_simple(Yx, X, Y, a, b, r, show=True)

        elif opcion == '3':           
            print("\n--- Gráfica Regresión No Lineal Simple ---")
            tipo = input("Ingrese el tipo (exponencial, inverso, potencial o logaritmico):\n")
            est = regresion.regresion_no_lineal_linealizable(X, Y, tipo)
            graficas.regresion_no_lineal_simple(est["modelo"], X, Y, r2 = est["r2_original"], show=True)
            
        elif opcion == '4':
            print("\n--- Gráfica Distribuciones ---")
            #graficas.distribucion_discreta(poisson(3),0,10,show=True, titulo = "Poisson λ = 3")
            #graficas.distribucion_continua(norm(0,1),-5,5, show=True, titulo = "Normal μ = 0 y  σ = 1")

        elif opcion == '5':
            print("\n--- Gráfica Puntos ---")
            graficas.puntos(X,Y, show = True)
            
        elif opcion == '0':
            break
 
 
def menu_intervalos():
    while True:
        print("\n" + "="*LONG)
        print(" "*((LONG - len(TINTERVALOS)) // 2) + TINTERVALOS)
        print("="*LONG)
        print("1. IC para β")
        print("2. IC para α")
        print("3. IC para E(Y | x0)")
        print("4. IP para y0 individual")
        print("0. Salir")
        print("-"*LONG)
 
        opcion = input("Opción: ")
            
        if opcion == '1':
            print("\n--- IC para β ---")
            intervalos.ic_parametro_beta(X, Y)
 
        elif opcion == '2':
            print("\n--- IC para α ---")
            intervalos.ic_parametro_alpha(X, Y)
 
        elif opcion == '3':
            print("\n--- IC para E(Y | x0) ---")
            x0 = float(input("Ingrese x0: "))
            intervalos.ic_media_condicional(X, Y, x0)
 
        elif opcion == '4':
            print("\n--- IP para y0 individual ---")
            x0 = float(input("Ingrese x0: "))
            intervalos.ip_prediccion_individual(X, Y, x0)
 
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
        print("5. Regresión No Lineal Simple")
        print("6. Comparar modelos no lineales")
        print("7. Intervalos")
        print("8. Gráficar")
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
            regresion.regresion_lineal_simple(X, Y) 

        elif opcion == '5':
            regresion.regresion_no_lineal_linealizable(X, Y)

        elif opcion == '6':
            regresion.comparar_modelos_no_lineales(X, Y)

        elif opcion == '7':
            menu_intervalos()
 
        elif opcion == '8':
            menu_graficar()
            
        elif opcion == '0':
            break

X,Y = None, None

if __name__ == '__main__':
    X = ingresar_datos()
    Y = ingresar_datos()
    menu()
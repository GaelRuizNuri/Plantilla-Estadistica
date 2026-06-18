from scipy.stats import chi2, norm, nbinom, beta, uniform, poisson, f
import numpy as np
import calculos
import bondad_ajuste
import independencia
import anova
import regresion_univariada
import regresion_multivariada
import intervalos
import graficas

LONG = 50
TMENU = "MENU"
TGRAFICAS = "GRÁFICAS"
TINTERVALOS = "INTERVALOS"
X, X1, X2, Y, observados, esperados = None, None, None, None, None, None
nombre_x1, nombre_x2, nombre_y = "X1", "X2", "Y"

def ingresar_datos():
    print("Ingrese los datos separados por espacios, comas o saltos de línea.")
    print("Ingrese la letra 'q' para finalizar la inserción")

    data = []

    while True:
        linea = input().strip()

        if linea.lower() == "q":
            break

        linea = linea.replace(",", " ")
        linea = linea.replace(";", " ")

        valores = linea.split()

        for valor in valores:
            data.append(float(valor))

    return data


def menu_graficar():
    while True:
        global X, Y, X1, X2, nombre_x1, nombre_x2, nombre_y, observados, esperados
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
            graficas.bondad_ajuste(observados, esperados, show=True)
 
        elif opcion == '2':           
            print("\n--- Gráfica Regresión Lineal ---")
            Yx, a, b, _, _, _, r, _, _ = regresion_univariada.regresion_lineal_simple(X, Y)
            graficas.regresion_lineal_simple(Yx, X, Y, a, b, r, show=True)

        elif opcion == '3':           
            print("\n--- Gráfica Regresión No Lineal Simple ---")
            tipo = input("Ingrese el tipo de modelo no lineal (exponencial, potencial, logaritmico, hiperbolica, reciproco): ").strip().lower()
            tipo = None if tipo == "" else tipo            
            est = regresion_univariada.regresion_no_lineal_linealizable(X, Y, tipo = tipo) # exponencial, potencial, logaritmico, hiperbolica, reciproco
            graficas.regresion_no_lineal_simple(est["modelo"], X, Y, r2 = est["r2_original"], show=True)
            
        elif opcion == '4':
            print("\n--- Gráfica Distribuciones ---")
            #graficas.distribucion_discreta(poisson(3),0,10,show=True, titulo = "Poisson λ = 3")
            #graficas.distribucion_continua(norm(0,1),-5,5, show=True, titulo = "Normal μ = 0 y  σ = 1")

        elif opcion == '5':
            print("\n--- Gráfica Puntos ---")
            #graficas.puntos(X,Y, show = True)
            
        elif opcion == '0':
            break


def menu_intervalos():
    global X, Y, X1, X2, nombre_x1, nombre_x2, nombre_y, observados, esperados
    confianza = 0.95
    if input("¿Desea cambiar el nivel de confianza (0.95 por defecto)? (s/n): ").lower() == "s":
        confianza = float(input("Ingrese el nuevo nivel de confianza: "))

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
            intervalos.ic_parametro_beta(X, Y, confianza=confianza)
 
        elif opcion == '2':
            print("\n--- IC para α ---")
            intervalos.ic_parametro_alpha(X, Y, confianza=confianza)
 
        elif opcion == '3':
            print("\n--- IC para E(Y | x0) ---")
            x0 = float(input("Ingrese x0: "))
            intervalos.ic_media_condicional(X, Y, x0, confianza=confianza)
 
        elif opcion == '4':
            print("\n--- IP para y0 individual ---")
            x0 = float(input("Ingrese x0: "))
            intervalos.ip_prediccion_individual(X, Y, x0, confianza=confianza)
 
        elif opcion == '0':
            break


def menu():
    global X, Y, X1, X2, nombre_x1, nombre_x2, nombre_y, observados, esperados
    while True:
        print("\n" + "="*LONG)
        print(" "*((LONG - len(TMENU)) // 2) + TMENU)
        print("="*LONG)
        print("1. Bondad de Ajuste")
        print("2. Independencia")
        print("3. ANOVA")
        print("4. Regresión Lineal (Univariada)")
        print("5. Regresión No Lineal Simple (Univariada)")
        print("6. Comparar modelos no lineales (Univariada)")
        print("7. Regresión Multivariada (Lineal y No Lineal)")
        print("8. Intervalos (Regresión Lineal Univariada, α = 0.05 por defecto)")
        print("9. Gráficar")
        print("0. Salir")
        print("-"*LONG)
        
        opcion = input("Opción: ")
        
        if opcion == '1':
            print("\n--- Bondad de Ajuste ---")
            print("Ingrese los valores observados: ")
            observados = ingresar_datos()
            #probs = calculos.probabilidades_discretas(poisson(mu = 1), [i for i in range(len(observados))])
            probs = [1/6]*len(observados) # Probabilidades uniformes como ejemplo
            chi2_obs, gl, observados, esperados = bondad_ajuste.chi_obs_bondad_ajuste(observados, probs)
            print(f"Valor p: {chi2.sf(chi2_obs, gl):.4f}")

        elif opcion == '2':
            print("\n--- Independencia ---")
            m = int(input("Número de columnas: "))
            print("Ingrese todas las columnas en orden: ")
            tabla_obs = []
            for i in range(m):
                print(f"Columna {i+1}:")
                tabla_obs.append(ingresar_datos())
            tabla_obs = list(map(list, zip(*tabla_obs)))
            chi2_obs, gl = independencia.chi_obs_independencia(tabla_obs)
            print(f"Valor p: {chi2.sf(chi2_obs, gl):.4f}")

        elif opcion == '3':
            print("\n--- ANOVA ---")
            k = int(input("Ingrese el número de grupos: "))
            grupos = []
            for i in range(k):
                print(f"Grupo {i+1}:")
                grupos.append(ingresar_datos())
            f_obs, gl = anova.f_obs_anova(grupos)
            print(f"Valor p: {f.sf(f_obs, *gl):.4f}")

        elif opcion == '4':
            print("\n--- Regresión Lineal (Univariada) ---")
            regresion_univariada.regresion_lineal_simple(X, Y) 

        elif opcion == '5':
            print("\n--- Regresión No Lineal Simple (Univariada) ---")
            tipo = input("Ingrese el tipo de modelo no lineal (exponencial, potencial, logaritmico, hiperbolica, reciproco): ").strip().lower()
            tipo = None if tipo == "" else tipo
            est = regresion_univariada.regresion_no_lineal_linealizable(X, Y, tipo = tipo) # exponencial, potencial, logaritmico, hiperbolica, reciproco
            if input("¿Desea cambiar X & Y a sus versiones lineales? (s/n): ").lower() == "s":
                X = est["X_transformado"]
                Y = est["Y_transformado"]

        elif opcion == '6':
            print("\n--- Comparar Modelos No Lineales (Univariada) ---")
            regresion_univariada.comparar_modelos_no_lineales(X, Y)

        elif opcion == '7':
            print("\n--- Regresión Multivariada (Lineal y No Lineal) ---")
            regresion_multivariada.regresion_multivariada(X1, X2, Y, nombre_x1, nombre_x2, nombre_y)

        elif opcion == '8':
            menu_intervalos()
 
        elif opcion == '9':
            menu_graficar()
            
        elif opcion == '0':
            break


if __name__ == '__main__':
    independientes = 2
    
    if independientes == 2:
        nombre_x1 = input("Nombre de la 1º var.Independiente: ").strip() or "X1"
        nombre_x2 = input("Nombre de la 2º var.Independiente: ").strip() or "X2"
        nombre_y = input("Nombre de la var.Dependiente: ").strip() or "Y"
        print(f"Datos de {nombre_x1}:")
        X1 = ingresar_datos()
        print(f"Datos de {nombre_x2}:")
        X2 = ingresar_datos()
        print(f"Datos de {nombre_y}:")
        Y = ingresar_datos()

    elif independientes == 1:
        nombre_x1 = input("Nombre de la var.Independiente: ").strip() or "X"
        nombre_y = input("Nombre de la var.Dependiente: ").strip() or "Y"
        print(f"Datos de {nombre_x1}:")
        X = ingresar_datos()
        print(f"Datos de {nombre_y}:")
        Y = ingresar_datos()

    menu()

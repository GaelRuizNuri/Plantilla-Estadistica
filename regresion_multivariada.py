import numpy as np

def regresion_multivariada(X1, X2, Y, nx1="x1", nx2="x2", ny="y"):
    X1_arr = np.array(X1, dtype=float)
    X2_arr = np.array(X2, dtype=float)
    Y_arr = np.array(Y, dtype=float)
    
    variables_locales = {
        nx1: X1_arr, nx2: X2_arr, ny: Y_arr,
        nx1.lower(): X1_arr, nx2.lower(): X2_arr, ny.lower(): Y_arr,
        nx1.upper(): X1_arr, nx2.upper(): X2_arr, ny.upper(): Y_arr,
        'np': np
    }
    
    cant_b_str = input("Cantidad de B del modelo: ").strip()
    try:
        cant_b = int(cant_b_str)
    except ValueError:
        cant_b = 3
        
    t_y = input(f"Transformación para {ny}: ").strip()
    Y_mod = eval(t_y, {"__builtins__": None}, variables_locales) if t_y else Y_arr
    n = len(Y_mod)
    
    X_matrix = np.ones((n, 1))
    t_xs = []
    
    for i in range(1, cant_b):
        t_x = input(f"Transformación para el término de b{i}: ").strip()
        if not t_x:
            if i == 1:
                col = X1_arr
                t_xs.append(nx1)
            elif i == 2:
                col = X2_arr
                t_xs.append(nx2)
            else:
                col = X1_arr
                t_xs.append(nx1)
        else:
            col = eval(t_x, {"__builtins__": None}, variables_locales)
            t_xs.append(t_x)
            
        X_matrix = np.column_stack((X_matrix, col))
        
    XT = np.transpose(X_matrix)
    XTMulX = np.matmul(XT, X_matrix)
    XTMulY = np.matmul(XT, Y_mod)
    B = np.linalg.solve(XTMulX, XTMulY)
    
    print("\n--- Sistema de Ecuaciones ---")
    for i in range(len(B)):
        eq = " + ".join([f"{XTMulX[i, j]:.4f}b{j}" for j in range(len(B))])
        print(f"{eq} = {XTMulY[i]:.4f}")
        
    print("\nValores de B calculados")
    for i in range(len(B)):
        print(f"b{i}' = {B[i]}")
        
    print("\n--- Transformaciones de B ---")
    B_orig = []
    for i in range(len(B)):
        t_b = input(f"Transformación para b{i} original: ").strip()
        locales_b = {f"b{i}": B[i], 'np': np}
        if t_b:
            b_orig = eval(t_b, {"__builtins__": None}, locales_b)
        else:
            b_orig = B[i]
        B_orig.append(b_orig)
        
    print("\nResultados:")
    for i in range(len(B_orig)):
        print(f"b{i} = {B_orig[i]}")
        
    print("\n\n--- Ecuación Final ---")
    modelo = t_y if t_y else ny.lower()
    ecuacion = f"{modelo} = {B_orig[0]}"
    for i in range(1, len(B_orig)):
        ecuacion += f" + {B_orig[i]}({t_xs[i-1].lower()})"
    print(f"{ecuacion}")
    evaluar_modelo(nx1, nx2, ny, B_orig, t_xs)


def evaluar_modelo(nx1, nx2, ny, B_orig, t_xs):
    resp = input("\n¿Evaluar el modelo? (s/n): ").strip().lower()
    if resp == 's':
        print("\n--- Evaluar el Modelo ---")
        try:
            val_x1 = float(input(f"Valor para {nx1.lower()}: "))
            val_x2 = float(input(f"Valor para {nx2.lower()}: "))

            variables_locales = {
                nx1: val_x1, nx2: val_x2,
                nx1.lower(): val_x1, nx2.lower(): val_x2,
                nx1.upper(): val_x1, nx2.upper(): val_x2,
                'np': np
            }
            
            resultado = B_orig[0]
            for i in range(1, len(B_orig)):
                termino = eval(t_xs[i-1], {"__builtins__": None}, variables_locales)
                resultado += B_orig[i] * termino
            print(f"{ny.lower()} ≈ {resultado}")

        except Exception as e:
            print(f"Error al evaluar: {e}")

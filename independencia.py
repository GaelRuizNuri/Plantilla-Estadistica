def chi_obs_independencia(tabla_obs):
    fil = len(tabla_obs)
    col = len(tabla_obs[0])

    total_cols = [sum(tabla_obs[i][j] for i in range(fil)) for j in range(col)]
    total_fils = [sum(fila) for fila in tabla_obs]
    n = sum(total_fils)
    tabla_esp = [[total_fils[i]*total_cols[j]/n for j in range(col)] for i in range(fil)]
    chi_obs = sum(((fo-fe)**2)/fe for fila_obs, fila_esp in zip(tabla_obs, tabla_esp) for fo, fe in zip(fila_obs, fila_esp))
    gl = (fil-1)*(col-1)
    print(f'Total de cada fila: {total_fils}\n' 
          f'Total de cada columna: {total_cols}\n' 
          f'Total de fila × columna: {n}'
    )
    print("Esperados")
    for fila in tabla_esp: print(fila)

    print(f'Chi Cuadrado observado: {chi_obs}\n'
          f'Grados de libertad = {gl}'
    )
    return chi_obs, gl

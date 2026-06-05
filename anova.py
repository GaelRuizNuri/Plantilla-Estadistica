def f_obs_anova(grupos):
    k = len(grupos)
    n = [len(grupo) for grupo in grupos]
    t = [sum(grupo) for grupo in grupos]
    N = sum(n)
    T = sum(t)
    datos = [x for grupo in grupos for x in grupo]
    suma_cuadrados = sum(x*x for x in datos)
    SST = suma_cuadrados - T*T/N
    SSA = sum([t[i]*t[i]/n[i] for i in range(k)]) - T*T/N
    SSE = SST - SSA
    f_obs = (SSA/SSE)*(N-k)/(k-1)
    gl = (k-1, N-k)
    print(f'k = {k}\n'
          f'n_i = {n}\n'
          f'T_i = {t}\n'
          f'N = {N}\n'
          f'T = {T}\n'
          f'ΣΣ(y_i_j)^2 = {suma_cuadrados}\n'
          f'SST = {SST}\n'
          f'SSA = {SSA}\n'
          f'SSE = {SSE}\n'
          f'Grados de libertad = {gl}\n'
          f'F_obs = {f_obs}')

    return f_obs, gl

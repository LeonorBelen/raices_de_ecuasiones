import numpy as np

# --- 1. Definición de la Función y su Derivada ---

def f(x):
    """Función: f(x) = x^3 - exp(0.8x) - 20"""
    return x**3 - np.exp(0.8 * x) - 20

def df(x):
    """Derivada: f'(x) = 3x^2 - 0.8 * exp(0.8x)"""
    return 3 * x**2 - 0.8 * np.exp(0.8 * x)

# Tolerancia de error absoluto
TOL = 0.0001

# --- 2. Método de Bisección ---

def biseccion(a, b, tol, raiz_id):
    print(f"\n--- Bisección (Raíz {raiz_id}) ---")
    print("{:<5} {:<10} {:<10} {:<10} {:<15}".format("k", "a", "b", "p", "f(p)"))
    
    fa = f(a)
    fb = f(b)
    
    if fa * fb > 0:
        return f"Error: El intervalo [{a}, {b}] no encierra la raíz (no hay cambio de signo)."
    
    p = 0
    
    for i in range(50): # Límite de iteraciones
        p = (a + b) / 2
        fp = f(p)
        
        # MOSTRAR ITERACIÓN
        print("{:<5} {:<10.6f} {:<10.6f} {:<10.6f} {:<15.6f}".format(i + 1, a, b, p, fp))
        
        # Criterio de parada: Ancho del intervalo (Error Absoluto)
        if np.abs(b - a) / 2 < tol:
            return f"Raíz encontrada: {p:.6f} en {i+1} iteraciones."
        
        if fp == 0:
            return f"Raíz exacta: {p:.6f} en {i+1} iteraciones."
        
        if fa * fp < 0:
            b = p
        else:
            a = p
            fa = fp # Actualizar f(a)
            
    return f"Convergencia lenta. Última aproximación: {p:.6f}"

# --- 3. Método de Newton-Raphson ---

def newton_raphson(x0, tol, raiz_id):
    print(f"\n--- Newton-Raphson (Raíz {raiz_id}) ---")
    print("{:<5} {:<15} {:<15} {:<15} {:<15}".format("k", "x_k", "f(x_k)", "f'(x_k)", "Error Abs"))
    x_k = x0
    
    for i in range(50): # Límite de iteraciones
        fx = f(x_k)
        dfx = df(x_k)
        
        if np.abs(dfx) < 1e-10: # Evitar división por cero
            return "División por cero (derivada cercana a cero)."
            
        x_k_nuevo = x_k - fx / dfx
        error_abs = np.abs(x_k_nuevo - x_k)
        
        # MOSTRAR ITERACIÓN
        print("{:<5} {:<15.8f} {:<15.8f} {:<15.8f} {:<15.8f}".format(i + 1, x_k, fx, dfx, error_abs))
        
        # Criterio de parada: Error Absoluto |x_k_nuevo - x_k|
        if error_abs < tol:
            return f"Raíz encontrada: {x_k_nuevo:.6f} en {i+1} iteraciones."
            
        x_k = x_k_nuevo
        
    return f"Convergencia lenta. Última aproximación: {x_k:.6f}"

# --- 4. Método de la Secante ---

def secante(x_menos_1, x0, tol, raiz_id):
    print(f"\n--- Método de la Secante (Raíz {raiz_id}) ---")
    print("{:<5} {:<15} {:<15} {:<15} {:<15}".format("k", "x_k", "f(x_k)", "x_k+1", "Error Abs"))
    x_k_menos_1 = x_menos_1
    x_k = x0
    
    for i in range(50): # Límite de iteraciones
        fx_menos_1 = f(x_k_menos_1)
        fx_k = f(x_k)
        
        if np.abs(fx_k - fx_menos_1) < 1e-10: # Evitar división por cero
            return "División por cero (denominador nulo)."
            
        # Fórmula de la Secante
        x_k_mas_1 = x_k - fx_k * (x_k_menos_1 - x_k) / (fx_menos_1 - fx_k)
        error_abs = np.abs(x_k_mas_1 - x_k)
        
        # MOSTRAR ITERACIÓN
        print("{:<5} {:<15.8f} {:<15.8f} {:<15.8f} {:<15.8f}".format(i + 1, x_k, fx_k, x_k_mas_1, error_abs))
        
        # Criterio de parada: Error Absoluto |x_k_mas_1 - x_k|
        if error_abs < tol:
            return f"Raíz encontrada: {x_k_mas_1:.6f} en {i+1} iteraciones."
            
        x_k_menos_1 = x_k
        x_k = x_k_mas_1
        
    return f"Convergencia lenta. Última aproximación: {x_k:.6f}"

# --- 5. Ejecución para la raíz ---

print("\n--- SOLUCIONES PARA f(x) = x^3 - exp(0.8x) - 20 ---")

# --- Raíz 1 (cercana a 3.26) ---
print("\n# Búsqueda de la Primera Raíz (x1)")
print(biseccion(3.0, 4.0, TOL, 1))
print(newton_raphson(3.5, TOL, 1))
print(secante(3.0, 4.0, TOL, 1))

# --- Salida del Código ---
"""
--- SOLUCIONES PARA f(x) = x^3 - exp(0.8x) - 20 ---

# Búsqueda de la Primera Raíz (x1)

--- Bisección (Raíz 1) ---
k     a          b          p          f(p)           
1     3.000000   4.000000   3.500000   1.836881       
2     3.000000   3.500000   3.250000   -0.117075      
3     3.250000   3.500000   3.375000   0.811802       
4     3.250000   3.375000   3.312500   0.330882       
5     3.250000   3.312500   3.281250   0.106674       
6     3.250000   3.281250   3.265625   -0.005526      
7     3.265625   3.281250   3.273438   0.050519       
8     3.265625   3.273438   3.269531   0.022425       
9     3.265625   3.269531   3.267578   0.008436       
10    3.265625   3.267578   3.266602   0.001452       
11    3.265625   3.266602   3.266113   -0.002038      
12    3.266113   3.266602   3.266357   -0.000293      
13    3.266357   3.266602   3.266479   0.000579       
14    3.266357   3.266479   3.266418   0.000143       
15    3.266357   3.266418   3.266387   -0.000075      
Raíz encontrada: 3.266387 en 15 iteraciones.

--- Newton-Raphson (Raíz 1) ---
k     x_k             f(x_k)          f'(x_k)         Error Abs      
1     3.50000000      1.83688062      26.69830847     0.06887550     
2     3.43112450      1.05072027      24.32047806     0.04311904     
3     3.38800545      0.44314228      22.84469792     0.01939884     
4     3.36860661      0.08271038      22.18663806     0.00372793     
5     3.36487868      0.00336214      22.06209802     0.00015239     
6     3.36472629      0.00000547      22.05703714     0.00000025     
Raíz encontrada: 3.364726 en 6 iteraciones.

--- Método de la Secante (Raíz 1) ---
k     x_k             f(x_k)          x_k+1           Error Abs      
1     4.00000000      8.87858349      3.26577579      0.73422421     
2     3.26577579      -0.00021389     3.26579603      0.00002024     
Raíz encontrada: 3.265796 en 2 iteraciones.
"""
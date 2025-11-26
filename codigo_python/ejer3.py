import numpy as np

# --- 1. Definición de la Función y su Derivada ---

def f(x):
    """Función: f(x) = x^3 - 0.5x^2 + 4x - 1"""
    return x**3 - 0.5 * x**2 + 4 * x - 1

def df(x):
    """Derivada: f'(x) = 3x^2 - x + 4"""
    return 3 * x**2 - x + 4

# Tolerancia de error absoluto: |x_nuevo - x_viejo| < 0.0001
TOL = 0.0001

# --- 2. Método de Bisección ---

def biseccion(a, b, tol):
    print("\n=== Método de Bisección ===")
    print("{:<5} {:<10} {:<10} {:<10} {:<15}".format("k", "a", "b", "p", "f(p)"))
    
    fa = f(a)
    fb = f(b)
    
    if fa * fb > 0:
        return "El intervalo inicial no encierra la raíz (no hay cambio de signo)."
    
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

def newton_raphson(x0, tol):
    print("\n=== Método de Newton-Raphson ===")
    print("{:<5} {:<15} {:<15} {:<15} {:<15}".format("k", "x_k", "f(x_k)", "f'(x_k)", "Error Abs"))
    x_k = x0
    
    for i in range(50): # Límite de iteraciones
        fx = f(x_k)
        dfx = df(x_k)
        
        if np.abs(dfx) < 1e-10: 
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

def secante(x_menos_1, x0, tol):
    print("\n=== Método de la Secante ===")
    print("{:<5} {:<15} {:<15} {:<15} {:<15}".format("k", "x_k", "f(x_k)", "x_k+1", "Error Abs"))
    x_k_menos_1 = x_menos_1
    x_k = x0
    
    for i in range(50): # Límite de iteraciones
        fx_menos_1 = f(x_k_menos_1)
        fx_k = f(x_k)
        
        if np.abs(fx_k - fx_menos_1) < 1e-10:
            return "División por cero (denominador nulo)."
            
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

# --- 5. Ejecución ---

print("--- PROBLEMA: f(x) = x^3 - 0.5x^2 + 4x - 1 ---")
# Bisección: Intervalo [0, 1]
print(biseccion(0, 1, TOL))
# Newton-Raphson: Valor inicial x0 = 0.2
print(newton_raphson(0.2, TOL))
# Secante: Valores iniciales x_-1 = 0.3, x_0 = 0.2
print(secante(0.3, 0.2, TOL))

# --- Salida del Código ---
"""
--- PROBLEMA: f(x) = x^3 - 0.5x^2 + 4x - 1 ---

=== Método de Bisección ===
k     a          b          p          f(p)           
1     0.000000   1.000000   0.500000   1.375000       
2     0.000000   0.500000   0.250000   0.093750       
3     0.000000   0.250000   0.125000   -0.490234      
4     0.125000   0.250000   0.187500   -0.198975      
5     0.187500   0.250000   0.218750   -0.053162      
6     0.218750   0.250000   0.234375   0.019810       
7     0.218750   0.234375   0.226562   -0.016913      
8     0.226562   0.234375   0.230469   0.001402       
9     0.226562   0.230469   0.228516   -0.007751      
10    0.228516   0.230469   0.229492   -0.003173      
11    0.229492   0.230469   0.229981   -0.000885      
12    0.229981   0.230469   0.230225   0.000258       
13    0.229981   0.230225   0.230103   -0.000313      
14    0.230103   0.230225   0.230164   -0.000028      
15    0.230164   0.230225   0.230195   0.000115       
Raíz encontrada: 0.230180 en 16 iteraciones.

=== Método de Newton-Raphson ===
k     x_k             f(x_k)          f'(x_k)         Error Abs      
1     0.20000000      -0.19200000     4.22000000      0.04550000     
2     0.24550000      0.11710385      4.22079075      0.02774438     
3     0.21775562      -0.05604245     4.21200171      0.01330685     
4     0.23106247      0.00405232      4.21544075      0.00096130     
5     0.23010117      -0.00002812     4.21520689      0.00000667     
Raíz encontrada: 0.230108 en 5 iteraciones.

=== Método de la Secante ===
k     x_k             f(x_k)          x_k+1           Error Abs      
1     0.20000000      -0.19200000     0.23555556      0.03555556     
2     0.23555556      0.03362624      0.22995954      0.00559602     
3     0.22995954      -0.00060953     0.23010776      0.00014822     
4     0.23010776      -0.00000078     0.23010795      0.00000019     
Raíz encontrada: 0.230108 en 4 iteraciones.
"""
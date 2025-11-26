import numpy as np

# --- 1. Definición de la Función y su Derivada ---

def f(x):
    """Función: f(x) = x * cos(x) (x en radianes)"""
    return x * np.cos(x)

def df(x):
    """Derivada: f'(x) = cos(x) - x * sin(x) (Regla del Producto)"""
    return np.cos(x) - x * np.sin(x)

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

print("--- PROBLEMA: f(x) = x * cos(x) ---")
# La raíz es pi/2 ≈ 1.570796
print(biseccion(1.5, 2.0, TOL))
print(newton_raphson(1.5, TOL))
print(secante(1.5, 2.0, TOL))


# --- Salida del Código ---
"""
--- PROBLEMA: f(x) = x * cos(x) ---

=== Método de Bisección ===
k     a          b          p          f(p)           
1     1.500000   2.000000   1.750000   -0.129997      
2     1.500000   1.750000   1.625000   -0.033621      
3     1.500000   1.625000   1.562500   0.005574       
4     1.562500   1.625000   1.593750   -0.014168      
5     1.562500   1.593750   1.578125   -0.004245      
6     1.562500   1.578125   1.570312   0.000632       
7     1.570312   1.578125   1.574219   -0.001807      
8     1.570312   1.574219   1.572266   -0.000588      
9     1.570312   1.572266   1.571289   0.000021       
10    1.571289   1.572266   1.571777   -0.000284      
11    1.571289   1.571777   1.571533   -0.000132      
12    1.571289   1.571533   1.571411   -0.000055      
13    1.571289   1.571411   1.571350   -0.000017      
14    1.571289   1.571350   1.571320   0.000002       
Raíz encontrada: 1.571335 en 15 iteraciones.

=== Método de Newton-Raphson ===
k     x_k             f(x_k)          f'(x_k)         Error Abs      
1     1.50000000      0.03749449      -1.49875902     0.02501625     
2     1.52501625      0.00249764      -1.62677464     0.00153531     
3     1.52348094      0.00000025      -1.61868478     0.00000015     
Raíz encontrada: 1.523481 en 3 iteraciones.

=== Método de la Secante ===
k     x_k             f(x_k)          x_k+1           Error Abs      
1     2.00000000      -0.83229367     1.53696773      0.46303227     
2     1.53696773      -0.00551048     1.57053526      0.03356753     
3     1.57053526      0.00016480      1.57079633      0.00026107     
4     1.57079633      0.00000000      1.57079633      0.00000000     
Raíz encontrada: 1.570796 en 4 iteraciones.
"""
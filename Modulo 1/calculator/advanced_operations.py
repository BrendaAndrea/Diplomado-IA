"""
Módulo de operaciones avanzadas.

Este módulo contiene funciones para realizar operaciones matemáticas avanzadas.
"""

import math

def potencia(base, exponente):
    """
    Calcula la potencia de un número.

    :param base: La base.
    :param exponente: El exponente.
    :return: El resultado de base elevado a exponente.
    """
    return base ** exponente

def raiz_cuadrada(numero):
    """
    Calcula la raíz cuadrada de un número.

    :param numero: El número.
    :return: La raíz cuadrada de numero.
    :raises ValueError: Si el número es negativo.
    """
    if numero < 0:
        raise ValueError("No se puede calcular la raíz cuadrada de un número negativo")
    return math.sqrt(numero)

def logaritmo(numero, base=math.e):
    """
    Calcula el logaritmo de un número en una base específica.

    :param numero: El número.
    :param base: La base del logaritmo. Por defecto es el número e.
    :return: El logaritmo de numero en la base especificada.
    :raises ValueError: Si el número es menor o igual a 0.
    """
    if numero <= 0:
        raise ValueError("El número debe ser mayor que cero")
    return math.log(numero, base)

def resolver_polinomio_cuadratico (a, b, c):
    
    discriminante = b**2 - 4*a*c
    
    
    if discriminante >= 0:
        raiz_real = discriminante**0.5
        raiz_imaginaria = 0
    else:
        raiz_real = 0
        raiz_imaginaria = (-discriminante)**0.5
    
     
    raiz1_real = (-b + raiz_real) / (2*a)
    raiz1_imaginaria = raiz_imaginaria / (2*a)
    raiz2_real = (-b - raiz_real) / (2*a)
    raiz2_imaginaria = -raiz_imaginaria / (2*a)
    
   
    raiz1 = (raiz1_real, raiz1_imaginaria)
    raiz2 = (raiz2_real, raiz2_imaginaria)
    
    return raiz1, raiz2

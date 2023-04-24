import queue
import heapq
import random
from queue import PriorityQueue
from heapq import heappush, heappop
from math import inf, radians, cos, sin, asin, sqrt
from sys import maxsize
import networkx as nx
import random, math
import matplotlib.pyplot as plt
from networkx.algorithms import approximation as approx
import time
import sys

#Infromación General
print("")
print("Búsquedas Informadas")
print("")
print("Universidad Panamericana")
print("")
print("Inteligencia Artificial")
print("")
print("Integrantes:")
print("-Anaya Viveros Francisco (0181879)")
print("-Vázquez Anaya Joel (0201031)")
print("-Vázquez Gurrola Javier (0215391)")
print("")
print("Fecha de entrega 26/04/2023")
print("")
print("Versión 1.9")
print("")

"""
El objetivo del código es poder generar búsquedas de maneras informadas comenzando creando una 
heurística, utilizando la fórmula Haversine. Después iniciamos con la búsqueda greedy, tenemos
un algoritmo A* con peso, después el A*, se reconstruye el camino entre los nodos que digamos
utilizando los nodos anteriores.
Luego iniciamos el algoritmo de beam, de branch and bound, steepest hil climbing, stochastic hil
climbing, genetic algorithm y por último el simulated anealing (resuelto con traveling salesman)
"""
 
"""
Para la heurística se utiliza la fórmula Haversine para calcular la distancia entre dos puntos en la superficie de la Tierra,
teniendo en cuenta que la Tierra es esférica en lugar de plana.
La función comienza definiendo dos diccionarios llamados "latitudes" y "longitudes" 
que contienen las coordenadas de latitud y longitud de cada nodo en el grafo.
A continuación, se define la constante R como el radio de la Tierra en kilómetros.
La función luego utiliza la biblioteca matemática de Python para convertir las coordenadas de latitud y longitud del nodo y del nodo objetivo a radianes,
que es la unidad requerida por la fórmula Haversine. Estos valores se asignan a las variables lat1, lon1 y lat2, lon2.
Luego, se calcula la diferencia de latitud y longitud entre los dos puntos mediante la resta de las coordenadas 
Finalmente, se calcula la distancia Haversine utilizando la fórmula matemática 
y se devuelve el resultado multiplicado por el radio de la Tierra (R) en kilómetros. 
La distancia calculada es una estimación de la distancia entre el nodo y el nodo objetivo.

"""


def heuristic(node): #Javier Vázquez Gurrola 
    # Coordenadas de los nodos
    latitudes = {'Cancún': 21.1213285, 'Valladolid': 20.688114, 'Felipe Carrillo Puerto': 19.5778903, 'Campeche': 19.8305682, 'Merida': 20.9800512, 
                 'Ciudad del Carmen': 18.6118375, 'Chetumal': 18.5221567, 'Villa Hermosa': 17.9925264, 'Tuxtla': 16.7459857, 'Francisco Escarcega': 18.6061556, 
                 'Acayucan': 17.951096, 'Tehuantepec': 16.320636, 'Alvarado': 18.7760455, 'Oaxaca': 17.0812951, 'Puerto Angel': 15.6679974, 
                 'Izucar de Matamoros': 18.5980563, 'Tehuacan': 18.462191, 'Pinotepa Nacional': 16.3442895, 'Cuernavaca': 18.9318685, 
                 'Puebla': 19.040034, 'Acapulco': 16.8354485, 'Cdmx': 19.3898319, 'Iguala': 18.3444, 'Ciudad Altamirano': 18.3547491,
                 'Cordoba': 18.8901707, 'Chilpancingo': 17.5477072, 'Tlaxcala': 19.4167798, 'Pachuca de Soto': 20.0825056, 'Queretaro': 20.6121228,
                 'Toluca de Lerdo': 19.294109, 'Zihuatanejo': 17.6405745, 'Veracruz': 19.1787635, 'Tuxpan de Rodriguez Cano': 20.9596561, 
                 'Atlacomulco': 19.7980152, 'Salamanca': 20.5664927, 'San Luis Potosi': 22.1127046, 'Playa Azul': 17.9842581, 'Tampico': 22.2662251, 
                 'Guanajuato': 21.0250928, 'Morelia': 19.7036417, 'Guadalajara': 20.6737777, 'Aguascalientes': 21.8857199, 'Zacatecas': 22.7636293,
                 'Durango': 24.0226824, 'Colima': 19.2400444, 'Manzanillo': 19.0775491, 'Ciudad Victoria': 23.7409928, 'Tepic': 21.5009822, 
                 'Hidalgo del Parral': 26.9489283, 'Mazatlan': 23.2467283, 'Soto la Marina': 23.7673729, 'Matamoros': 25.8433787, 'Monterrey': 25.6487281, 
                 'Chihuahua': 28.6708592, 'Topolobampo': 25.6012747, 'Culiacan': 24.8049008, 'Reynosa': 26.0312262, 'Monclova': 26.907775, 
                 'Ciudad Juárez': 31.6538179, 'Janos': 30.8898127, 'Ciudad Obregon': 27.4827355, 'Torreon': 25.548597, 'Ojinaga': 29.5453292, 
                 'Nuevo Laredo': 27.4530856, 'Agua Prieta': 31.3115272, 'Guaymas': 27.9272572, 'Piedras Negras': 28.6910517, 'Santa Ana': 30.5345457, 
                 'Hermosillo': 29.082137, 'Mexicali': 32.6137391, 'Tijuana': 32.4966818, 'San Felipe': 31.009535, 'Ensenada': 31.8423096,
                 'San Quintin': 30.5711324, 'Santa Rosalia': 27.3408761, 'Santo Domingo': 25.3487297, 'La Paz': 24.1164209, 'Cabo San Lucas': 22.8962253 }
    
    longitudes = {'Cancún': -86.9192738, 'Valladolid': -88.2204456, 'Felipe Carrillo Puerto': -88.0630853, 'Campeche': -90.5798365, 'Merida': -89.7029587, 
                'Ciudad del Carmen': -91.8927345, 'Chetumal': -88.3397982, 'Villa Hermosa': -92.9881407, 'Tuxtla': -93.1996103, 'Francisco Escarcega': -90.8176486, 
                'Acayucan': -94.9306961, 'Tehuantepec': -95.27521, 'Alvarado': -95.7731952, 'Oaxaca': -96.7707511, 'Puerto Angel': -96.4933733, 
                'Izucar de Matamoros': -98.5076767, 'Tehuacan': -97.4437333, 'Pinotepa Nacional': -98.1315923, 'Cuernavaca': -99.3106054, 'Puebla': -98.2630056,
                'Acapulco': -99.9323491, 'Cdmx': -99.7180148, 'Iguala': -99.5652232, 'Ciudad Altamirano': -100.6817619, 'Cordoba': -96.9751108, 
                'Chilpancingo': -99.5324349, 'Tlaxcala': -98.4471127, 'Pachuca de Soto': -98.8268184, 'Queretaro': -100.4802576, 'Toluca de Lerdo': -99.6662331, 
                'Zihuatanejo': -101.5601369, 'Veracruz': -96.2113357, 'Tuxpan de Rodriguez Cano': -97.4158767, 'Atlacomulco': -99.89317, 'Salamanca': -101.2176511, 
                'San Luis Potosi': -101.0261099, 'Playa Azul': -102.357616, 'Tampico': -97.939526, 'Guanajuato': -101.3296402, 'Morelia': -101.2761644, 
                'Guadalajara': -103.4054536, 'Aguascalientes': -102.36134, 'Zacatecas': -102.623638, 'Durango': -104.7177652, 'Colima': -103.7636273, 'Manzanillo': -104.4789574, 
                'Ciudad Victoria': -99.1783576, 'Tepic': -104.9119242, 'Hidalgo del Parral': -105.8211168, 'Mazatlan': -106.4923175, 'Soto la Marina': -98.2157573, 
                'Matamoros': -97.5849847, 'Monterrey': -100.4431819, 'Chihuahua': -106.2047036, 'Topolobampo': -109.0687891, 'Culiacan': -107.4933545, 'Reynosa': -98.3662435, 
                'Monclova': -101.4940069, 'Ciudad Juárez': -106.5890206, 'Janos': -108.208458, 'Ciudad Obregon': -110.0844111, 'Torreon': -103.4719562, 'Ojinaga': -104.4305246, 
                'Nuevo Laredo': -99.6881218, 'Agua Prieta': -109.5855873, 'Guaymas': -110.9779564, 'Piedras Negras': -100.5801829, 'Santa Ana': -111.1580567, 
                'Hermosillo': -111.059027, 'Mexicali': -115.5203312, 'Tijuana': -117.087892, 'San Felipe': -114.8727296, 'Ensenada': -116.6799816, 'San Quintin': -115.9588544, 
                'Santa Rosalia': -112.2825762, 'Santo Domingo': -111.9975909, 'La Paz': -110.3727673, 'Cabo San Lucas': -109.9505077}
    
    # Distancia Haversine
    R = 6372.8  # Radio de la Tierra en kilómetros
    
    # Convertir coordenadas del nodo y del objetivo a radianes
    lat1, lon1 = radians(latitudes[node]), radians(longitudes[node])
    lat2, lon2 = radians(latitudes[goal_verificado]), radians(longitudes[goal_verificado])
    
    # Calcular diferencia de latitud y longitud
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    # Calcular la distancia Haversine
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * asin(sqrt(a))
    return R * c

#La función get_neighbors(node) devuelve los vecinos del nodo "node" en el grafo "graph"
# Función para Greedy best-first
def get_neighbors(node): #Javier Vázquez Gurrola 
    return graph[node].keys()

#Fúnción para imprimir el grafo procesado
#Muestra el grafo
def imprimir(grafo):#Javier Vázquez Gurrola 
	print("Grafo utilizado: \n")
	for key, lista in grafo.items():
		print(key)
		print(lista)
print("--------------------------------")

"""
La función greedy implementa el algoritmo de búsqueda Greedy Best First Search para encontrar el camino más corto entre un nodo de inicio 
y un nodo objetivo en un grafo. El algoritmo utiliza una función heurística para estimar la distancia desde cada nodo al nodo objetivo
y selecciona el siguiente nodo a explorar basándose en la heurística.
La función toma como entrada cuatro argumentos:
graph: un diccionario que representa el grafo. Las claves del diccionario son los nodos del grafo y 
    los valores son diccionarios que representan los vecinos y el peso de las aristas que los conectan.
start: el nodo de inicio de la búsqueda.
goal: el nodo objetivo de la búsqueda.
heuristic: una función heurística que toma como entrada un nodo y devuelve una estimación de la distancia desde ese nodo al nodo objetivo.

La función comienza comprobando si el nodo de inicio y el nodo objetivo son iguales. 
Si son iguales, devuelve una lista que contiene solo el nodo de inicio.
Luego, la función inicializa una pila con el nodo de inicio y su valor heurístico, y un conjunto de nodos explorados y un diccionario de padres. 
El algoritmo utiliza la pila para almacenar los nodos que se explorarán en orden de su valor heurístico.
El algoritmo utiliza un bucle while para explorar los nodos en la pila. 
En cada iteración, el algoritmo obtiene el nodo con el menor valor heurístico de la pila, lo marca como explorado y explora sus vecinos. 
Para cada vecino no explorado, el algoritmo calcula su valor heurístico, lo agrega a la pila con su costo acumulado y lo agrega al diccionario de padres.
Si el nodo objetivo se encuentra durante la exploración, el algoritmo construye el camino desde el nodo objetivo hasta el nodo de inicio utilizando 
el diccionario de padres y devuelve la lista de nodos del camino. Si no se encuentra un camino, la función devuelve None.
En caso de no encontrar la solución el algoritmo entrará en estado de backtrack para recorrer otro camino repitiendo el proceso en busca de la solución.
"""

#Función Greedy con pasos
def greedy_con_pasos(graph, start, goal): #Javier Vázquez Gurrola 
    if start == goal:
        return [start]
    # Inicializar la pila con el nodo de inicio y su valor heurístico
    stack = [(start, 0, heuristic(start))]
    # Inicializar el conjunto de nodos explorados y el diccionario de padres
    explored = set()
    parents = {}
    while stack:
         # Obtener el nodo de la pila con el menor valor heurístico
        current, cost, h = stack.pop(0)
        print("Imprime actual")
        print(current)
        print("")
        if current == goal:
            # Construir el camino desde el nodo objetivo hasta el nodo de inicio
            path = [current]
            while path[-1] != start:
                path.append(parents[path[-1]])
            print("Costo del camino:", cost)
            print("Esta es la lista de exploración")
            print(explored)
            print("")
            return path[::-1]
         # Marcar el nodo actual como explorado
        explored.add(current)
        print("Esta es la lista de los nodos explorados")
        print(explored)
        print("")
         # Explorar los vecinos del nodo actual
        for neighbor, weight in graph[current].items():
            if neighbor not in explored:
                print("Imprime los vecinos del nodo actual")
                print(neighbor)
                print("")
                # Calcular el valor heurístico del vecino
                h_neighbor = heuristic(neighbor)
                # Agregar el vecino a la pila con su valor heurístico y su costo acumulado
                stack.append((neighbor, cost + weight, h_neighbor))
                # Agregar el vecino al diccionario de padres
                parents[neighbor] = current
                priority = heuristic(neighbor)
                print("Imprime el vecino prioridad con base a la heurística")
                print(priority)
                print("")
                print("Esta es la lista de los nodos explorados")
                print(explored)
                print("")
                print("----------------------------------------------------------------------------------------")
         # Ordenar la pila por el valor heurístico de los nodos restantes
        stack = sorted(stack, key=lambda x: x[2])
    # Si se llega a este punto, significa que no se encontró un camino
    return None
    
#Función Greedy sin pasos 
def greedy(graph, start, goal): #Javier Vázquez Gurrola
    if start == goal:
        return [start]
    # Inicializar la pila con el nodo de inicio y su valor heurístico
    stack = [(start, 0, heuristic(start))]
    # Inicializar el conjunto de nodos explorados y el diccionario de padres
    explored = set()
    parents = {}
    while stack:
        # Obtener el nodo de la pila con el menor valor heurístico
        current, cost, h = stack.pop(0)
        if current == goal:
            # Construir el camino desde el nodo objetivo hasta el nodo de inicio
            path = [current]
            while path[-1] != start:
                path.append(parents[path[-1]])
            print("Costo del camino:", cost)
            return path[::-1]
        # Marcar el nodo actual como explorado
        explored.add(current)
        # Explorar los vecinos del nodo actual
        for neighbor, weight in graph[current].items():
            if neighbor not in explored:
                # Calcular el valor heurístico del vecino
                h_neighbor = heuristic(neighbor)
                # Agregar el vecino a la pila con su valor heurístico y su costo acumulado
                stack.append((neighbor, cost + weight, h_neighbor))
                # Agregar el vecino al diccionario de padres
                parents[neighbor] = current
        # Ordenar la pila por el valor heurístico de los nodos restantes
        stack = sorted(stack, key=lambda x: x[2])
    # Si se llega a este punto, significa que no se encontró un camino
    return None
"""
La función edge_cost toma una tupla de dos nodos y devuelve el costo de la arista que los conecta en el grafo.
En otras palabras, dado un borde, esta función devuelve la distancia o el costo de viajar de un nodo a otro.
Esta función se utiliza más adelante en la función weighted_astar para calcular el nuevo costo cuando se expande un nodo.

"""
#Función para Weighted A*
def edge_cost(edge): #Javier Vázquez Gurrola 
    node1, node2 = edge
    return graph[node1][node2]
"""
La función successors toma un nodo y devuelve un diccionario de sus sucesores en el grafo, junto con los pesos asociados.
Los sucesores son nodos conectados al nodo actual por una arista. 
En otras palabras, esta función devuelve un diccionario donde las claves son nodos vecinos y los valores son los costos 
de las aristas que conectan esos nodos con el nodo actual.

"""
#Función para Weighted A*
def successors(node): #Javier Vázquez Gurrola 
    return graph.get(node, {})
"""
La función weighted_astar es la implementación principal de Weighted A*. 
Toma los siguientes argumentos:
start: el nodo de inicio.
goal: el nodo de destino.
heuristic: una función que toma un nodo y devuelve una estimación heurística del costo de llegar desde ese nodo hasta el nodo objetivo.
successors: una función que toma un nodo y devuelve un diccionario de sus sucesores en el grafo, junto con los costos asociados.
edge_cost: una función que toma una tupla de dos nodos y devuelve el costo de la arista que los conecta en el grafo.
w: el peso del algoritmo. Un valor mayor de w hace que el algoritmo sea más orientado por heurística, pero puede llevar a una búsqueda menos completa.
En resumen, el algoritmo comienza con el nodo de inicio y lo coloca en una cola de prioridad (frontier) junto con el valor obtenido 
por heurística para llegar al nodo objetivo. 
Luego, mientras haya nodos en la cola de prioridad, el algoritmo selecciona el nodo con la estimación heurística más baja y lo expande.
Para cada sucesor del nodo actual, se calcula el costo total de llegar a ese sucesor sumando el costo actual del nodo al peso de la arista 
que conecta el nodo actual con el sucesor. 
Si el costo total es menor que el costo actual conocido del sucesor, el costo del sucesor se actualiza y se agrega a la cola de prioridad. 
Si el sucesor ya ha sido visitado, su costo se actualiza sólo si el nuevo costo es menor. 
El proceso continúa hasta que se llega al nodo objetivo o hasta que la cola de prioridad está vacía. 
Cuando se llega al nodo objetivo, se devuelve el camino que se tomó para llegar a él.

"""

def weighted_astar_con_pasos(start, goal, heuristic, successors, edge_cost, w): #Joel Vázquez Anaya
    # Creamos la cola de prioridad inicial con el nodo de inicio y su valor heurístico
    frontier = [(heuristic(start), start, 0)]
    print("Creamos la cola de prioridad inicial con el nodo de incio y su heurística")
    print(frontier)
    print("")
    # Inicializamos el conjunto de nodos visitados, guardando el nodo, su predecesor y el costo acumulado
    visited = {start: (None, 0)}
    print("Inicializamos el conjunto de nodos visitados, guardando el nodo, su predecesor y el costo acumulado")
    print(visited)
    print("")
    # Empezamos el bucle principal de búsqueda
    while frontier:
        # Sacamos el nodo con menor valor heurístico de la cola de prioridad
        _, node, cost = heappop(frontier)
        print("Sacamos el nodo con menor valor heurístico de la cola de prioridad")
        print(_, node, cost)
        print("")
        # Si llegamos al nodo objetivo, reconstruimos el camino y lo devolvemos
        if node == goal:
            print("Si llegamos al nodo objetivo, reconstruimos el camino y lo devolvemos")
            path = [node]
            print(path)
            print("")
            while node != start:
                node, _ = visited[node]
                print(node, _)
                path.append(node)
                print("Muestra la construcción del camino")
                print(path)
                print("")
            return list(reversed(path))
        
        # Para cada sucesor del nodo actual, calculamos el costo y actualizamos los nodos visitados
        for successor, successor_cost in successors(node).items():
            print("Para cada sucesor del nodo actual, calculamos el costo y actualizamos los nodos visitados")
            new_cost = visited[node][1] + edge_cost((node, successor))
            print("Nuevo costo")
            print(new_cost)
            print("")
            if successor not in visited or new_cost < visited[successor][1]:
                visited[successor] = (node, new_cost)
                print("Muestra lista de los nodos visitados con sus sucesores y su nuevo costo")
                print(visited[successor])
                print("")
                # Calculamos la prioridad de la cola de prioridad para el sucesor actual y lo agregamos
                print("Calculamos la prioridad de la cola de prioridad para el sucesor actual y lo agregamos")
                priority = new_cost + w * heuristic(successor)
                print(priority)
                print("")
                heappush(frontier, (priority, successor, new_cost))
                print("----------------------------------------------------------------------------------------")
    # Si no encontramos un camino, devolvemos una lista vacía
    print("Si no encontramos un camino, devolvemos una lista vacía")
    return []

def weighted_astar(start, goal, heuristic, successors, edge_cost, w): #Javier Vázquez Gurrola 
    # Creamos la cola de prioridad inicial con el nodo de inicio y su valor heurístico
    frontier = [(heuristic(start), start, 0)]
    # Inicializamos el conjunto de nodos visitados, guardando el nodo, su predecesor y el costo acumulado
    visited = {start: (None, 0)}
    # Empezamos el bucle principal de búsqueda
    while frontier:
        # Sacamos el nodo con menor valor heurístico de la cola de prioridad
        _, node, cost = heappop(frontier)
        # Si llegamos al nodo objetivo, reconstruimos el camino y lo devolvemos
        if node == goal:
            path = [node]
            while node != start:
                node, _ = visited[node]
                path.append(node)
            return list(reversed(path))
        # Para cada sucesor del nodo actual, calculamos el costo y actualizamos los nodos visitados
        for successor, successor_cost in successors(node).items():
            new_cost = visited[node][1] + edge_cost((node, successor))
            if successor not in visited or new_cost < visited[successor][1]:
                visited[successor] = (node, new_cost)
                # Calculamos la prioridad de la cola de prioridad para el sucesor actual y lo agregamos
                priority = new_cost + w * heuristic(successor)
                heappush(frontier, (priority, successor, new_cost))
    # Si no encontramos un camino, devolvemos una lista vacía
    return []

"""

Implementación del algoritmo A* para encontrar el camino más corto desde un nodo inicial hasta un nodo objetivo en un grafo ponderado y dirigido.
Toma los siguientes parámetros
start: Nodo inicial.
goal: Nodo objetivo.
graph: Grafo ponderado y dirigido representado como un diccionario de diccionarios.
contiene los nodos del grafo como claves, y los valores son otros diccionarios que representan los nodos adyacentes y sus respectivos costos. 
Por ejemplo: {'A': {'B': 6, 'C': 8}, 'B': {'D': 3}, ...}
heuristics: Diccionario que contiene las heurísticas (en este caso, las distancias haversine) para cada nodo.
Las claves son los nodos del grafo y los valores son los valores de la heurística para cada nodo.
Por ejemplo: {'A': 2442.076702837977, 'B': 2504.893734691217, 'C': 3334.165797319547, ...}
return: Tupla que contiene dos diccionarios: el diccionario 'came_from', que contiene los nodos antecesores 
de cada nodo en el camino más corto desde el nodo inicial hasta el nodo objetivo, y el diccionario 'cost_so_far'
que contiene el costo acumulado para llegar a cada nodo en el camino más corto desde el nodo inicial.
El algoritmo comienza inicializando un conjunto de diccionarios que almacenarán los nodos antecesores y los costos acumulados
y una cola de prioridad (heap) que almacenará los nodos por los que iremos pasando durante la búsqueda. 
La cola de prioridad está implementada como una lista de tuplas, donde el primer elemento de la tupla es la suma del costo acumulado 
y la heurística del nodo, y el segundo elemento es el propio nodo. 
Los elementos de la cola de prioridad se ordenan según la suma del costo acumulado y la heurística
de manera que los nodos con menor suma estén al principio de la cola.
Luego, el algoritmo comienza a iterar mientras la cola de prioridad tenga elementos. 
En cada iteración, se obtiene el nodo actual de la cola de prioridad, es decir, el nodo con menor suma de costo acumulado y heurística.
Si hemos llegado al nodo objetivo, terminamos la búsqueda y devolvemos los diccionarios de nodos antecesores y costos acumulados.
Si no hemos llegado al nodo objetivo, iteramos sobre los nodos adyacentes al nodo actual y calculamos el costo acumulado 
para llegar a cada uno de ellos desde el nodo inicial, sumando el costo acumulado para llegar al nodo actual y el costo del arco 
entre el nodo actual y el vecino. Si el vecino no está en el diccionario de costos acumulados o si hemos encontrado un camino más corto 
para llegar al vecino, actualizamos el diccionario de costos acumulados y el diccionario de nodos antecesores. 
Luego, añadimos el vecino a la cola de prioridad con una prioridad que es la suma del costo acumulado y la heurística del vecino.
Finalmente, devolvemos los diccionarios de nodos antecesores y costos acumulados.

"""

def astar_con_pasos(start, goal, graph, heuristic) : #Joel Vázquez Anaya
    frontier = [(0, start)]
    came_from = {start: None}
    print("Mostramos el primer nodo")
    print(came_from)
    print("")
    
    # Inicializamos el costo acumulado del nodo inicial a 0
    cost_so_far = {start: 0}
    print("Inicializamos el costo acumulado del nodo inicial a 0")
    print(cost_so_far)
    print("")
    # Iteramos mientras la cola de prioridad tenga elementos
    while frontier:
        _, current = heapq.heappop(frontier)
        print("Iteramos mientras la cola de prioridad tenga elementos")
        print(_, current)
# Obtenemos el nodo actual de la cola de prioridad, es decir, el nodo con menor suma de costo acumulado y heurística
# Si hemos llegado al nodo objetivo, terminamos la búsqueda y devolvemos los diccionarios de nodos antecesores y costos acumulados
        if current == goal:
            break
# Iteramos sobre los nodos adyacentes al nodo actual
        for next_node, cost in graph[current].items():
        # Calculamos el costo acumulado para llegar al vecino desde el nodo inicial, sumando el costo acumulado
        # para llegar al nodo actual y el costo del arco entre el nodo actual y el vecino
            new_cost = cost_so_far[current] + cost
            print("Calculamos el costo acumulado para llegar al vecino desde el nodo inicial, sumando el costo acumulado")
            print(new_cost)
            # Si el vecino no está en el diccionario de costos acumulados o si hemos encontrado un camino más corto
            # para llegar al vecino, actualizamos el diccionario de costos acumulados y el diccionario de nodos
            # antecesores
            if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                cost_so_far[next_node] = new_cost
                priority = new_cost + heuristic(next_node)
                heapq.heappush(frontier, (priority, next_node))
                came_from[next_node] = current
                print("Actualizamos el diccionario de costos acumulados y el diccionario de nodos antecesores")
                print(cost_so_far)
                print("Agregamos el nodo prioridad al directorio")
                print(priority)
                print("Actualizamos el nodo recurrente con su nuevo costo")
                print(came_from)
# Devolvemos los diccionarios de nodos antecesores y costos acumulados
    return came_from, cost_so_far


def astar(start, goal, graph, heuristic) : #Javier Vázquez Gurrola 
    frontier = [(0, start)]
    came_from = {start: None}
    # Inicializamos el costo acumulado del nodo inicial a 0
    cost_so_far = {start: 0}
    # Iteramos mientras la cola de prioridad tenga elementos
    while frontier:
        _, current = heapq.heappop(frontier)
# Obtenemos el nodo actual de la cola de prioridad, es decir, el nodo con menor suma de costo acumulado y heurística
# Si hemos llegado al nodo objetivo, terminamos la búsqueda y devolvemos los diccionarios de nodos antecesores y costos acumulados
        if current == goal:
            break
# Iteramos sobre los nodos adyacentes al nodo actual
        for next_node, cost in graph[current].items():
        # Calculamos el costo acumulado para llegar al vecino desde el nodo actual
        # Calculamos el costo acumulado para llegar al vecino desde el nodo inicial, sumando el costo acumulado
        # para llegar al nodo actual y el costo del arco entre el nodo actual y el vecino
            new_cost = cost_so_far[current] + cost
            # Si el vecino no está en el diccionario de costos acumulados o si hemos encontrado un camino más corto
            # para llegar al vecino, actualizamos el diccionario de costos acumulados y el diccionario de nodos
            # antecesores
            if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                cost_so_far[next_node] = new_cost
                priority = new_cost + heuristic(next_node)
                heapq.heappush(frontier, (priority, next_node))
                came_from[next_node] = current
# Devolvemos los diccionarios de nodos antecesores y costos acumulados
    return came_from, cost_so_far

"""
La función expand_fn recibe un nodo y devuelve una lista de tuplas que representan los estados y los costos para llegar a esos estados. 
Esta función es utilizada en la búsqueda Beam, para expandir los nodos.
Expande un estado, es decir, devuelve todos los hijos del estado actual con sus respectivos costos.

"""
#Función de beam
def expand_fn(state): #Javier Vázquez Gurrola
    return list(graph.get(state, {}).items())

# Función de búsqueda beam search
"""
La función beam_search implementa la búsqueda en haz (beam search) para encontrar el camino más corto desde 
el estado inicial hasta el estado objetivo. Los parámetros de la función son:
start_state: el estado inicial del problema.
goal_fn: una función que determina si un estado dado es el estado objetivo o no.
beam_width: el ancho del haz de búsqueda, es decir, la cantidad de estados que se mantienen en la frontera.
goal: el nodo objetivo del problema.
heuristic: una función heurística que estima el costo para llegar al estado objetivo desde cualquier estado dado.
La función comienza creando un haz de búsqueda que contiene el estado inicial con costo 0. 
Luego, se inicializa un diccionario llamado paths que almacenará los caminos hacia los estados de la frontera. 
El bucle principal se ejecuta mientras el haz de búsqueda tenga elementos. 
Dentro del bucle, se itera sobre los estados en el haz actual, y para cada estado, se expande su frontera utilizando la función expand_fn. 
Se calcula el costo y la heurística para cada estado hijo, y si se encuentra un estado objetivo, se devuelve el costo y el camino para llegar a él.
Si no se encuentra el estado objetivo, se agregan los hijos al siguiente haz de búsqueda y se actualiza el diccionario paths con los nuevos caminos. 
Finalmente, se actualiza el haz de búsqueda seleccionando los beam_width estados con menor costo y se continúa con la siguiente iteración. 
Si no quedan elementos en el haz de búsqueda, la función devuelve None.
La función también imprime el haz de búsqueda en cada iteración, lo que puede ser útil para entender cómo se va expandiendo durante la búsqueda.

"""

def beam_search_con_pasos(start_state, goal_fn, expand_fn, beam_width, goal, heuristic): #Joel Vázquez Anaya
    # Inicializamos el beam de búsqueda con el estado inicial y su costo acumulado.
    beam = [(0, start_state)]
    print("Inicializamos el beam de búsqueda con el estado inicial y su costo acumulado.")
    print(beam)
    print("")
    # Diccionario que contiene los caminos a cada estado.
    paths = {start_state: [start_state]}
    print("Diccionario que contiene los caminos a cada estado.")
    print(paths)
    print("")
    while True:
    # Creamos una lista que contendrá los estados del siguiente haz de búsqueda.
        next_beam = []
        print("Creamos una lista que contendrá los estados del siguiente haz de búsqueda.")
        print(next_beam)
        print("")
        # Iteramos sobre cada estado en el haz de búsqueda actual.
        print("Iteramos sobre cada estado en el haz de búsqueda actual.")
        for cost, state in beam:
        # Iteramos sobre cada estado hijo del nodo actual.
            for child_state, child_cost in expand_fn(state):
                print("Iteramos sobre cada estado hijo del nodo actual.")
            # Calculamos el nuevo costo acumulado para el estado hijo.
                new_cost = cost + child_cost
                print("Calculamos el nuevo costo acumulado para el estado hijo.")
                print(new_cost)
                print("")
                # Calculamos la heurística del nodo hijo.
                h = heuristic(child_state)
                print("Calculamos la heurística del nodo hijo.")
                print(h)
                print("")
                # Calculamos la función de costo f para el nodo hijo.
                f = new_cost + h
                print("Calculamos la función de costo f para el nodo hijo.")
                print(f)
                print("")
                # Si el estado hijo es el objetivo, devolvemos el costo y el camino que lleva hasta él.
                
                if goal_fn(child_state):
                    path = paths[state] + [child_state]
                    print("Si el estado hijo es el objetivo, devolvemos el costo y el camino que lleva hasta él.")
                    print(new_cost, path)
                    print("")
                    return (new_cost, path)
                # Si el estado hijo no es el objetivo, agregamos el estado hijo y su función de costo f a la lista de estados
                # del siguiente haz de búsqueda.
                next_beam.append((f, child_state))
                print("Si el estado hijo no es el objetivo, agregamos el estado hijo y su función de costo f a la lista de estados")
                print(next_beam)
                print("")
                # Actualizamos el diccionario de caminos, agregando el estado hijo al camino que lleva hasta él.
                paths[child_state] = paths[state] + [child_state]
                print("Actualizamos el diccionario de caminos, agregando el estado hijo al camino que lleva hasta él.")
                print(paths[child_state])
                print("")
                # Seleccionamos los estados con menor función de costo f para formar el siguiente haz de búsqueda.
        beam = heapq.nsmallest(beam_width, next_beam, key=lambda x: x[0])
        print("eleccionamos los estados con menor función de costo f para formar el siguiente haz de búsqueda.")
        print(beam)
        # Si no quedan estados en el siguiente haz de búsqueda, devolvemos None.
        if not beam:
            print("Si no quedan estados en el siguiente haz de búsqueda, devolvemos None.")
            print(None)
            print("")
            return None
        print("Imprime el beam de búsqueda en cada iteración")
        print("Beam:", beam) # Imprime el beam de búsqueda en cada iteración


def beam_search(start_state, goal_fn, expand_fn, beam_width, goal, heuristic): #Javier Vázquez Gurrola 
    # Inicializamos el beam de búsqueda con el estado inicial y su costo acumulado.
    beam = [(0, start_state)]
    # Diccionario que contiene los caminos a cada estado.
    paths = {start_state: [start_state]}
    while True:
    # Creamos una lista que contendrá los estados del siguiente haz de búsqueda.
        next_beam = []
        # Iteramos sobre cada estado en el haz de búsqueda actual.
        for cost, state in beam:
        # Iteramos sobre cada estado hijo del nodo actual.
            for child_state, child_cost in expand_fn(state):
            # Calculamos el nuevo costo acumulado para el estado hijo.
                new_cost = cost + child_cost
                # Calculamos la heurística del nodo hijo.
                h = heuristic(child_state)
                # Calculamos la función de costo f para el nodo hijo.
                f = new_cost + h
                # Si el estado hijo es el objetivo, devolvemos el costo y el camino que lleva hasta él.
                if goal_fn(child_state):
                    path = paths[state] + [child_state]
                    return (new_cost, path)
                # Si el estado hijo no es el objetivo, agregamos el estado hijo y su función de costo f a la lista de estados
                # del siguiente haz de búsqueda.
                next_beam.append((f, child_state))
                # Actualizamos el diccionario de caminos, agregando el estado hijo al camino que lleva hasta él.
                paths[child_state] = paths[state] + [child_state]
                # Seleccionamos los estados con menor función de costo f para formar el siguiente haz de búsqueda.
        beam = heapq.nsmallest(beam_width, next_beam, key=lambda x: x[0])
        # Si no quedan estados en el siguiente haz de búsqueda, devolvemos None.
        if not beam:
            return None
        print("Beam:", beam) # imprime el beam de búsqueda en cada iteración

"""
Parametros de la función:

graph: diccionario de diccionarios que representa el grafo
start: nodo de inicio
end: nodo de destino
heuristic_func: función heurística que estima el costo restante desde un nodo hasta el nodo de destino
El código implementa el algoritmo de búsqueda Branch and Bound para encontrar el camino más corto desde un nodo de inicio hasta un nodo de destino 
en un grafo. El algoritmo utiliza una función heurística para estimar el costo restante desde un nodo hasta el nodo de destino.
En la primera parte del código, se definen la cola de prioridad (heap) y el diccionario de costos mínimos. 
La cola de prioridad se utiliza para ordenar los nodos adyacentes según su costo total más una estimación heurística del costo restante 
hasta el nodo destino. El diccionario de costos mínimos se utiliza para llevar un registro de los costos mínimos 
conocidos para llegar a cada nodo del grafo.
La cola de prioridad se inicializa con una tupla que contiene el costo total estimado desde el nodo de inicio hasta el nodo de destino, 
el nodo de inicio, el camino recorrido hasta el momento (que inicialmente solo contiene el nodo de inicio) 
y el costo total acumulado hasta el momento (que inicialmente es cero). 
El costo total estimado se calcula sumando el costo actual y la estimación heurística del costo restante hasta el nodo de destino.
El diccionario de costos mínimos se inicializa con un valor infinito para cada nodo del grafo, 
excepto para el nodo de inicio, que se inicializa con cero más la estimación heurística del costo restante hasta el nodo de destino.
Luego, se inicia un bucle que recorre la cola de prioridad hasta encontrar el camino más corto desde el nodo de inicio hasta el nodo de destino. 
En cada iteración del bucle, se extrae el nodo con el costo total más bajo de la cola de prioridad y se exploran sus nodos adyacentes.
Si el nodo extraído es el nodo de destino, se devuelve el camino recorrido hasta el momento y el costo total acumulado.
Si el costo actual es mayor que el costo mínimo conocido para el nodo, se ignora el nodo actual y se continúa con el siguiente.
Si el costo actual es menor que el costo mínimo conocido para el nodo adyacente, se actualiza el costo mínimo conocido y se agrega el nodo adyacente 
a la cola de prioridad con el nuevo costo total estimado, el nuevo camino y el nuevo costo total acumulado.
Si no se encuentra un camino desde el nodo de inicio hasta el nodo de destino, se devuelve None.
En resumen, este código implementa el algoritmo de búsqueda Branch and Bound para encontrar el camino más corto desde 
un nodo de inicio hasta un nodo de destino en un grafo, utilizando una función heurística para estimar el costo restante hasta el nodo de destino.
"""

def branch_and_bound_shortest_path_con_pasos(graph, start, end, heuristic_func): #Joel Vázquez Anaya
    # Definir la cola de prioridad (heap) y el diccionario de costos mínimo
    pq = []
    heapq.heappush(pq, (0 + heuristic_func(start), start, [start], 0))
    min_costs = {node: float('inf') for node in graph}
    min_costs[start] = 0 + heuristic_func(start)
    print("Definir la cola de prioridad (heap) y el diccionario de costos mínimos")
    print(pq)
    print(min_costs)
    print(min_costs[start])
    print("")
    # Recorrer la cola de prioridad hasta encontrar el camino más corto desde el nodo "start" hasta el nodo "end"
    while pq:
        cost, node, path, total_cost = heapq.heappop(pq)
        print("Recorrer la cola de prioridad hasta encontrar el camino más corto desde el nodo start hasta el nodo end")
        print("Costo: ",cost, "Nodo: ", node, "Lista: ",path, "Costo Total: ", total_cost)
        print("")
        # Si se ha encontrado el nodo de destino, devolver el camino y el costo total
        if node == end:
            print("Si se ha encontrado el nodo de destino, devolver el camino y el costo total")
            print(path, total_cost)
            return path, total_cost
        # Si el costo actual es mayor que el costo mínimo conocido, ignorar el nodo actual
        if cost > min_costs[node]:
            continue
        # Explorar los nodos adyacentes al nodo actual
        for adj_node, adj_cost in graph[node].items():
            new_cost = cost - heuristic_func(node) + adj_cost + heuristic_func(adj_node)
            new_path = path + [adj_node]
            new_total_cost = total_cost + adj_cost
            # Si se ha encontrado un camino más corto al nodo adyacente, actualizar el costo mínimo y agregar el nodo a la cola de prioridad
            if new_cost < min_costs[adj_node]:
                min_costs[adj_node] = new_cost
                heapq.heappush(pq, (new_cost, adj_node, new_path, new_total_cost))
    # Si no se encuentra un camino desde el nodo "start" hasta el nodo "end", devolver None
    return None


def branch_and_bound_shortest_path(graph, start, end, heuristic_func): #Javier Vázquez Gurrola 
    # Definir la cola de prioridad (heap) y el diccionario de costos mínimos
    pq = []
    heapq.heappush(pq, (0 + heuristic_func(start), start, [start], 0))
    min_costs = {node: float('inf') for node in graph}
    min_costs[start] = 0 + heuristic_func(start)
    # Recorrer la cola de prioridad hasta encontrar el camino más corto desde el nodo "start" hasta el nodo "end"
    while pq:
        cost, node, path, total_cost = heapq.heappop(pq)
        # Si se ha encontrado el nodo de destino, devolver el camino y el costo total
        if node == end:
            return path, total_cost
        # Si el costo actual es mayor que el costo mínimo conocido, ignorar el nodo actual
        if cost > min_costs[node]:
            continue
        # Explorar los nodos adyacentes al nodo actual
        for adj_node, adj_cost in graph[node].items():
            new_cost = cost - heuristic_func(node) + adj_cost + heuristic_func(adj_node)
            new_path = path + [adj_node]
            new_total_cost = total_cost + adj_cost
            # Si se ha encontrado un camino más corto al nodo adyacente, actualizar el costo mínimo y agregar el nodo a la cola de prioridad
            if new_cost < min_costs[adj_node]:
                min_costs[adj_node] = new_cost
                heapq.heappush(pq, (new_cost, adj_node, new_path, new_total_cost))
    # Si no se encuentra un camino desde el nodo "start" hasta el nodo "end", devolver None
    return None

#Función Steepest Hill Climbing con pasos
def steepest_hill_climbing_con_pasos(graph, initial_node):# Joel Vázquez Anaya
    #Guardamos el nodo inicial en una variable denomina nodo actual
    current_node = initial_node
    print("Guardamos el nodo inicial en una variable denomina nodo actual")
    print(current_node ,"=", initial_node)
    print("")
    #Creamos un ciclo para que se calcule la heuristica con cada uno de los nodos recurrentes
    print("Creamos un ciclo para que se calcule la heuristica con cada uno de los nodos recurrentes")
    while True:
        #Calculamos la heuristica para el nodo actual  y la comparas con los vecinos
        current_score = heuristic(current_node)
        best_score = current_score
        best_node = current_node
        print("Calculamos la heuristica para el nodo actual  y la comparas con los vecinos")
        print("Verificamos cual es el mejor nodo, viendo cual tiene el mejor valor")
        print(best_score)
        print("Guardamos el nodo con el mejor valor")
        print(best_node)
        print("")
        
        #Calcula la heuristica para cada uno de los vecinos del nodo acual y toma el que tenga 
        #una heuristica mejor para poderlo tomar como el mejor nodo
        print("Calcula la heuristica para cada uno de los vecinos del nodo acual")
        for neighbor in graph[current_node]:
            neighbor_score = heuristic(neighbor)
            print("Sacamos el valor de la heuristica del vecino")
            print(graph[current_node])
            print("Vecino: ", neighbor, "Heuristica: ", neighbor_score)
            if neighbor_score > best_score:
                print("Si el valor es menor o igual que el valor del nodo actual regresa el nodo actual ")
                best_score = neighbor_score
                best_node = neighbor
                print("Mejor valor: ", best_score)
                print("Mejor nodo: ", best_node)
        # Si es mejor valor es menor o igual que el valor del nodo actual regresa el nodo actual 
        # porque esa es la mejor opción
        if best_score <= current_score:
            print("Sino regresa el nodo actual como el mejor valor")
            print(current_node)
            return current_node
        #Sino regresa el nodo actual como el mejor valor
        current_node = best_node


# Función de Steepest Hil Climbing sin pasos
def steepest_hill_climbing(graph, initial_node):# Joel Vázquez Anaya
    #Verficamos que el nodo actual sea el nodo inicial y si es así se termina la función
    current_node = initial_node
    #Creamos un ciclo para que se calcule la heuristica con cada uno de los nodos recurrentes
    while True:
        #Calculamos la heuristica para el nodo actual para ver cual es su heuristica y poder compararlo con los vecinos
        current_score = heuristic(current_node)
        best_score = current_score
        best_node = current_node
        #Calcula la heuristica para cada uno de los vecinos del nodo acual y toma el que tenga 
        #una heuristica mejor para poderlo tomar como el mejor nodo
        for neighbor in graph[current_node]:
            neighbor_score = heuristic(neighbor)
            if neighbor_score > best_score:
                best_score = neighbor_score
                best_node = neighbor
        # Si es mejor valor es menor o igual que el valor del nodo actual regresa el nodo actual 
        # porque esa es la mejor opción
        if best_score <= current_score:
            return current_node
        #Sino regresa el nodo actual como el mejor valor
        current_node = best_node
        
#Función Stochastic Hil Clambing con pasos
def stochastic_hill_climbing_con_pasos(graph, initial_node, heuristic):# Joel Vázquez Anaya
    #Verificamos si el nodo actual es el nodo de inicio si es asi el programa termina pero
    #si no es de este modo se hace una virifcación entre cada uno de los nodos vecinos de forma
    #aleatoria para poder ver el mejor camino con la mayor heuristica
    current_node = initial_node
    print("Guardamos el nodo inicial en una variable llamada current_node")
    print(current_node, "=", initial_node)
    print("")
    while True:
        #Se hace la heuristica al nodo actual para ver su mejor forma
        current_score = heuristic(current_node)
        print("Se hace la heuristica al nodo actual para ver su mejor forma")
        print(current_score)
        print("")
        #Se guarda en una lista los nodos la recorridos por el grafo
        neighbors = graph[current_node]
        print("Se guarda en una lista los nodos la recorridos por el grafo")
        print(neighbors)
        # random.choices() debe recibir una lista y un valor de peso opcional.
        # En este caso, no se necesita un valor de peso, por lo que simplemente se
        # convierte el conjunto de vecinos en una lista antes de pasarla a
        # random.choices().
        random_neighbor = random.choices(list(neighbors))[0]
        neighbor_score = heuristic(random_neighbor)
        print("Utilizamos random.choices() que debe recibir una lista y un valor de peso opcional.")
        print("simplemente se convierte el conjunto de vecinos en una lista antes de pasarla a random.choices().")
        print("Vecino elegido aleatoriamente: ",random_neighbor)
        print("Valor del vecino: ", neighbor_score)
        
        if neighbor_score > current_score:
            print("Si es valor del nodo actual es menor a la del vecino se convierte en el nodo actual")
            print(current_node)
            current_node = random_neighbor
        else:
            ("Regresa el mejor nodo")
            print(current_node)
            return current_node


#Función Stochastic Hil Clambing sin pasos
def stochastic_hill_climbing(graph, initial_node, heuristic):# Joel Vázquez Anaya
    #Verificamos si el nodo actual es el nodo de inicio si es asi el programa termina pero
    #si no es de este modo se hace una virifcación entre cada uno de los nodos vecinos de forma
    #aleatoria para poder ver el mejor camino con la mayor heuristica
    current_node = initial_node
    while True:
        #Se hace la heuristica al nodo actual para ver su mejor forma
        current_score = heuristic(current_node)
        #Se guarda en una lista los nodos la recorridos por el grafo
        neighbors = graph[current_node]
        # random.choices() debe recibir una lista y un valor de peso opcional.
        # En este caso, no se necesita un valor de peso, por lo que simplemente se
        # convierte el conjunto de vecinos en una lista antes de pasarla a
        # random.choices().
        random_neighbor = random.choices(list(neighbors))[0]
        neighbor_score = heuristic(random_neighbor)
        if neighbor_score > current_score:
            current_node = random_neighbor
        else:
            return current_node

#Simulated annealing


def genetic_algorithm_con_pasos(graph, population_size, num_generations, mutation_rate):# Joel Vázquez Anaya
    population = generate_initial_population(population_size, graph)
    for generation in range(num_generations):
        fitness_scores = [fitness_function(chromosome, graph) for chromosome in population]
        parent1 , parent2 = select_parents(population)
        offspring = generate_offspring(parent1 , parent2)
        population = mutate_population(offspring, mutation_rate,graph)
    best_chromosome = max(population, key=lambda chromosome: fitness_function(chromosome, graph))
    return best_chromosome

#Función Genetic Algorithm con pasos
def genetic_algorithm_con_pasos(graph, population_size, num_generations, mutation_rate):# Joel Vázquez Anaya
    population = generate_initial_population_con_pasos(population_size, graph)
    for generation in range(num_generations):
        fitness_scores = [fitness_function_con_pasos(chromosome, graph) for chromosome in population]
        parent1 , parent2 = select_parents_con_pasos(population)
        offspring = generate_offspring_con_pasos(parent1 , parent2)
        population = mutate_population_con_pasos(offspring, mutation_rate,graph)
    best_chromosome = max(population, key=lambda chromosome: fitness_function_con_pasos(chromosome, graph))
    return best_chromosome

#Función Genetic Algorithm sin pasos
def genetic_algorithm(graph, population_size, num_generations, mutation_rate):# Joel Vázquez Anaya
    population = generate_initial_population(population_size, graph)
    for generation in range(num_generations):
        fitness_scores = [fitness_function(chromosome, graph) for chromosome in population]
        parent1 , parent2 = select_parents(population)
        offspring = generate_offspring(parent1 , parent2)
        population = mutate_population(offspring, mutation_rate,graph)
    best_chromosome = max(population, key=lambda chromosome: fitness_function(chromosome, graph))
    return best_chromosome


"""
Función generate_initial_population para realizar el algoritmo de Genetic Algorithm
Lo que hace esta función es generar una población de manera aleatoria, esta población es 
cada uno de los nodos del grafo, los cuales representan una posible solución del problema,
esta función toma cada nodo del grafo y los coloca de modo de un directorio para poder ver la 
mejor solución al problema
 
"""
def generate_initial_population_con_pasos(population_size, graph): #Joel Vázquez Anaya
    population = []
    nodes = list(graph.keys())
    for i in range(population_size):
        print("Generar una población de manera aleatoria, esta población es cada uno de los nodos del grafo, los cuales representan una posible solución del problema")
        chromosome = random.sample(nodes, len(nodes))
        population.append(chromosome)
        print(chromosome)
        print(population)
        print("")
    return population

def generate_initial_population(population_size, graph): #Joel Vázquez Anaya
    population = []
    nodes = list(graph.keys())
    for i in range(population_size):
        chromosome = random.sample(nodes, len(nodes))
        population.append(chromosome)
    return population

"""
Función de fitness_function para realizar el algoritmo de Genetic Algorithm
Esta función hace la implemnetación de ver que tanta aplitud tiene cada nodo
esto quiere decir que ve cuantos vecinos tiene cada uno de los nodos que se ecnuentran
en el cromosoma, se busca maximizar los vecinos que son soluciones en el cromosoma

"""
def fitness_function_con_pasos(chromosome, graph): #Joel Vázquez Anaya
    fitness = 0
    print("Vemos la cantidad de vecinos que tiene el nodo")
    for node in graph:
        for neighbor in graph[node]:
            if neighbor in chromosome:
                fitness += 1
                print("Nodo: ", chromosome)
                print("Vecino: ", neighbor)
                print("Tamaño: ", fitness)
                print("")
    return fitness

def fitness_function(chromosome, graph): #Joel Vázquez Anaya
    fitness = 0
    for node in graph:
        for neighbor in graph[node]:
            if neighbor in chromosome:
                fitness += 1
    return fitness


"""

Función select_parents para realizar el algoritmo de Genetic Algorithm
La función genera un cruce entre dos padres la cual lo hace de la siguiente manera,
toma un candidato y lo que hace es que selecciona un nodo de manera aleatoria para poder 
encontrar a un candidato con el mayor fitness, ya cuando tenga el candidato con mayor fitness
ese lo toma como padre y este procedimiento lo hace 2 veces para tener los 2 padres y lo 
agrega a la lista de mejores andidatos la cual se guarda en padres

"""
def select_parents_con_pasos(population): #Joel Vázquez Anaya
    print("Realizamos la cruza de dos padres de manera aleatoria")
    print("Se hace de ña siguiente manera: ")
    print("Toma un candidato y lo que hace es que selecciona un nodo de manera aleatoria para poder")
    print("encontrar a un candidato con el mayor fitness, ya cuando tenga el candidato con mayor fitness")
    print("ese lo toma como padre y este procedimiento lo hace 2 veces para tener los 2 padres y lo")
    print("agrega a la lista de mejores andidatos la cual se guarda en padres")
    print("")
    parent1 = random.choice(population)
    parent2 = random.choice(population)
    print("Padre 1: ", parent1)
    print("Padre 2: ", parent2)
    print("")
    while parent2 == parent1 and len(population) > 1:
        print("Si el Padre 1 es igual al padre 2 y todavía tenemos más nodos en el arreglo seleccionamos otro Padre de manera aleatoria")
        parent2 = random.choice(population)
    print("Padre 2: ", parent2)
    print("")
    print("Regresamos los padres seleccionados")
    print("Padre 1: ", parent1)
    print("Padre 2: ", parent2)
    print("")
    return parent1, parent2


def select_parents(population): #Joel Vázquez Anaya
    parent1 = random.choice(population)
    parent2 = random.choice(population)
    while parent2 == parent1 and len(population) > 1:
        parent2 = random.choice(population)
    return parent1, parent2

"""

Función generate_offspring para realizar el algoritmo de Genetic Algorithm
Esta función lo que hace es que a partir de los 2 padres crea una desendencia 
lo hace a partir de una de las partes de cada uno de los padres para poder 
realizar esta desecendia y esto lo hace de la siguiente manera si un número 
aleatorio generado al azar es mayor que la tasa de cruce, se clona uno de los 
padres como el descendiente sin realizar cruce. Si el número aleatorio es menor 
o igual a la tasa de cruce, se selecciona un punto de cruce aleatorio entre 1 y 
la longitud del cromosoma menos 1. Luego, se combina la primera parte del primer 
padre con la segunda parte del segundo padre a partir del punto de cruce para 
formar el descendiente.

"""
def generate_offspring_con_pasos(parent1, parent2): #Joel Vázquez Anaya
    print("Hacemos a partir de los 2 padres crea una desendencia lo hace a partir de una de las partes de cada uno de los padres")
    if len(parent1) <= 1:  # verificación de longitud de parent1
        return parent1
    crossover_point = random.randrange(1, len(parent1))
    child = parent1[:crossover_point] + parent2[crossover_point:]
    print("Tomando al tasa de cruce se clona una de las desendencias")
    print(crossover_point)
    print("Hijo: ", child)
    print("")
    """if random.random() < mutation_rate:
        print("Llamamos a la función de mutación de la población")
        mutate_population(child)"""
    return 

def generate_offspring(parent1, parent2): #Joel Vázquez Anaya
    if len(parent1) <= 1:  # verificación de longitud de parent1
        return parent1
    crossover_point = random.randrange(1, len(parent1))
    child = parent1[:crossover_point] + parent2[crossover_point:]

    return child


"""

Función mutate_population para realizar el algoritmo de Genetic Algorithm
Esta función lo que hace es generar una mutación en los cromosomas, esto lo hace tomando
de forma aleatoria un nodo del cromosoma para poder realizar la mutación, esto se hace 
con las siguientes reglas, ya que la función itera a través de cada cromosoma en la población
y verifica si se debe aplicar una mutación. Si un número aleatorio generado al azar es menor 
o igual a la tasa de mutación (mutation_rate), se realiza la mutación. Se crea una copia del 
cromosoma original y se cambia un nodo aleatorio en el cromosoma por otro nodo elegido al azar 
del grafo, el cromosoma mutado se agrega a una lista de cromosomas mutados, que se devuelve al 
final de la función.
 
"""
def mutate_population_con_pasos(population, mutation_rate, graph): #Joel Vázquez Anaya
    print("Generar una mutación en los cromosomas, esto lo hace tomando de forma aleatoria un nodo del cromosoma para poder realizar la mutación")
    print("Esto se hace con las siguientes reglas: ")
    print("Si un número aleatorio generado al azar es menor o igual a la tasa de mutación (mutation_rate), se realiza la mutación")
    print("Se crea una copia del cromosoma original y se cambia un nodo aleatorio en el cromosoma por otro nodo elegido al azar del grafo")
    print(" el cromosoma mutado se agrega a una lista de cromosomas mutados, que se devuelve al final de la función")
    print("")
    mutated_population = []
    for chromosome in population:
        mutated_chromosome = list(chromosome)
        for i in range(len(chromosome)):
            if random.random() < mutation_rate:
                print("Verificamos la tasa de mutación en un cromosoma")
                mutated_chromosome[i] = random.choice(list(graph.keys()))
                print(mutated_chromosome[i])
                print("")
        mutated_population.append(mutated_chromosome)
    print("Lista de cromosomas mutados: ", mutated_population)
    
    return mutated_population

def mutate_population(population, mutation_rate, graph): #Joel Vázquez Anaya
    mutated_population = []
    for chromosome in population:
        mutated_chromosome = list(chromosome)
        for i in range(len(chromosome)):
            if random.random() < mutation_rate:
                mutated_chromosome[i] = random.choice(list(graph.keys()))
                print(mutated_chromosome[i])
        mutated_population.append(mutated_chromosome)
    
    return mutated_population

def simulated_annealing(initial_state, destination, adjacency_list, max_iterations, temperature): #Francisco Anaya Viveros
    current_state = initial_state
    current_cost = heuristic(current_state)
    best_state = current_state
    best_cost = current_cost
    
    for i in range(max_iterations):
    # Generar un vecino aleatorio
      neighbors = get_neighbors(current_state, adjacency_list)
      next_state = random.choice(neighbors)
      next_cost = heuristic(next_state)
    
    # Evaluar si se acepta el vecino
      delta_cost = next_cost - current_cost
      if delta_cost < 0:
          current_state = next_state
          current_cost = next_cost
      else:
          prob = math.exp(-delta_cost / temperature)
          if random.random() < prob:
              current_state = next_state
              current_cost = next_cost
    
    # Actualizar la mejor solución encontrada
      if current_cost < best_cost:
          best_state = current_state
          best_cost = current_cost
    
    # Actualizar la temperatura
      temperature *= 0.9999
        
      return best_state, best_cost


def get_neighbors(node, adjacency_list): #Francisco Anaya Viveros
    return adjacency_list[node]


def dibujar_mapa(): #Francisco Anaya Viveros
  latitudes = {'Cancún': 21.1213285, 'Valladolid': 20.688114, 'Felipe Carrillo Puerto': 19.5778903, 'Campeche': 19.8305682, 'Merida': 20.9800512, 
                 'Ciudad del Carmen': 18.6118375, 'Chetumal': 18.5221567, 'Villa Hermosa': 17.9925264, 'Tuxtla': 16.7459857, 'Francisco Escarcega': 18.6061556, 
                 'Acayucan': 17.951096, 'Tehuantepec': 16.320636, 'Alvarado': 18.7760455, 'Oaxaca': 17.0812951, 'Puerto Angel': 15.6679974, 
                 'Izucar de Matamoros': 18.5980563, 'Tehuacan': 18.462191, 'Pinotepa Nacional': 16.3442895, 'Cuernavaca': 18.9318685, 
                 'Puebla': 19.040034, 'Acapulco': 16.8354485, 'Cdmx': 19.3898319, 'Iguala': 18.3444, 'Ciudad Altamirano': 18.3547491,
                 'Cordoba': 18.8901707, 'Chilpancingo': 17.5477072, 'Tlaxcala': 19.4167798, 'Pachuca de Soto': 20.0825056, 'Queretaro': 20.6121228,
                 'Toluca de Lerdo': 19.294109, 'Zihuatanejo': 17.6405745, 'Veracruz': 19.1787635, 'Tuxpan de Rodriguez Cano': 20.9596561, 
                 'Atlacomulco': 19.7980152, 'Salamanca': 20.5664927, 'San Luis Potosi': 22.1127046, 'Playa Azul': 17.9842581, 'Tampico': 22.2662251, 
                 'Guanajuato': 21.0250928, 'Morelia': 19.7036417, 'Guadalajara': 20.6737777, 'Aguascalientes': 21.8857199, 'Zacatecas': 22.7636293,
                 'Durango': 24.0226824, 'Colima': 19.2400444, 'Manzanillo': 19.0775491, 'Ciudad Victoria': 23.7409928, 'Tepic': 21.5009822, 
                 'Hidalgo del Parral': 26.9489283, 'Mazatlan': 23.2467283, 'Soto la Marina': 23.7673729, 'Matamoros': 25.8433787, 'Monterrey': 25.6487281, 
                 'Chihuahua': 28.6708592, 'Topolobampo': 25.6012747, 'Culiacan': 24.8049008, 'Reynosa': 26.0312262, 'Monclova': 26.907775, 
                 'Ciudad Juárez': 31.6538179, 'Janos': 30.8898127, 'Ciudad Obregon': 27.4827355, 'Torreon': 25.548597, 'Ojinaga': 29.5453292, 
                 'Nuevo Laredo': 27.4530856, 'Agua Prieta': 31.3115272, 'Guaymas': 27.9272572, 'Piedras Negras': 28.6910517, 'Santa Ana': 30.5345457, 
                 'Hermosillo': 29.082137, 'Mexicali': 32.6137391, 'Tijuana': 32.4966818, 'San Felipe': 31.009535, 'Ensenada': 31.8423096,
                 'San Quintin': 30.5711324, 'Santa Rosalia': 27.3408761, 'Santo Domingo': 25.3487297, 'La Paz': 24.1164209, 'Cabo San Lucas': 22.8962253 }
    
  longitudes = {'Cancún': -86.9192738, 'Valladolid': -88.2204456, 'Felipe Carrillo Puerto': -88.0630853, 'Campeche': -90.5798365, 'Merida': -89.7029587, 
                'Ciudad del Carmen': -91.8927345, 'Chetumal': -88.3397982, 'Villa Hermosa': -92.9881407, 'Tuxtla': -93.1996103, 'Francisco Escarcega': -90.8176486, 
                'Acayucan': -94.9306961, 'Tehuantepec': -95.27521, 'Alvarado': -95.7731952, 'Oaxaca': -96.7707511, 'Puerto Angel': -96.4933733, 
                'Izucar de Matamoros': -98.5076767, 'Tehuacan': -97.4437333, 'Pinotepa Nacional': -98.1315923, 'Cuernavaca': -99.3106054, 'Puebla': -98.2630056,
                'Acapulco': -99.9323491, 'Cdmx': -99.7180148, 'Iguala': -99.5652232, 'Ciudad Altamirano': -100.6817619, 'Cordoba': -96.9751108, 
                'Chilpancingo': -99.5324349, 'Tlaxcala': -98.4471127, 'Pachuca de Soto': -98.8268184, 'Queretaro': -100.4802576, 'Toluca de Lerdo': -99.6662331, 
                'Zihuatanejo': -101.5601369, 'Veracruz': -96.2113357, 'Tuxpan de Rodriguez Cano': -97.4158767, 'Atlacomulco': -99.89317, 'Salamanca': -101.2176511, 
                'San Luis Potosi': -101.0261099, 'Playa Azul': -102.357616, 'Tampico': -97.939526, 'Guanajuato': -101.3296402, 'Morelia': -101.2761644, 
                'Guadalajara': -103.4054536, 'Aguascalientes': -102.36134, 'Zacatecas': -102.623638, 'Durango': -104.7177652, 'Colima': -103.7636273, 'Manzanillo': -104.4789574, 
                'Ciudad Victoria': -99.1783576, 'Tepic': -104.9119242, 'Hidalgo del Parral': -105.8211168, 'Mazatlan': -106.4923175, 'Soto la Marina': -98.2157573, 
                'Matamoros': -97.5849847, 'Monterrey': -100.4431819, 'Chihuahua': -106.2047036, 'Topolobampo': -109.0687891, 'Culiacan': -107.4933545, 'Reynosa': -98.3662435, 
                'Monclova': -101.4940069, 'Ciudad Juárez': -106.5890206, 'Janos': -108.208458, 'Ciudad Obregon': -110.0844111, 'Torreon': -103.4719562, 'Ojinaga': -104.4305246, 
                'Nuevo Laredo': -99.6881218, 'Agua Prieta': -109.5855873, 'Guaymas': -110.9779564, 'Piedras Negras': -100.5801829, 'Santa Ana': -111.1580567, 
                'Hermosillo': -111.059027, 'Mexicali': -115.5203312, 'Tijuana': -117.087892, 'San Felipe': -114.8727296, 'Ensenada': -116.6799816, 'San Quintin': -115.9588544, 
                'Santa Rosalia': -112.2825762, 'Santo Domingo': -111.9975909, 'La Paz': -110.3727673, 'Cabo San Lucas': -109.9505077}
    
  fig, ax = plt.subplots(1, 2, figsize=(35,15)) # especificar el tamaño de la figura
    # Dibujar grafo antes de la búsqueda
  pos = {k: (longitudes[k], latitudes[k]) for k in latitudes.keys()}
  nx.draw(G, pos, with_labels=True, ax=ax[0]) # dibujar en el primer subplot
  ax[0].set_title("Grafo antes de la búsqueda", fontsize=25) # establecer el título del primer subplot

    # Realizar búsqueda
  best_state, best_cost = simulated_annealing(start_verificado, goal_verificado, adjacency_list, max_iterations, temperature)

    # Dibujar grafo después de la búsqueda
  nx.draw(G, pos, with_labels=True, ax=ax[1]) # dibujar en el segundo subplot
  nx.draw_networkx_nodes(G, pos, nodelist=[best_state], node_color='r', node_size=500)
  ax[1].set_title("Grafo después de la búsqueda", fontsize=25) # establecer el título del segundo subplot

    # Mostrar figura
  plt.subplots_adjust(wspace=0.4) # ajustar espacio entre figuras
  plt.show()


def verificacion(nodo): #Javier Vázquez Gurrola 
  bandera = False
  todos_los_nodos = ['Cabo San Lucas','La Paz','Santo Domingo','Santa Rosalia','Santa Quintin','Ensenada','San Felipe','Tijuana','Mexicalli','Santa Ana','Agua Prieta','Hermosillo','Janos','Guaymas','Ciudad Obregon','Chihuahua','Juarez','Topolobampo','Hidalgo del Parral','Culiacan','Mazatlan','Tepic','Ojinaga','Monclava','Torreon','Piedras Negras','Nuevo Laredo','Reynosa','Matamoros','Ciudad Victoria','Soto la Marina','Monterrey','Durango','Zacatecas','San Luis Potosi','Aguascalientes','Guanajuato','Guadalajara','Manzanillo','Colima','Salamanca','Atlacomulco','Queretaro','Tlaxcala','Tampico','Tuxpan de Rodiguez Cano','Pachuca de Soto','Playa Azul','Zihuatanejo','Ciudad Altamirano','Iguala','Chilpancingo','Acapulco','Pinotepa Nacional','Puerto Angel','Oaxaca','Cdmx','Cuernavaca','Puebla','Vercruz','Cordoba','Izucar de Matamoros','Tehuacan','Oaxaca','Alvarado','Acayucan','Tehuantepec','Tuxtla','Villa Hermosa','Ciudad del Carmen','Campeche','Merida','Francisco Escarcega','Chetumal','Felipe Carrillo Puerto','Valladolid','Cancún']
  #todos_los_nodos = ['A', 'B', 'C', 'D', 'E', 'F']
  #for elemento en todos_los_nodos:
  if nodo in todos_los_nodos: 
        print("El elemento", nodo, "está en la lista, puede continuar...\n") 
        bandera = True
        return bandera
  else: 
        print("El elemento", nodo, "no está en la lista, intente de nuevo...\n")
        return bandera

#Joel Vázquez Anaya y Javier Vázquez Gurrola 
graph = {
            'Cancún': {'Valladolid': 90, 'Felipe Carrillo Puerto' : 100 },
            'Valladolid' : {'Felipe Carrillo Puerto': 90 },
            'Felipe Carrillo Puerto': {'Campeche' : 60 },
            'Campeche': { 'Merida': 90, 'Chetumal': 100, 'Ciudad del Carmen' : 90 },
            'Merida': {},
            'Chetumal': {'Francisco Escarcega' : 110 },
            'Francisco Escarcega' : {},
            'Ciudad del Carmen': {'Villa Hermosa': 90, 'Tuxtla' : 90 },
            'Villa Hermosa': { 'Acayucan': 90 }, 
            'Tuxtla': { 'Acayucan' : 90 },
            'Acayucan': {'Tehuantepec': 80, 'Alvarado' : 110 },
            'Tehuantepec' : {},
            'Alvarado'  : { 'Oaxaca' : 100 },
            'Oaxaca' : { 'Puerto Angel': 90 , 'Tehuacan': 80, 'Izucar de Matamoros': 90 },
            'Tehuacan' : {},
            'Puerto Angel' : { 'Pinotepa Nacional' : 100 },
            'Pinotepa Nacional' : {'Acapulco': 100 },
            'Acapulco' : { 'Chilpancingo' : 140 },
            'Chilpancingo' : { 'Iguala': 90 },
            'Iguala' : { 'Cuernavaca' : 100, 'Ciudad Altamirano': 110 },
            'Izucar de Matamoros' : {'Puebla': 90,'Cuernavaca':100 },
            'Puebla' : {'Cordoba':80,'Cdmx' :90 },
            'Cordoba' : {'Veracruz':90 },
            'Veracruz' : {},
            'Cuernavaca' : {'Cdmx':100, 'Ciudad Altamirano': 100},
            'Cdmx' : {'Pachuca de Soto':100,'Queretaro':90,'Toluca de Lerdo':110,'Tlaxcala':100},
            'Tlaxcala' : {},
            'Toluca de Lerdo' : {'Ciudad Altamirano':100 },
            'Ciudad Altamirano' : {'Zihuatanejo':90},
            'Zihuatanejo' : {'Playa Azul':90},
            'Playa Azul' : {'Morelia':100,'Colima':100,'Manzanillo':100},
            'Morelia' : {'Colima':90, 'Salamanca':90},
            'Colima' : {'Manzanillo':50, 'Guadalajara':50},
            'Manzanillo' : {'Guadalajara':80},
            'Guadalajara' : {'Tepic':110,'Aguascalientes':70},
            'Salamanca' : {'Guanajuato':90,'Guadalajara':90},
            'Guanajuato' : { 'Aguascalientes' : 80 },
            'Aguascalientes' : {'San Luis Potosi': 100 },
            'Queretaro' : {'Salamanca':90, 'Atlacomulco': 90, 'San Luis Potosi' :90 },
            'Atlacomulco' : {},
            'Pachuca de Soto' : {'Tuxpan de Rodriguez Cano':110},
            'Tuxpan de Rodriguez Cano' : {'Tampico':80},
            'Tampico' : {'Ciudad Victoria':80},
            'Ciudad Victoria' : {'Soto la Marina':80,'Matamoros':80,'Monterrey':80,'Durango':80 },
            'Soto la Marina' : {},
            'San Luis Potosi' : {'Zacatecas': 90, 'Durango':70},
            'Zacatecas' :  {},
            'Tepic' : {'Mazatlan': 110},
            'Mazatlan' : {'Culiacan':90 },
            'Durango' : {'Mazatlan': 90, 'Hidalgo del Parral': 90, 'Torreon': 110 },
            'Torreon' : {'Monclova': 110 },
            'Matamoros' : { 'Reynosa': 90 },
            'Reynosa' : {'Nuevo Laredo' : 100 },
            'Monterrey' : {'Nuevo Laredo': 110 , 'Monclova' :70 },
            'Nuevo Laredo' : { 'Piedras Negras' : 100 },
            'Piedras Negras' : { 'Monclova': 100 },
            'Monclova' : {'Ojinaga': 140 },
            'Culiacan' : {'Topolobampo':110, 'Hidalgo del Parral': 80 },
            'Ojinaga' : {'Chihuahua':90 },
            'Hidalgo del Parral' :  {'Chihuahua' :130 },
            'Chihuahua' : {'Ciudad Juárez':90,'Janos':90 },
            'Ciudad Juárez' : {},
            'Topolobampo' : {'Ciudad Obregon': 90, 'Hidalgo del Parral': 110 },
            'Ciudad Obregon' : {'Guaymas':80 },
            'Guaymas' : {'Hermosillo':80 },
            'Hermosillo' : {'Santa Ana': 60 },
            'Janos' : {'Agua Prieta' : 110 },
            'Agua Prieta' : {'Santa Ana': 60 },
            'Santa Ana' : {'Mexicali': 150 },
            'Mexicali' : {'San Felipe': 70, 'Tijuana' : 110},
            'Tijuana' : {'Ensenada' : 50 },
            'San Felipe' : {'Ensenada' :50 },
            'Ensenada' : {'San Quintin':60 },
            'San Quintin' : {'Santa Rosalia': 60},
            'Santa Rosalia' : {'Santo Domingo' :60},
            'Santo Domingo' : {'La Paz':70 },
            'La Paz' : {'Cabo San Lucas':70 },
            'Cabo San Lucas' : {}
        }

# Grafo para simulated annealing
adjacency_list = { #Francisco Anaya Viveros
            'Cancún': ['Valladolid', 'Felipe Carrillo Puerto'],
            'Valladolid' : ['Felipe Carrillo Puerto'],
            'Felipe Carrillo Puerto': ['Campeche'],
            'Campeche': ['Merida', 'Chetumal', 'Ciudad del Carmen'],
            'Merida': [],
            'Chetumal': ['Francisco Escarcega'],
            'Francisco Escarcega' : [],
            'Ciudad del Carmen': ['Villa Hermosa', 'Tuxtla'],
            'Villa Hermosa': ['Acayucan'], 
            'Tuxtla': ['Acayucan'],
            'Acayucan': ['Tehuantepec', 'Alvarado'],
            'Tehuantepec' : [],
            'Alvarado'  : ['Oaxaca'],
            'Oaxaca' : ['Puerto Angel', 'Tehuacan', 'Izucar de Matamoros'],
            'Tehuacan' : [],
            'Puerto Angel' : ['Pinotepa Nacional'],
            'Pinotepa Nacional' : ['Acapulco'],
            'Acapulco' : ['Chilpancingo'],
            'Chilpancingo' : ['Iguala'],
            'Iguala' : ['Cuernavaca', 'Ciudad Altamirano'],
            'Izucar de Matamoros' : ['Puebla','Cuernavaca'],
            'Puebla' : ['Cordoba','Cdmx'],
            'Cordoba' : ['Veracruz'],
            'Veracruz' : [],
            'Cuernavaca' : ['Cdmx', 'Ciudad Altamirano'],
            'Cdmx' : ['Pachuca de Soto','Queretaro','Toluca de Lerdo','Tlaxcala'],
            'Tlaxcala' : [],
            'Toluca de Lerdo' : ['Ciudad Altamirano'],
            'Ciudad Altamirano' : ['Zihuatanejo'],
            'Zihuatanejo' : ['Playa Azul'],
            'Playa Azul' : ['Morelia','Colima','Manzanillo'],
            'Morelia' : ['Colima', 'Salamanca'],
            'Colima' : ['Manzanillo', 'Guadalajara'],
            'Manzanillo' : ['Guadalajara'],
            'Guadalajara' : ['Tepic','Aguascalientes'],
            'Salamanca' : ['Guanajuato','Guadalajara'],
            'Guanajuato' : ['Aguascalientes'],
            'Aguascalientes' : ['San Luis Potosi'],
            'Queretaro' : ['Salamanca', 'Atlacomulco', 'San Luis Potosi'],
            'Atlacomulco' : [],
            'Pachuca de Soto' : ['Tuxpan de Rodriguez Cano'],
            'Tuxpan de Rodriguez Cano' : ['Tampico'],
            'Tampico' : ['Ciudad Victoria'],
            'Ciudad Victoria' : ['Soto la Marina','Matamoros','Monterrey','Durango'],
            'Soto la Marina' : [],
            'San Luis Potosi' : ['Zacatecas', 'Durango'],
            'Zacatecas' :  [],
            'Tepic' : ['Mazatlan'],
            'Mazatlan' : ['Culiacan'],
            'Durango' : ['Mazatlan', 'Hidalgo del Parral', 'Torreon'],
            'Torreon' : ['Monclova'],
            'Matamoros' : ['Reynosa'],
            'Reynosa' : ['Nuevo Laredo'],
            'Monterrey' : ['Nuevo Laredo', 'Monclova'],
            'Nuevo Laredo' : ['Piedras Negras'],
            'Piedras Negras' : ['Monclova'],
            'Monclova' : ['Ojinaga'],
            'Culiacan' : ['Topolobampo', 'Hidalgo del Parral'],
            'Ojinaga' : ['Chihuahua'],
            'Hidalgo del Parral' :  ['Chihuahua'],
            'Chihuahua' : ['Ciudad Juárez','Janos'],
            'Ciudad Juárez' : [],
            'Topolobampo' : ['Ciudad Obregon', 'Hidalgo del Parral'],
            'Ciudad Obregon' : ['Guaymas'],
            'Guaymas' : ['Hermosillo'],
            'Hermosillo' : ['Santa Ana'],
            'Janos' : ['Agua Prieta'],
            'Agua Prieta' : ['Santa Ana'],
            'Santa Ana' : ['Mexicali'],
            'Mexicali' : ['San Felipe', 'Tijuana'],
            'Tijuana' : ['Ensenada'],
            'San Felipe' : ['Ensenada'],
            'Ensenada' : ['San Quintin'],
            'San Quintin' : ['Santa Rosalia'],
            'Santa Rosalia' : ['Santo Domingo'],
            'Santo Domingo' : ['La Paz'],
            'La Paz' : ['Cabo San Lucas'],
            'Cabo San Lucas' : []
        }

# Función menú
"""

Esta función esta para que muestre las opciones de los algorítmos que llevamos
a cabo en este proyecto, el menú va a poder delimitar que función pueda ver el 
usuario para que no tenga que ver todas las funciones al mismo tiempo o va a tener
la opción para que el usuario pueda ver todos los algoritmos al mismo tiempo

"""

def menu_muestra_pasos():
    print("Menú muestra para mostrar los pasos")
    print("1. Ver paso a paso la ejecuación")
    print("2. Ver solo el resultado")
    print("---------------------------------------------------------")
    print("")
    opcion_menu2 = input("Ingrese el número de la opción que desea: ")
    return opcion_menu2

def menu():#Joel Vázquez Anaya y Javier Vázquez Gurrola
    print("Menú:")
    print("1.  Greedy best-first")
    print("2.  A* con peso")
    print("3.  A*")
    print("4.  Beam")
    print("5.  Branch and Bound")
    print("6.  Steepest hil climbing")
    print("7.  Stochastic hil clambing")
    print("8.  simulated Annealing")
    print("9.  Genetic Algorithm")
    print("10. Todos los anteriores")
    print("11. Mostrar grafo procesado")
    print("12. Salir del programa")
    print("---------------------------------------------------------")
    print("")
    opcion = input("Ingrese el número de la opción que desea: ")
    print("")
    return opcion

opcion = None
opcion_menu2 = None
banderaOrigen = False
banderaDestino = False

while opcion != "12":
    opcion = menu()
    if opcion == "1":
        while opcion_menu2 != "1":
            opcion_menu2 = menu_muestra_pasos()
            if opcion_menu2 == "1": 
                while not banderaOrigen: #Joel Vázquez Anaya
                    start = input("Ingresa el nodo de origen: ")
                    start_verificado = start.title()
                    banderaOrigen = verificacion(start_verificado)

                while not banderaDestino: #Joel Vázquez Anaya
                    goal = input("Ingresa el nodo de destino: ")
                    goal_verificado = goal.title()
                    banderaDestino = verificacion(goal_verificado)
                print("-----------------------------------------------------------------------------")
                # Ejecutamos el algoritmo Greedy
                tiempo_inicio = time.time()
                path = greedy_con_pasos(graph, start_verificado, goal_verificado)
                if path is not None:
                    tiempo_fin = time.time()
                    tiempo_total_Greedy = (tiempo_fin - tiempo_inicio)
                    print("Resultado Greedy")
                    print(f"El camino más corto desde '{start_verificado}' hasta '{goal_verificado}' es: {path}")
                    print("La función tardó", tiempo_total_Greedy, "segundos en ejecutarse")
                else:
                    tiempo_fin = time.time()
                    tiempo_total_Greedy = (tiempo_fin - tiempo_inicio)
                    print(f"No se pudo encontrar un camino válido desde '{start_verificado}' hasta '{goal_verificado}'.")
                    print("La función tardó", tiempo_total_Greedy, "segundos en ejecutarse")
                print("----------------------------------------------------------------------------")
                banderaOrigen = False
                banderaDestino = False
                
            elif opcion_menu2 == "2":
                while not banderaOrigen: #Joel Vázquez Anaya
                    start = input("Ingresa el nodo de origen: ")
                    start_verificado = start.title()
                    banderaOrigen = verificacion(start_verificado)

                while not banderaDestino: #Joel Vázquez Anaya
                    goal = input("Ingresa el nodo de destino: ")
                    goal_verificado = goal.title()
                    banderaDestino = verificacion(goal_verificado)
                print("-----------------------------------------------------------------------------")
                # Ejecutamos el algoritmo Greedy
                tiempo_inicio = time.time()
                path = greedy(graph, start_verificado, goal_verificado)
                if path is not None:
                    tiempo_fin = time.time()
                    tiempo_total_Greedy = (tiempo_fin - tiempo_inicio)
                    print("Resultado Greedy")
                    print(f"El camino más corto desde '{start_verificado}' hasta '{goal_verificado}' es: {path}")
                    print("La función tardó", tiempo_total_Greedy, "segundos en ejecutarse")
                else:
                    tiempo_fin = time.time()
                    tiempo_total_Greedy = (tiempo_fin - tiempo_inicio)
                    print(f"No se pudo encontrar un camino válido desde '{start_verificado}' hasta '{goal_verificado}'.")
                    print("La función tardó", tiempo_total_Greedy, "segundos en ejecutarse")
                print("----------------------------------------------------------------------------")
                banderaOrigen = False
                banderaDestino = False
                
            else:
                print("Opción inválida, por favor seleccione una opción del 1 o 2")
                
    elif opcion == "2":
        opcion_menu2 = menu_muestra_pasos()
        banderaOrigen = False
        banderaDestino = False
        if opcion_menu2 == "1":
            while not banderaOrigen: #Joel Vázquez Anaya
                start = input("Ingresa el nodo de origen: ")
                start_verificado = start.title()
                banderaOrigen = verificacion(start_verificado)

            while not banderaDestino: #Joel Vázquez Anaya
                goal = input("Ingresa el nodo de destino: ")
                goal_verificado = goal.title()
                banderaDestino = verificacion(goal_verificado)    
            print("-----------------------------------------------------------------------------")
            w = float(input("Ingrese el peso(W): "))
            # Ejecutamos el algoritmo A* con peso
            tiempo_inicio = time.time()
            path = weighted_astar_con_pasos(start_verificado, goal_verificado, heuristic, successors, edge_cost, w)
            tiempo_fin = time.time()
            tiempo_total_A_pesos = (tiempo_fin - tiempo_inicio)
            print("Resultado weighted A*")
            print(path)
            print("La función tardó", tiempo_total_A_pesos, "segundos en ejecutarse")
            print("----------------------------------------------------------------------------")
            banderaOrigen = False
            banderaDestino = False

            
        elif opcion_menu2 == "2":
            while not banderaOrigen: #Joel Vázquez Anaya
                start = input("Ingresa el nodo de origen: ")
                start_verificado = start.title()
                banderaOrigen = verificacion(start_verificado)

            while not banderaDestino: #Joel Vázquez Anaya
                goal = input("Ingresa el nodo de destino: ")
                goal_verificado = goal.title()
                banderaDestino = verificacion(goal_verificado)    
            print("-----------------------------------------------------------------------------")
            w = float(input("Ingrese el peso(W): "))
            # Ejecutamos el algoritmo A* con peso
            tiempo_inicio = time.time()
            path = weighted_astar(start_verificado, goal_verificado, heuristic, successors, edge_cost, w)
            tiempo_fin = time.time()
            tiempo_total_A_pesos = (tiempo_fin - tiempo_inicio)
            print("Resultado weighted A*")
            print(path)
            print("La función tardó", tiempo_total_A_pesos, "segundos en ejecutarse")
            print("----------------------------------------------------------------------------")
            banderaOrigen = False
            banderaDestino = False
            
        else:
            print("Opción inválida, por favor seleccione una opción del 1 o 2")

    elif opcion == "3":
        opcion_menu2 = menu_muestra_pasos()        
        if opcion_menu2 == "1":
            while not banderaOrigen: #Joel Vázquez Anaya
                start = input("Ingresa el nodo de origen: ")
                start_verificado = start.title()
                banderaOrigen = verificacion(start_verificado)

            while not banderaDestino: #Joel Vázquez Anaya
                goal = input("Ingresa el nodo de destino: ")
                goal_verificado = goal.title()
                banderaDestino = verificacion(goal_verificado)
            print("-----------------------------------------------------------------------------")    
            # Ejecutamos el algoritmo A*
            tiempo_inicio = time.time()
            came_from, cost_so_far = astar_con_pasos(start_verificado, goal_verificado, graph, heuristic)
            tiempo_fin = time.time()
            tiempo_total_A = (tiempo_fin - tiempo_inicio)
            # Mostramos el resultado
            if goal not in came_from:
                print(f"No se encontró un camino desde {start_verificado} hasta {goal_verificado}")
                
            else:
            # Reconstruimos el camino desde el nodo inicial al nodo objetivo utilizando el diccionario de nodos antecesores
                path = [goal]
                node = goal
                while node != start:
                    node = came_from[node]
                    path.append(node)
                path.reverse()
            # Imprimimos el camino y el costo total
                print("Resultado A*")
                print(" -> ".join(node for node in path))
                print(f"Costo total: {cost_so_far[goal]}")
                print("La función tardó", tiempo_total_A, "segundos en ejecutarse")
            print("----------------------------------------------------------------------------")
            banderaOrigen = False
            banderaDestino = False
            break
            
        elif opcion_menu2 == "2":
            while not banderaOrigen: #Joel Vázquez Anaya
                start = input("Ingresa el nodo de origen: ")
                start_verificado = start.title()
                banderaOrigen = verificacion(start_verificado)

            while not banderaDestino: #Joel Vázquez Anaya
                goal = input("Ingresa el nodo de destino: ")
                goal_verificado = goal.title()
                banderaDestino = verificacion(goal_verificado)
            print("-----------------------------------------------------------------------------")    
            # Ejecutamos el algoritmo A*
            tiempo_inicio = time.time()
            came_from, cost_so_far = astar(start_verificado, goal_verificado, graph, heuristic)
            tiempo_fin = time.time()
            tiempo_total_A = (tiempo_fin - tiempo_inicio)
            # Mostramos el resultado
            if goal not in came_from:
                print(f"No se encontró un camino desde {start_verificado} hasta {goal_verificado}")
                
            else:
            # Reconstruimos el camino desde el nodo inicial al nodo objetivo utilizando el diccionario de nodos antecesores
                path = [goal]
                node = goal
                while node != start:
                    node = came_from[node]
                    path.append(node)
                path.reverse()
            # Imprimimos el camino y el costo total
                print("Resultado A*")
                print(" -> ".join(node for node in path))
                print(f"Costo total: {cost_so_far[goal]}")
                print("La función tardó", tiempo_total_A, "segundos en ejecutarse")
            print("----------------------------------------------------------------------------")
            banderaOrigen = False
            banderaDestino = False
            break
            
        else:
            print("Opción inválida, por favor seleccione una opción del 1 o 2")
        
    elif opcion == "4":
        opcion_menu2 = menu_muestra_pasos()    
        if opcion_menu2 == "1":
            while not banderaOrigen: #Joel Vázquez Anaya
                start = input("Ingresa el nodo de origen: ")
                start_verificado = start.title()
                banderaOrigen = verificacion(start_verificado)

            while not banderaDestino: #Joel Vázquez Anaya
                goal = input("Ingresa el nodo de destino: ")
                goal_verificado = goal.title()
                banderaDestino = verificacion(goal_verificado)
            print("----------------------------------------------------------------------------")
            beam_width = int(input("Ingrese el valor de anchura del beam en un número entero(beam_width): "))
            # Ejecutamos el algoritmo Beam
            tiempo_inicio = time.time()
            result = beam_search_con_pasos(start_verificado, lambda n: n == goal_verificado, expand_fn, beam_width, goal, heuristic)
            tiempo_fin = time.time()
            tiempo_total_Beam = (tiempo_fin - tiempo_inicio)
            print("Resultado Beam")
            print(result)
            print("La función tardó", tiempo_total_Beam, "segundos en ejecutarse")
            print("----------------------------------------------------------------------------")
            banderaOrigen = False
            banderaDestino = False
            
        elif opcion_menu2 == "2":
            while not banderaOrigen: #Joel Vázquez Anaya
                start = input("Ingresa el nodo de origen: ")
                start_verificado = start.title()
                banderaOrigen = verificacion(start_verificado)

            while not banderaDestino: #Joel Vázquez Anaya
                goal = input("Ingresa el nodo de destino: ")
                goal_verificado = goal.title()
                banderaDestino = verificacion(goal_verificado)
            print("----------------------------------------------------------------------------")
            beam_width = int(input("Ingrese el valor de anchura del beam en un número entero(beam_width): "))
            # Ejecutamos el algoritmo Beam
            tiempo_inicio = time.time()
            result = beam_search(start_verificado, lambda n: n == goal_verificado, expand_fn, beam_width, goal, heuristic)
            tiempo_fin = time.time()
            tiempo_total_Beam = (tiempo_fin - tiempo_inicio)
            print("Resultado Beam")
            print(result)
            print("La función tardó", tiempo_total_Beam, "segundos en ejecutarse")
            print("----------------------------------------------------------------------------")
            banderaOrigen = False
            banderaDestino = False
            
        else:
            print("Opción inválida, por favor seleccione una opción del 1 o 2")
        
    elif opcion == "5":
        opcion_menu2 = menu_muestra_pasos()
        if opcion_menu2 == "1":
            while not banderaOrigen: #Joel Vázquez Anaya
                start = input("Ingresa el nodo de origen: ")
                start_verificado = start.title()
                banderaOrigen = verificacion(start_verificado)

            while not banderaDestino: #Joel Vázquez Anaya
                goal = input("Ingresa el nodo de destino: ")
                goal_verificado = goal.title()
                banderaDestino = verificacion(goal_verificado)
            print("----------------------------------------------------------------------------")
            # Ejecutamos el algoritmo Branch and Bound
            tiempo_inicio = time.time()
            path, cost = branch_and_bound_shortest_path_con_pasos(graph, start_verificado, goal_verificado, heuristic)
            tiempo_fin = time.time()
            tiempo_total_Breanch_and_Bound = (tiempo_fin - tiempo_inicio)
            print("Resultado de Branch and Bound")
            print("Camino más corto:", path)
            print("Costo total:", cost)
            print("La función tardó", tiempo_total_Breanch_and_Bound, "segundos en ejecutarse")
            print("----------------------------------------------------------------------------")
            banderaOrigen = False
            banderaDestino = False
        
        elif opcion_menu2 == "2":
            while not banderaOrigen: #Joel Vázquez Anaya
                start = input("Ingresa el nodo de origen: ")
                start_verificado = start.title()
                banderaOrigen = verificacion(start_verificado)

            while not banderaDestino: #Joel Vázquez Anaya
                goal = input("Ingresa el nodo de destino: ")
                goal_verificado = goal.title()
                banderaDestino = verificacion(goal_verificado)
            print("----------------------------------------------------------------------------")
            # Ejecutamos el algoritmo Branch and Bound
            tiempo_inicio = time.time()
            path, cost = branch_and_bound_shortest_path(graph, start_verificado, goal_verificado, heuristic)
            tiempo_fin = time.time()
            tiempo_total_Breanch_and_Bound = (tiempo_fin - tiempo_inicio)
            print("Resultado de Branch and Bound")
            print("Camino más corto:", path)
            print("Costo total:", cost)
            print("La función tardó", tiempo_total_Breanch_and_Bound, "segundos en ejecutarse")
            print("----------------------------------------------------------------------------")
            banderaOrigen = False
            banderaDestino = False
        
        else:
            print("Opción inválida, por favor seleccione una opción del 1 o 2")
            
        
    elif opcion == "6":
        opcion_menu2 = menu_muestra_pasos()
        if opcion_menu2 == "1":
            while not banderaOrigen: #Joel Vázquez Anaya
                start = input("Ingresa el nodo de origen: ")
                start_verificado = start.title()
                banderaOrigen = verificacion(start_verificado)
            print("----------------------------------------------------------------------------")    
            #Ejecución del aloritmo Steepest Hil Climbing con pasos
            tiempo_inicio = time.time()
            resultado = steepest_hill_climbing_con_pasos(graph, start_verificado)
            tiempo_fin = time.time()
            tiempo_total_Steepest_Hil_Climbing = (tiempo_fin - tiempo_inicio)
            print("El resultado de Steepest hil climbing")
            print(resultado)
            print("La función tardó", tiempo_total_Steepest_Hil_Climbing, "segundos en ejecutarse")
            print("----------------------------------------------------------------------------")
            banderaOrigen = False
            banderaDestino = False
            
        elif opcion_menu2 == "2":
            while not banderaOrigen: #Joel Vázquez Anaya
                start = input("Ingresa el nodo de origen: ")
                start_verificado = start.title()
                banderaOrigen = verificacion(start_verificado)
            print("----------------------------------------------------------------------------")
            #Ejecución del aloritmo Steepest Hil Climbing sin pasos
            tiempo_inicio = time.time()
            resultado = steepest_hill_climbing(graph, start_verificado)
            tiempo_fin = time.time()
            tiempo_total_Steepest_Hil_Climbing = (tiempo_fin - tiempo_inicio)
            print("El resultado de Steepest hil climbing")
            print(resultado)
            print("La función tardó", tiempo_total_Steepest_Hil_Climbing, "segundos en ejecutarse")
            print("----------------------------------------------------------------------------")
            banderaOrigen = False
            banderaDestino = False
            
        else:
            print("Opción inválida, por favor seleccione una opción del 1 o 2")
        
    elif opcion == "7":
        opcion_menu2 = menu_muestra_pasos()            
        if opcion_menu2 == "1": 
            while not banderaOrigen: #Joel Vázquez Anaya
                start = input("Ingresa el nodo de origen: ")
                start_verificado = start.title()
                banderaOrigen = verificacion(start_verificado)
                print("----------------------------------------------------------------------------")
            
            #Ejecución del algoritmo Stochastic Hil Clambing con pasos
            tiempo_inicio = time.time()
            resultado_stochastic = stochastic_hill_climbing_con_pasos(graph, start_verificado, heuristic)
            tiempo_fin = time.time()
            tiempo_total_Stochastic_hil_clambing = (tiempo_fin - tiempo_inicio)
            print("El resultado de Stochastic hil clambing")
            print(resultado_stochastic)
            print("La función tardó", tiempo_total_Stochastic_hil_clambing, "segundos en ejecutarse")
            print("----------------------------------------------------------------------------")
            banderaOrigen = False
                
            
        elif opcion_menu2 == "2":
            while not banderaOrigen: #Joel Vázquez Anaya
                start = input("Ingresa el nodo de origen: ")
                start_verificado = start.title()
                banderaOrigen = verificacion(start_verificado)
                print("----------------------------------------------------------------------------")
            
            #Ejecución del algoritmo Stochastic Hil Clambing sin pasos
            tiempo_inicio = time.time()
            resultado_stochastic = stochastic_hill_climbing(graph, start_verificado, heuristic)
            tiempo_fin = time.time()
            tiempo_total_Stochastic_hil_clambing = (tiempo_fin - tiempo_inicio)
            print("El resultado de Stochastic hil clambing")
            print(resultado_stochastic)
            print("La función tardó", tiempo_total_Stochastic_hil_clambing, "segundos en ejecutarse")
            print("----------------------------------------------------------------------------")
            banderaOrigen = False
                
        
        else:
            print("Opción inválida, por favor seleccione una opción del 1 o 2")        
        
    elif opcion == "8":
        opcion_menu2 = menu_muestra_pasos()
        if opcion_menu2 == "1": 
            #Ejecución del algoritmo simulated annealing con pasos
            while not banderaOrigen: 
                start = input("Ingresa el nodo de origen: ")
                start_verificado = start.title()
                banderaOrigen = verificacion(start_verificado)

            while not banderaDestino: #Joel Vázquez Anaya
                goal = input("Ingresa el nodo de destino: ")
                goal_verificado = goal.title()
                banderaDestino = verificacion(goal_verificado)
                print("----------------------------------------------------------------------------")
           
            print("----------------------------------------------------------------------------")
        
        elif opcion_menu2 == "2":
            while not banderaOrigen: 
                start = input("Ingresa el nodo de origen: ")
                start_verificado = start.title()
                banderaOrigen = verificacion(start_verificado)

            while not banderaDestino: #Joel Vázquez Anaya
                goal = input("Ingresa el nodo de destino: ")
                goal_verificado = goal.title()
                banderaDestino = verificacion(goal_verificado)
                print("----------------------------------------------------------------------------")
            #Ejecución del algoritmo simulated annealing
            tiempo_inicio = time.time()
            G = nx.Graph(adjacency_list)
            max_iterations = int(input("Ingrese la cantidad de iteraciones maximas: "))
            temperature = int(input("Ingrese la temperatura inicial: "))
            best_state, best_cost = simulated_annealing(start_verificado, goal_verificado, adjacency_list, max_iterations, temperature)
            print("Mejor estado encontrado:", best_state)
            print("Costo del mejor estado:", best_cost)
            dibujar_mapa()
            tiempo_fin = time.time()
            tiempo_total_Traveling_Salesman = (tiempo_fin - tiempo_inicio)
            print("La función tardó", tiempo_total_Traveling_Salesman, "segundos en ejecutarse")
            print("----------------------------------------------------------------------------")
        
        else:
            print("Opción inválida, por favor seleccione una opción del 1 o 2")  

    elif opcion == "9":
        opcion_menu2 = menu_muestra_pasos()
        if opcion_menu2 == "1":
            population_size = int(input("Ingrese el tamaño de la población: "))
            num_generations = int(input("Ingrese el número de generaciones: "))
            mutation_rate = float(input("Ingrese la taza de mutación que quiere que tenga su población(La taza de mutació puede estar entre el 0 y 1): "))
            #Ejecución del algoritmo Genetic Algorithm
            tiempo_inicio = time.time()
            Resultado_genetic = genetic_algorithm_con_pasos(graph, population_size, num_generations, mutation_rate)
            tiempo_fin = time.time()
            tiempo_total_Genetic_Algorithm = (tiempo_fin - tiempo_inicio)
            print("El resultado de Genetic Algorithm")
            print(Resultado_genetic)
            print("La función tardó", tiempo_total_Genetic_Algorithm, "segundos en ejecutarse")
            print("----------------------------------------------------------------------------")
            opcion_menu2 = "0"
        elif opcion_menu2 == "2":
            population_size = int(input("Ingrese el tamaño de la población: "))
            num_generations = int(input("Ingrese el número de generaciones: "))
            mutation_rate = float(input("Ingrese la taza de mutación que quiere que tenga su población(La taza de mutació puede estar entre el 0 y 1): "))
            #Ejecución del algoritmo Genetic Algorithm
            tiempo_inicio = time.time()
            Resultado_genetic = genetic_algorithm(graph, population_size, num_generations, mutation_rate)
            tiempo_fin = time.time()
            tiempo_total_Genetic_Algorithm = (tiempo_fin - tiempo_inicio)
            print("El resultado de Genetic Algorithm")
            print(Resultado_genetic)
            print("La función tardó", tiempo_total_Genetic_Algorithm, "segundos en ejecutarse")
            print("----------------------------------------------------------------------------")
        else:
          print("----------------------------------------------------------------------------")  
            
    elif opcion == "10":
        opcion_menu2 = menu_muestra_pasos()
        if opcion_menu2 == "1":
            while not banderaOrigen: #Joel Vázquez Anaya
                start = input("Ingresa el nodo de origen: ")
                start_verificado = start.title()
                banderaOrigen = verificacion(start_verificado)

            while not banderaDestino: #Joel Vázquez Anaya
                goal = input("Ingresa el nodo de destino: ")
                goal_verificado = goal.title()
                banderaDestino = verificacion(goal_verificado)
            print("-----------------------------------------------------------------------------")
            # Ejecutamos el algoritmo Greedy
            
            print("-------------------------------Greedy------------------------------------------")
            tiempo_inicio = time.time()
            path = greedy_con_pasos(graph, start_verificado, goal_verificado)
            if path is not None:
                tiempo_fin = time.time()
                tiempo_total_Greedy = (tiempo_fin - tiempo_inicio)
                print("Resultado Greedy")
                print(f"El camino más corto desde '{start_verificado}' hasta '{goal_verificado}' es: {path}")
                print("La función tardó", tiempo_total_Greedy, "segundos en ejecutarse")
            else:
                tiempo_fin = time.time()
                tiempo_total_Greedy = (tiempo_fin - tiempo_inicio)
                print(f"No se pudo encontrar un camino válido desde '{start_verificado}' hasta '{goal_verificado}'.")
                print("La función tardó", tiempo_total_Greedy, "segundos en ejecutarse")
            print("----------------------------------------------------------------------------")
            print("")
                
            # Ejecutamos el algoritmo A* con peso
            print("---------------------------------A* con peso---------------------------------")
            w = float(input("Ingrese el peso(W): "))
            tiempo_inicio = time.time()
            path = weighted_astar_con_pasos(start_verificado, goal_verificado, heuristic, successors, edge_cost, w)
            tiempo_fin = time.time()
            tiempo_total_A_pesos = (tiempo_fin - tiempo_inicio)
            print("Resultado weighted A*")
            print(path)
            print("La función tardó", tiempo_total_A_pesos, "segundos en ejecutarse")
            print("----------------------------------------------------------------------------")
                
            # Ejecutamos el algoritmo A*
            print("--------------------------------- A* ---------------------------------")
            tiempo_inicio = time.time()
            came_from, cost_so_far = astar_con_pasos(start_verificado, goal_verificado, graph, heuristic)
            tiempo_fin = time.time()
            tiempo_total_A = (tiempo_fin - tiempo_inicio)
            # Mostramos el resultado
            if goal not in came_from:
                print(f"No se encontró un camino desde {start_verificado} hasta {goal_verificado}")
                opcion_menu2 = "0"
            else:
            # Reconstruimos el camino desde el nodo inicial al nodo objetivo utilizando el diccionario de nodos antecesores
                path = [goal]
                node = goal
                while node != start:
                    node = came_from[node]
                    path.append(node)
                path.reverse()
            # Imprimimos el camino y el costo total
                print("Resultado A*")
                print(" -> ".join(node for node in path))
                print(f"Costo total: {cost_so_far[goal]}")
                print("La función tardó", tiempo_total_A, "segundos en ejecutarse")
            print("----------------------------------------------------------------------------")
                
            # Ejecutamos el algoritmo Beam
            print("--------------------------------- Beam ---------------------------------")
            beam_width = int(input("Ingrese el valor de anchura del beam en un número entero(beam_width): "))
            tiempo_inicio = time.time()
            result = beam_search_con_pasos(start_verificado, lambda n: n == goal_verificado, expand_fn, beam_width, goal, heuristic)
            tiempo_fin = time.time()
            tiempo_total_Beam = (tiempo_fin - tiempo_inicio)
            print("Resultado Beam")
            print(result)
            print("La función tardó", tiempo_total_Beam, "segundos en ejecutarse")
            print("----------------------------------------------------------------------------")
                
            # Ejecutamos el algoritmo Branch and Bound
            print("---------------------------------Branch and Bound---------------------------------")
            tiempo_inicio = time.time()
            path, cost = branch_and_bound_shortest_path_con_pasos(graph, start_verificado, goal_verificado, heuristic)
            tiempo_fin = time.time()
            tiempo_total_Breanch_and_Bound = (tiempo_fin - tiempo_inicio)
            print("Resultado de Branch and Bound")
            print("Camino más corto:", path)
            print("Costo total:", cost)
            print("La función tardó", tiempo_total_Breanch_and_Bound, "segundos en ejecutarse")
            print("----------------------------------------------------------------------------")
                
            #Ejecución del aloritmo Steepest Hil Climbing con pasos
            print("--------------------------------- Steepest Hil Climbing ---------------------------------")
            tiempo_inicio = time.time()
            resultado = steepest_hill_climbing_con_pasos(graph, start_verificado)
            tiempo_fin = time.time()
            tiempo_total_Steepest_Hil_Climbing = (tiempo_fin - tiempo_inicio)
            print("El resultado de Steepest hil climbing")
            print(resultado)
            print("La función tardó", tiempo_total_Steepest_Hil_Climbing, "segundos en ejecutarse")
            print("----------------------------------------------------------------------------")
                
            #Ejecución del algoritmo Stochastic Hil Clambing con pasos
            print("--------------------------------- Stochastic Hil Clambing ---------------------------------")
            tiempo_inicio = time.time()
            resultado_stochastic = stochastic_hill_climbing_con_pasos(graph, start_verificado, heuristic)
            tiempo_fin = time.time()
            tiempo_total_Stochastic_hil_clambing = (tiempo_fin - tiempo_inicio)
            print("El resultado de Stochastic hil clambing")
            print(resultado_stochastic)
            print("La función tardó", tiempo_total_Stochastic_hil_clambing, "segundos en ejecutarse")
            print("----------------------------------------------------------------------------")
                
            #Agregar código de Cisco
                
            #Ejecución del algoritmo Genetic Algorithm
            print("--------------------------------- Genetic Algorithm ---------------------------------")
            population_size = int(input("Ingrese el tamaño de la población: "))
            num_generations = int(input("Ingrese el número de generaciones: "))
            mutation_rate = float(input("Ingrese la taza de mutación que quiere que tenga su población(La taza de mutació puede estar entre el 0 y 1): "))
            tiempo_inicio = time.time()
            Resultado_genetic = genetic_algorithm_con_pasos(graph, population_size, num_generations, mutation_rate)
            tiempo_fin = time.time()
            tiempo_total_Genetic_Algorithm = (tiempo_fin - tiempo_inicio)
            print("El resultado de Genetic Algorithm")
            print(Resultado_genetic)
            print("La función tardó", tiempo_total_Genetic_Algorithm, "segundos en ejecutarse")
            print("----------------------------------------------------------------------------")
            banderaOrigen = False
            banderaDestino = False
            
            
        elif opcion_menu2 == "2":
            while not banderaOrigen: #Joel Vázquez Anaya
                    start = input("Ingresa el nodo de origen: ")
                    start_verificado = start.title()
                    banderaOrigen = verificacion(start_verificado)

            while not banderaDestino: #Joel Vázquez Anaya
                goal = input("Ingresa el nodo de destino: ")
                goal_verificado = goal.title()
                banderaDestino = verificacion(goal_verificado)
                
            # Ejecutamos el algoritmo Greedy
            print("-------------------------------Greedy------------------------------------------")
            tiempo_inicio = time.time()
            path = greedy(graph, start_verificado, goal_verificado)
            if path is not None:
                tiempo_fin = time.time()
                tiempo_total_Greedy = (tiempo_fin - tiempo_inicio)
                print("Resultado Greedy")
                print(f"El camino más corto desde '{start_verificado}' hasta '{goal_verificado}' es: {path}")
                print("La función tardó", tiempo_total_Greedy, "segundos en ejecutarse")
            else:
                tiempo_fin = time.time()
                tiempo_total_Greedy = (tiempo_fin - tiempo_inicio)
                print(f"No se pudo encontrar un camino válido desde '{start_verificado}' hasta '{goal_verificado}'.")
                print("La función tardó", tiempo_total_Greedy, "segundos en ejecutarse")
            print("----------------------------------------------------------------------------")

            # Ejecutamos el algoritmo A* con peso
            print("---------------------------------A* con peso---------------------------------")
            w = float(input("Ingrese el peso(W): "))
            tiempo_inicio = time.time()
            path = weighted_astar(start_verificado, goal_verificado, heuristic, successors, edge_cost, w)

            # Ejecutamos el algoritmo A*
            print("--------------------------------- A* ---------------------------------")
            tiempo_inicio = time.time()
            came_from, cost_so_far = astar(start_verificado, goal_verificado, graph, heuristic)
            tiempo_fin = time.time()
            tiempo_total_A = (tiempo_fin - tiempo_inicio)
            # Mostramos el resultado
            if goal not in came_from:
                print(f"No se encontró un camino desde {start_verificado} hasta {goal_verificado}")
            else:
            # Reconstruimos el camino desde el nodo inicial al nodo objetivo utilizando el diccionario de nodos antecesores
                path = [goal]
                node = goal
                while node != start_verificado:
                    node = came_from[node]
                    path.append(node)
                path.reverse()
            # Imprimimos el camino y el costo total
                print("Resultado A*")
                print(" -> ".join(node for node in path))
                print(f"Costo total: {cost_so_far[goal]}")
                print("La función tardó", tiempo_total_A, "segundos en ejecutarse")
            print("----------------------------------------------------------------------------")

            # Ejecutamos el algoritmo Beam
            print("--------------------------------- Beam ---------------------------------")
            beam_width = int(input("Ingrese el valor de anchura del beam en un número entero(beam_width): "))
            tiempo_inicio = time.time()
            result = beam_search(start_verificado, lambda n: n == goal_verificado, expand_fn, beam_width, goal_verificado, heuristic)
            tiempo_fin = time.time()
            tiempo_total_Beam = (tiempo_fin - tiempo_inicio)
            print("Resultado Beam")
            print(result)
            print("La función tardó", tiempo_total_Beam, "segundos en ejecutarse")
            print("----------------------------------------------------------------------------")

            # Ejecutamos el algoritmo Branch and Bound
            print("---------------------------------Branch and Bound---------------------------------")
            tiempo_inicio = time.time()
            path, cost = branch_and_bound_shortest_path(graph, start_verificado, goal_verificado, heuristic)
            tiempo_fin = time.time()
            tiempo_total_Breanch_and_Bound = (tiempo_fin - tiempo_inicio)
            print("Resultado de Branch and Bound")
            print("Camino más corto:", path)
            print("Costo total:", cost)
            print("La función tardó", tiempo_total_Breanch_and_Bound, "segundos en ejecutarse")
            print("----------------------------------------------------------------------------")

            #Ejecución del aloritmo Steepest hil climbing
            print("--------------------------------- Steepest Hil Climbing ---------------------------------")
            tiempo_inicio = time.time()
            resultado = steepest_hill_climbing(graph, start_verificado)
            tiempo_fin = time.time()
            tiempo_total_Steepest_Hil_Climbing = (tiempo_fin - tiempo_inicio)
            print("El resultado de steepest hil climbing")
            print(resultado)
            print("La función tardó", tiempo_total_Steepest_Hil_Climbing, "segundos en ejecutarse")
            print("----------------------------------------------------------------------------")

            #Ejecución del algoritmo Stochastic hil clambing
            print("--------------------------------- Stochastic Hil Clambing ---------------------------------")
            tiempo_inicio = time.time()
            resultado_stochastic = stochastic_hill_climbing(graph, start_verificado, heuristic)
            tiempo_fin = time.time()
            tiempo_total_Stochastic_hil_clambing = (tiempo_fin - tiempo_inicio)
            print("El resultado de Stochastic hil clambing")
            print(resultado_stochastic)
            print("La función tardó", tiempo_total_Stochastic_hil_clambing, "segundos en ejecutarse")
            print("----------------------------------------------------------------------------")

            #Agregar código cisco

            #Ejecución del algoritmo Genetic Algorithm
            print("--------------------------------- Genetic Algorithm ---------------------------------")
            population_size = int(print("Ingrese el tamaño de la población: "))
            num_generations = int(print("Ingrese el número de generaciones: "))
            mutation_rate = float(print("Ingrese la taza de mutación que quiere que tenga su población: "))

            tiempo_inicio = time.time()
            Resultado_genetic = genetic_algorithm(graph, population_size, num_generations, mutation_rate)
            tiempo_fin = time.time()
            tiempo_total_Genetic_Algorithm = (tiempo_fin - tiempo_inicio)
            print("El resultado de Genetic Algorithm")
            print(Resultado_genetic)
            print("La función tardó", tiempo_total_Genetic_Algorithm, "segundos en ejecutarse")
            print("----------------------------------------------------------------------------")
            banderaOrigen = False
            banderaDestino = False
        
    elif opcion == "11":
        imprimir(graph)
        print("")
        
    elif opcion == "12":
        sys.exit("Gracias por usar nuestro algoritmo, pónganos 10 profe :)")
    
else:
        print("Opción inválida, por favor seleccione una opción del 1 al 11")
        print("")
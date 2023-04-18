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

#Infromación General
"""
Búsquedas Informadas

Universidad Panamericana

Inteligencia Artificial

Integrantes:
-Javier Vázquez Gurrola (0215391)
-Joel Vázquez Anaya (0201031)
-Francisco Anaya Viveros (0181879)

Fecha de entrega 26/04/2023

Versión 1.5

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
    latitudes = {'A': 40.4168, 'B': 40.4146, 'C': 40.4262,
                 'D': 40.4301, 'E': 40.4266, 'F': 40.4297}
    longitudes = {'A': -3.7038, 'B': -3.7015, 'C': -3.6764,
                  'D': -3.7007, 'E': -3.6974, 'F': -3.6984}
    # Distancia Haversine
    R = 6372.8  # Radio de la Tierra en kilómetros
    
    # Convertir coordenadas del nodo y del objetivo a radianes
    lat1, lon1 = radians(latitudes[node]), radians(longitudes[node])
    lat2, lon2 = radians(latitudes[goal]), radians(longitudes[goal])
    
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

"""
La función greedy realiza una búsqueda Greedy Best-First para encontrar el camino más corto desde el nodo "start" hasta el nodo "goal". 
El código es una implementación del algoritmo de búsqueda Greedy best-first search que utiliza una cola de prioridad para explorar 
los nodos en un orden determinado por una heurística. 
Se inicializa una cola de prioridad "frontier" y un conjunto de nodos explorados "explored".
El nodo de inicio se inserta en la cola de prioridad con una prioridad de 0.
Se comienza a iterar mientras la cola de prioridad no esté vacía.
Se extrae el primer elemento de la cola de prioridad, que es el nodo con la menor prioridad, y se almacena en la variable current.
Si el nodo actual es el nodo objetivo, se imprime el conjunto de nodos explorados y se rompe el ciclo.
El nodo actual se agrega al conjunto de nodos explorados.
Se exploran todos los vecinos del nodo actual. Si un vecino no ha sido explorado previamente, se calcula su prioridad mediante la heurística
y se inserta en la cola de prioridad.
Al final del ciclo, se devuelve el conjunto de nodos explorados.
La heurística utilizada en el código es la función "heuristic", que toma un nodo y devuelve la distancia Haversine entre ese nodo y el nodo objetivo.

"""
def greedy(graph, start, goal): #Javier Vázquez Gurrola 
   frontier = PriorityQueue()
   frontier.put(start, 0)
   explored = set()
   while not frontier.empty():
    current = frontier.get()
    if current == goal:
        print(explored)
        break
    explored.add(current)
    for neighbor in graph[current]:
        if neighbor not in explored:
            priority = heuristic(neighbor)
            frontier.put(neighbor, priority)
    return explored
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

def weighted_astar(start, goal, heuristic, successors, edge_cost, w=1): #Javier Vázquez Gurrola 
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

# Función de steepest hil climbing
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

#Función Stochastic hil clambing
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
"""
Es una búsqueda que se usa para encontrar el camino más ótimo al comparar todos los caminos
para descubrir este resultado existen muchas maneras de sacarlo, como por medio de analisis
random, la ruta de una hormiga al hormiguero y el de la función del viajero.
Para resolver este caso se uso la función del viajero.
Cantidad de vectores que se van a analizar, podemos pensarlos como los nodos que se usan
más abajo para imprimir el grafo.

"""
v = 6
#La función traelling salesman sirve para calcular el camino más corto
def travelling_salesman_function(graph2, s):#Francisco Anaya Viveros
  #lista vacia para almacenar el camino
  vertex = []
  #analizará para la cantidad de puntos
  for i in range(v):
    #si nuestra i es diferente de s se hace una append al vertex
    if i != s:
      vertex.append(i)
  
  min_path = maxsize
  #Loop infinito
  while True:
    #costo incial
    current_cost = 0
    k = s
    #haces el loop otra vez pero ahora con vertex
    for i in range(len(vertex)):
      #mi costo total debe incrementarse por el costo de vertex
      current_cost += graph2[k][vertex[i]]
      #vamos a guardar la posición del vertex cada vez que se guarde un valor
      k = vertex[i]
    current_cost += graph2[k][s]
    #se va a calcular el vamor mínimo de un grafo comparando el valor mínimo con el del costo actual
    min_path = min(min_path, current_cost)
    #es una función que nos va a decir si existiera otro camino mínimo
    if not next_perm(vertex):
      break
  return min_path
#se llama a la lista
def next_perm(l):
  n = len(l)
  i = n-2
#si estoy en 1 puedo ir a 2 o 3 entonces veo la que tenga mayor costo y la eliminamos
  while i >= 0 and l[i] > l[i+1]:
    i -= 1
#cuando nuestra i sea -1 significa que no se puede encontrar un camino mínimo
  if i == -1:
    return False
#cuando se encuentra un camino mínimo entonces se ocupa la función con j
  j = i+1
  #solo comparo si es menos costo ir primero a 3 que a 2 o si al revés
  while j < n and l[j] > l[i]:
    j += 1
  #cuando ya agrege ese camino me concentraré en un nuevo camino de ahí la reducción en 1
  j -= 1
  #va a servir para saber que camino tomar
  l[i], l[j] = l[j], l[i]
  left = i+1
  right = n-1
  #si por ejemplo estamos ya en 3, si deberíamos tomar camino hacía el punto 2 o 4
  while left < right:
    l[left], l[right] = l[right], l[left]
    left += 1
    right -= 1
  return True



#Función Genetic Algorithm
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
def generate_initial_population(population_size, graph):
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
def fitness_function(chromosome, graph):
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
def select_parents(population):
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
def generate_offspring(parent1, parent2):
    if len(parent1) <= 1:  # verificación de longitud de parent1
        return parent1
    crossover_point = random.randrange(1, len(parent1))
    child = parent1[:crossover_point] + parent2[crossover_point:]
    if random.random() < mutation_rate:
        mutate_population(child)
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
def mutate_population(population, mutation_rate, graph):
    mutated_population = []
    for chromosome in population:
        mutated_chromosome = list(chromosome)
        for i in range(len(chromosome)):
            if random.random() < mutation_rate:
                mutated_chromosome[i] = random.choice(list(graph.keys()))
        mutated_population.append(mutated_chromosome)
    return mutated_population


graph = {
    'A': {'B': 1, 'C': 2},
    'B': {'A': 1, 'C': 1, 'D': 2},
    'C': {'A': 2, 'B': 1, 'D': 1},
    'D': {'B': 2, 'C': 1, 'E': 3},
    'E': {'D': 3, 'F': 1},
    'F': {'E': 1}
}

start = 'A'
goal = 'E'
beam_width = 3

#imprimir el grafo

def print_graph(G):
  # nodos
  pos = nx.spring_layout(G, seed=7)
  #color para poner el nombre en este caso azul
  nx.draw_networkx_nodes(G, pos, node_color = 'blue', node_size = 5000)
  #color de la letra que irá dentro del círculo
  nx.draw_networkx_labels(G, pos, font_size=10, font_family="sans-serif", font_color='white')

  # líneas
  nx.draw_networkx_edges(
      #líneas para marcar los pesos dentro que serán en color negro con una estilo dashed
      G, pos, edgelist=G.edges, width=6, alpha=1, edge_color="black", style="dashed"
      )
  edge_labels = nx.get_edge_attributes(G, "weight")
  #el texto del peso será hecho con el color de fuente rojo
  nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=10, font_color='red')

  # ploting
  ax = plt.gca()
  ax.margins(0.08)
  plt.axis("off")
  plt.tight_layout()
  plt.show()

G = nx.DiGraph()

G.add_weighted_edges_from({
    #cada nodo con sus pesos de donde se acomodarán para hacer el grafo después
    ("Sibiu", "Rimnicu Vilcea", 80), ("Sibiu", "Fagaras", 99),
    ("Rimnicu Vilcea", "Sibiu", 80), ("Fagaras", "Sibiu", 99),
    ("Rimnicu Vilcea", "Craiova", 146), ("Rimnicu Vilcea", "Pitesti", 97),
    ("Craiova", "Rimnicu Vilcea", 146), ("Pitesti", "Rimnicu Vilcea", 97),
    ("Craiova", "Pitesti", 138),
    ("Pitesti", "Craiova", 138),
    ("Pitesti", "Bucharest", 101),
    ("Bucharest", "Pitesti", 101),
    ("Fagaras", "Bucharest", 211),
    ("Bucharest", "Fagaras", 211),
    ("Sibiu", "Craiova", 1000), ("Sibiu", "Pitesti", 1000), ("Sibiu", "Bucharest", 1000),
    ("Craiova", "Sibiu", 1000), ("Pitesti", "Sibiu", 1000), ("Bucharest", "Sibiu", 1000),
    ("Rimnicu Vilcea", "Fagaras", 1000), ("Rimnicu Vilcea", "Bucharest", 1000),
    ("Fagaras", "Rimnicu Vilcea", 1000), ("Bucharest", "Rimnicu Vilcea", 1000),
    ("Craiova", "Bucharest", 1000), ("Craiova", "Fagaras", 1000),
    ("Bucharest", "Craiova", 1000), ("Fagaras", "Craiova", 1000),
    ("Pitesti", "Fagaras", 1000),
    ("Fagaras", "Pitesti", 1000),
})

# Función menú
"""

Esta función esta para que muestre las opciones de los algorítmos que llevamos
a cabo en este proyecto, el menú va a poder delimitar que función pueda ver el 
usuario para que no tenga que ver todas las funciones al mismo tiempo o va a tener
la opción para que el usuario pueda ver todos los algoritmos al mismo tiempo

"""
def menu():
    print("Menú:")
    print("1.  Greedy best-first")
    print("2.  A* con peso")
    print("3.  A*")
    print("4.  Beam")
    print("5.  Branch and Bound")
    print("6.  Steepest hil climbing")
    print("7.  Stochastic hil clambing")
    print("8.  Traveling Salesman")
    print("9.  Genetic Algorithm")
    print("10. Todos los anteriores")
    print("11. Salir del programa")
    opcion = input("Ingrese el número de la opción que desea: ")
    print("")
    return opcion

opcion = None
while opcion != "11":
    opcion = menu()
    if opcion == "1":
        # Ejecutamos el algoritmo Greedy
        tiempo_inicio = time.time()
        path = greedy(graph, 'A', 'F')
        if path is not None:
            tiempo_fin = time.time()
            tiempo_total_Greedy = (tiempo_fin - tiempo_inicio) * 1000
            print("Resultado Greedy")
            print(f"El camino más corto desde '{start}' hasta '{goal}' es: {path}")
            print("La función tardó", tiempo_total_Greedy, "segundos en ejecutarse")
        else:
            tiempo_fin = time.time()
            tiempo_total_Greedy = (tiempo_fin - tiempo_inicio) * 1000
            print(f"No se pudo encontrar un camino válido desde '{start}' hasta '{goal}'.")
            print("La función tardó", tiempo_total_Greedy, "segundos en ejecutarse")
        print("----------------------------------------------------------------------------")
        
    elif opcion == "2":
        # Ejecutamos el algoritmo A* con peso
        tiempo_inicio = time.time()
        path = weighted_astar(start, goal, heuristic, successors, edge_cost, w=1.5)
        tiempo_fin = time.time()
        tiempo_total_A_pesos = (tiempo_fin - tiempo_inicio)*1000
        print("Resultado weighted A*")
        print(path)
        print("La función tardó", tiempo_total_A_pesos, "segundos en ejecutarse")
        print("----------------------------------------------------------------------------")
        
    elif opcion == "3":
        # Ejecutamos el algoritmo A*
        tiempo_inicio = time.time()
        came_from, cost_so_far = astar(start, goal, graph, heuristic)
        tiempo_fin = time.time()
        tiempo_total_A = (tiempo_fin - tiempo_inicio) * 1000
        # Mostramos el resultado
        if goal not in came_from:
            print(f"No se encontró un camino desde {start} hasta {goal}")
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
        
    elif opcion == "4":
        # Ejecutamos el algoritmo Beam
        tiempo_inicio = time.time()
        result = beam_search(start, lambda n: n == goal, expand_fn, beam_width, goal, heuristic)
        tiempo_fin = time.time()
        tiempo_total_Beam = (tiempo_fin - tiempo_inicio) * 1000
        print("Resultado Beam")
        print(result)
        print("La función tardó", tiempo_total_Beam, "segundos en ejecutarse")
        print("----------------------------------------------------------------------------")
        
    elif opcion == "5":
        # Ejecutamos el algoritmo Branch and Bound
        tiempo_inicio = time.time()
        path, cost = branch_and_bound_shortest_path(graph, start, goal, heuristic)
        tiempo_fin = time.time()
        tiempo_total_Breanch_and_Bound = (tiempo_fin - tiempo_inicio) * 1000
        print("Resultado de Branch and Bound")
        print("Camino más corto:", path)
        print("Costo total:", cost)
        print("La función tardó", tiempo_total_Breanch_and_Bound, "segundos en ejecutarse")
        print("----------------------------------------------------------------------------")
        
    elif opcion == "6":
         #Ejecución del aloritmo Steepest hil climbing
        tiempo_inicio = time.time()
        resultado = steepest_hill_climbing(graph, start)
        tiempo_fin = time.time()
        tiempo_total_Steepest_Hil_Climbing = (tiempo_fin - tiempo_inicio) * 1000
        print("El resultado de Steepest hil climbing")
        print(resultado)
        print("La función tardó", tiempo_total_Steepest_Hil_Climbing, "segundos en ejecutarse")
        print("----------------------------------------------------------------------------")
        
    elif opcion == "7":
        #Ejecución del algoritmo Stochastic hil clambing
        tiempo_inicio = time.time()
        resultado_stochastic = stochastic_hill_climbing(graph, start, heuristic)
        tiempo_fin = time.time()
        tiempo_total_Stochastic_hil_clambing = (tiempo_fin - tiempo_inicio) * 1000
        print("El resultado de Stochastic hil clambing")
        print(resultado_stochastic)
        print("La función tardó", tiempo_total_Stochastic_hil_clambing, "segundos en ejecutarse")
        print("----------------------------------------------------------------------------")
        
    elif opcion == "8":
        #Ejecución del algoritmo Traveling Salesman
        graph2 = [[0,80,99,1000,1000,1000],[88,0,1000,146,97,1000],[99,1000,0,0,1000,211],[1000,146,0,0,138,1000],[1000,97,1000,138,0,101],[1000,1000,211,1000,101,0]]
        s = 0
        tiempo_inicio = time.time()
        res = travelling_salesman_function(graph2,s)
        print("La ruta con menos costo es: " + str(res))
        tiempo_fin = time.time()
        tiempo_total_Traveling_Salesman = (tiempo_fin - tiempo_inicio) * 1000
        print("La función tardó", tiempo_total_Traveling_Salesman, "segundos en ejecutarse")
        print("----------------------------------------------------------------------------")

        #Ejecución del algoritmo para generar la imagen del grafo
        print(G.nodes())
        print(G.edges())
        print_graph(G)
        print("----------------------------------------------------------------------------")
        
    elif opcion == "9":
        #Ejecución del algoritmo Genetic Algorithm
        
        population_size = int(input("Ingrese el tamaño de la población: "))
        num_generations = int(input("Ingrese el número de generaciones: "))
        mutation_rate = float(input("Ingrese la taza de mutación que quiere que tenga su población(La taza de mutació puede estar entre el 0 y 1): "))

        tiempo_inicio = time.time()
        Resultado_genetic = genetic_algorithm(graph, population_size, num_generations, mutation_rate)
        tiempo_fin = time.time()
        tiempo_total_Genetic_Algorithm = (tiempo_fin - tiempo_inicio) * 1000
        print("El resultado de Genetic Algorithm")
        print(Resultado_genetic)
        print("La función tardó", tiempo_total_Genetic_Algorithm, "segundos en ejecutarse")
        print("----------------------------------------------------------------------------")
    elif opcion == "10":
        # Ejecutamos el algoritmo Greedy
        tiempo_inicio = time.time()
        path = greedy(graph, 'A', 'F')
        if path is not None:
            tiempo_fin = time.time()
            tiempo_total_Greedy = (tiempo_fin - tiempo_inicio) * 1000
            print("Resultado Greedy")
            print(f"El camino más corto desde '{start}' hasta '{goal}' es: {path}")
            print("La función tardó", tiempo_total_Greedy, "segundos en ejecutarse")
        else:
            tiempo_fin = time.time()
            tiempo_total_Greedy = (tiempo_fin - tiempo_inicio) * 1000
            print(f"No se pudo encontrar un camino válido desde '{start}' hasta '{goal}'.")
            print("La función tardó", tiempo_total_Greedy, "segundos en ejecutarse")
        print("----------------------------------------------------------------------------")

        # Ejecutamos el algoritmo A* con peso
        tiempo_inicio = time.time()
        path = weighted_astar(start, goal, heuristic, successors, edge_cost, w=1.5)
        tiempo_fin = time.time()
        tiempo_total_A_pesos = (tiempo_fin - tiempo_inicio)*1000
        print("Resultado weighted A*")
        print(path)
        print("La función tardó", tiempo_total_A_pesos, "segundos en ejecutarse")
        print("----------------------------------------------------------------------------")


        # Ejecutamos el algoritmo A*
        tiempo_inicio = time.time()
        came_from, cost_so_far = astar(start, goal, graph, heuristic)
        tiempo_fin = time.time()
        tiempo_total_A = (tiempo_fin - tiempo_inicio) * 1000
        # Mostramos el resultado
        if goal not in came_from:
            print(f"No se encontró un camino desde {start} hasta {goal}")
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
        tiempo_inicio = time.time()
        result = beam_search(start, lambda n: n == goal, expand_fn, beam_width, goal, heuristic)
        tiempo_fin = time.time()
        tiempo_total_Beam = (tiempo_fin - tiempo_inicio) * 1000
        print("Resultado Beam")
        print(result)
        print("La función tardó", tiempo_total_Beam, "segundos en ejecutarse")
        print("----------------------------------------------------------------------------")

        # Ejecutamos el algoritmo Branch and Bound
        tiempo_inicio = time.time()
        path, cost = branch_and_bound_shortest_path(graph, start, goal, heuristic)
        tiempo_fin = time.time()
        tiempo_total_Breanch_and_Bound = (tiempo_fin - tiempo_inicio) * 1000
        print("Resultado de Branch and Bound")
        print("Camino más corto:", path)
        print("Costo total:", cost)
        print("La función tardó", tiempo_total_Breanch_and_Bound, "segundos en ejecutarse")
        print("----------------------------------------------------------------------------")

        #Ejecución del aloritmo Steepest hil climbing
        tiempo_inicio = time.time()
        resultado = steepest_hill_climbing(graph, start)
        tiempo_fin = time.time()
        tiempo_total_Steepest_Hil_Climbing = (tiempo_fin - tiempo_inicio) * 1000
        print("El resultado de steepest hil climbing")
        print(resultado)
        print("La función tardó", tiempo_total_Steepest_Hil_Climbing, "segundos en ejecutarse")
        print("----------------------------------------------------------------------------")

        #Ejecución del algoritmo Stochastic hil clambing
        tiempo_inicio = time.time()
        resultado_stochastic = stochastic_hill_climbing(graph, start, heuristic)
        tiempo_fin = time.time()
        tiempo_total_Stochastic_hil_clambing = (tiempo_fin - tiempo_inicio) * 1000
        print("El resultado de Stochastic hil clambing")
        print(resultado_stochastic)
        print("La función tardó", tiempo_total_Stochastic_hil_clambing, "segundos en ejecutarse")
        print("----------------------------------------------------------------------------")

        #Ejecución del algoritmo Traveling Salesman
        graph2 = [[0,80,99,1000,1000,1000],[88,0,1000,146,97,1000],[99,1000,0,0,1000,211],[1000,146,0,0,138,1000],[1000,97,1000,138,0,101],[1000,1000,211,1000,101,0]]
        s = 0
        tiempo_inicio = time.time()
        res = travelling_salesman_function(graph2,s)
        print("La ruta con menos costo es: " + str(res))
        tiempo_fin = time.time()
        tiempo_total_Traveling_Salesman = (tiempo_fin - tiempo_inicio) * 1000
        print("La función tardó", tiempo_total_Traveling_Salesman, "segundos en ejecutarse")
        print("----------------------------------------------------------------------------")

        #Ejecución del algoritmo para generar la imagen del grafo
        print("----------------------------------------------------------------------------")
        print(G.nodes())
        print(G.edges())
        print_graph(G)

        #Ejecución del algoritmo Genetic Algorithm
        population_size = int(print("Ingrese el tamaño de la población: "))
        num_generations = int(print("Ingrese el número de generaciones: "))
        mutation_rate = float(print("Ingrese la taza de mutación que quiere que tenga su población: "))

        tiempo_inicio = time.time()
        Resultado_genetic = genetic_algorithm(graph, population_size, num_generations, mutation_rate)
        tiempo_fin = time.time()
        tiempo_total_Genetic_Algorithm = (tiempo_fin - tiempo_inicio) * 1000
        print("El resultado de Genetic Algorithm")
        print(Resultado_genetic)
        print("La función tardó", tiempo_total_Genetic_Algorithm, "segundos en ejecutarse")
        print("----------------------------------------------------------------------------")
       
    elif opcion == "11":
        print("Saliendo del programa...")
        break
    
    else:
        print("Opción inválida, por favor seleccione una opción del 1 al 11")
        
print("")
print("El timepo de ejecución que se muestra en las funciones es en milisegundos")
print("----------------------------------------------------------------------------")
print("")

menu()


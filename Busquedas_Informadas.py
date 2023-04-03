import queue
import heapq
from heapq import heappush, heappop
from math import inf, radians, cos, sin, asin, sqrt

def heuristic(node):
    # Coordenadas de los nodos
    latitudes = {'A': 40.4168, 'B': 40.4146, 'C': 40.4262,
                 'D': 40.4301, 'E': 40.4266, 'F': 40.4297}
    longitudes = {'A': -3.7038, 'B': -3.7015, 'C': -3.6764,
                  'D': -3.7007, 'E': -3.6974, 'F': -3.6984}
    # Distancia Haversine
    R = 6372.8  # Radio de la Tierra en kilómetros
    lat1, lon1 = radians(latitudes[node]), radians(longitudes[node])
    lat2, lon2 = radians(latitudes[goal]), radians(longitudes[goal])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * asin(sqrt(a))
    return R * c

# Función para Greedy best-first
def get_neighbors(node):
    return graph[node].keys()
# Función para Greedy best-first
def greedy(start, goal):
    queue2 = queue.PriorityQueue()
    queue2.put((heuristic(start), start))
    visited = set()
    while not queue2.empty():
        node = queue2.get()[1]

        if node == goal:
            # Si hemos llegado al nodo objetivo, podemos calcular el costo total del camino.
            return visited
        visited.add(node)

        for neighbor in get_neighbors(node):
            if neighbor not in visited:
                # Al agregar un vecino a la cola de prioridad, utilizamos la heurística para priorizar los nodos.
                queue2.put((heuristic(neighbor), neighbor))

    return None

#Función para Weighted A*
def edge_cost(edge):
    node1, node2 = edge
    return graph[node1][node2]
#Función para Weighted A*
def successors(node):
    return graph.get(node, {})

def weighted_astar(start, goal, heuristic, successors, edge_cost, w=1):
    """Weighted A* search algorithm"""
    frontier = [(heuristic(start), start, 0)]
    visited = {start: (None, 0)}
    while frontier:
        _, node, cost = heappop(frontier)
        if node == goal:
            path = [node]
            while node != start:
                node, _ = visited[node]
                path.append(node)
            return list(reversed(path))
        for successor, successor_cost in successors(node).items():
            new_cost = visited[node][1] + edge_cost((node, successor))
            if successor not in visited or new_cost < visited[successor][1]:
                visited[successor] = (node, new_cost)
                priority = new_cost + w * heuristic(successor)
                heappush(frontier, (priority, successor, new_cost))
    return []

def astar(start, goal, graph, heuristic) :
    """
    Implementación del algoritmo A* para encontrar el camino más corto desde un nodo inicial hasta un nodo objetivo
    en un grafo ponderado y dirigido.

    :param start: Nodo inicial.
    :param goal: Nodo objetivo.
    :param graph: Grafo ponderado y dirigido representado como un diccionario de diccionarios. El diccionario exterior
                  contiene los nodos del grafo como claves, y los valores son otros diccionarios que representan los
                  nodos adyacentes y sus respectivos costos. Por ejemplo: {'A': {'B': 6, 'C': 8}, 'B': {'D': 3}, ...}
    :param heuristics: Diccionario que contiene las heurísticas (en este caso, las distancias haversine) para cada nodo.
                       Las claves son los nodos del grafo y los valores son los valores de la heurística para cada nodo.
                       Por ejemplo: {'A': 2442.076702837977, 'B': 2504.893734691217, 'C': 3334.165797319547, ...}
    :return: Tupla que contiene dos diccionarios: el diccionario 'came_from', que contiene los nodos antecesores de cada
             nodo en el camino más corto desde el nodo inicial hasta el nodo objetivo, y el diccionario 'cost_so_far',
             que contiene el costo acumulado para llegar a cada nodo en el camino más corto desde el nodo inicial.
    """
    # Inicializamos los diccionarios que almacenarán los nodos antecesores y los costos acumulados
    # Creamos una cola de prioridad (heap) que almacenará los nodos por los que iremos pasando durante la búsqueda.
    # La cola de prioridad está implementada como una lista de tuplas, donde el primer elemento de la tupla es la
    # suma del costo acumulado y la heurística del nodo, y el segundo elemento es el propio nodo.
    # Los elementos de la cola de prioridad se ordenan según la suma del costo acumulado y la heurística, de manera
    # que los nodos con menor suma estén al principio de la cola.
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

def beam_search(start_state, goal_fn, expand_fn, beam_width, goal, heuristic):
    beam = [(0, start_state)]
    paths = {start_state: [start_state]}
    while True:
        next_beam = []
        for cost, state in beam:
            for child_state, child_cost in expand_fn(state):
                new_cost = cost + child_cost
                h = heuristic(child_state)
                f = new_cost + h
                if goal_fn(child_state):
                    path = paths[state] + [child_state]
                    return (new_cost, path)
                next_beam.append((f, child_state))
                paths[child_state] = paths[state] + [child_state]
        beam = heapq.nsmallest(beam_width, next_beam, key=lambda x: x[0])
        if not beam:
            return None
        print("Beam:", beam) # imprime el haz de búsqueda en cada iteración

#Función de beam
def expand_fn(state):
    return list(graph.get(state, {}).items())

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

# Ejecutamos el algoritmo Greedy
visited = greedy(start, goal)
if visited is not None:
    print("Resultado Greedy")
    print(f"El camino más corto desde '{start}' hasta '{goal}' es: {visited}")
else:
    print(f"No se pudo encontrar un camino válido desde '{start}' hasta '{goal}'.")

# Ejecutamos el algoritmo A* con peso
path = weighted_astar(start, goal, heuristic, successors, edge_cost, w=1.5)
print("Resultado weighted A*")
print(path)

# Ejecutamos el algoritmo A*
came_from, cost_so_far = astar(start, goal, graph, heuristic)

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
    #print(f"Costo total: {cost_so_far[goal_node]}")

# Ejecutamos el algoritmo Beam
result = beam_search(start, lambda n: n == goal, expand_fn, beam_width, goal, heuristic)
print("Resultado Beam")
print(result)
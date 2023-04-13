import queue
import heapq
from heapq import heappush, heappop
from math import inf, radians, cos, sin, asin, sqrt

 
#Para la heurística se utiliza la fórmula Haversine para calcular la distancia entre dos puntos en la superficie de la Tierra,
# teniendo en cuenta que la Tierra es esférica en lugar de plana.
# La función comienza definiendo dos diccionarios llamados "latitudes" y "longitudes" 
# que contienen las coordenadas de latitud y longitud de cada nodo en el grafo.
# A continuación, se define la constante R como el radio de la Tierra en kilómetros.
# La función luego utiliza la biblioteca matemática de Python para convertir las coordenadas de latitud y longitud del nodo y del nodo objetivo a radianes,
# que es la unidad requerida por la fórmula Haversine. Estos valores se asignan a las variables lat1, lon1 y lat2, lon2.
# Luego, se calcula la diferencia de latitud y longitud entre los dos puntos mediante la resta de las coordenadas 
# Finalmente, se calcula la distancia Haversine utilizando la fórmula matemática 
# y se devuelve el resultado multiplicado por el radio de la Tierra (R) en kilómetros. 
# La distancia calculada es una estimación de la distancia entre el nodo y el nodo objetivo.
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

#La función greedy(start, goal) realiza una búsqueda Greedy Best-First para encontrar el camino más corto desde el nodo "start" hasta el nodo "goal". 
#El algoritmo comienza en el nodo "start" y en cada paso expande el nodo con la heurística, tomando el valor más bajo
#priorizando los nodos que están más cerca del objetivo.
#La función crea una cola de prioridad "queue2" y agrega el nodo "start" con su heurística correspondiente (calculada por la función heuristic(node)). 
#El algoritmo también inicializa un arreglo "visited" que almacenará los nodos visitados. 
#Luego, el algoritmo itera mientras la cola de prioridad no esté vacía. 
#En cada iteración, el nodo con la heurística más baja se obtiene de la cola de prioridad. 
#Si este nodo es el objetivo, la función devuelve el conjunto de nodos visitados. 
#De lo contrario, el nodo actual se agrega al conjunto de nodos visitados.
#A continuación, se itera a través de los vecinos del nodo actual y se agregan a la cola de prioridad si aún no han sido visitados.
#En este paso, la heurística se utiliza para priorizar los vecinos en la cola de prioridad.
#Si el objetivo no se encuentra después de recorrer todos los nodos accesibles, la función devuelve None.
#En general, la función greedy(start, goal) implementa una búsqueda de camino más corto utilizando la heurística para guiar el proceso de búsqueda.
#El algoritmo es muy rápido en encontrar una solución, pero no siempre encuentra el camino más corto.
# Función para Greedy best-first
def greedy(start, goal): #Javier Vázquez Gurrola 
    # Inicializar una cola de prioridad y agregar el nodo de inicio con su heurística correspondiente.
    queue2 = queue.PriorityQueue()
    queue2.put((heuristic(start), start))
    # Inicializar un conjunto de nodos visitados.
    visited = set()
    # Mientras la cola de prioridad no esté vacía, seguir explorando nodos.
    while not queue2.empty():
        # Obtener el nodo con la heurística más baja de la cola de prioridad.
        node = queue2.get()[1]
        # Si el nodo actual es el objetivo, devolver el conjunto de nodos visitados.
        if node == goal:
            return visited
        # Agregar el nodo actual al conjunto de nodos visitados.
        visited.add(node)
        # Iterar a través de los vecinos del nodo actual.
        for neighbor in get_neighbors(node):
            # Si el vecino no ha sido visitado, agregarlo a la cola de prioridad con su heurística correspondiente.
            if neighbor not in visited:
                queue2.put((heuristic(neighbor), neighbor))
    # Si el objetivo no se encuentra, devolver None.
    return None

#La función edge_cost toma una tupla de dos nodos y devuelve el costo de la arista que los conecta en el grafo.
#En otras palabras, dado un borde, esta función devuelve la distancia o el costo de viajar de un nodo a otro.
#Esta función se utiliza más adelante en la función weighted_astar para calcular el nuevo costo cuando se expande un nodo.
#Función para Weighted A*
def edge_cost(edge): #Javier Vázquez Gurrola 
    node1, node2 = edge
    return graph[node1][node2]

#La función successors toma un nodo y devuelve un diccionario de sus sucesores en el grafo, junto con los pesos asociados.
#Los sucesores son nodos conectados al nodo actual por una arista. 
#En otras palabras, esta función devuelve un diccionario donde las claves son nodos vecinos y los valores son los costos 
#de las aristas que conectan esos nodos con el nodo actual.
#Función para Weighted A*
def successors(node): #Javier Vázquez Gurrola 
    return graph.get(node, {})

#La función weighted_astar es la implementación principal de Weighted A*. 
#Toma los siguientes argumentos:
#start: el nodo de inicio.
#goal: el nodo de destino.
#heuristic: una función que toma un nodo y devuelve una estimación heurística del costo de llegar desde ese nodo hasta el nodo objetivo.
#successors: una función que toma un nodo y devuelve un diccionario de sus sucesores en el grafo, junto con los costos asociados.
#edge_cost: una función que toma una tupla de dos nodos y devuelve el costo de la arista que los conecta en el grafo.
#w: el peso del algoritmo. Un valor mayor de w hace que el algoritmo sea más orientado por heurística, pero puede llevar a una búsqueda menos completa.
#En resumen, el algoritmo comienza con el nodo de inicio y lo coloca en una cola de prioridad (frontier) junto con el valor obtenido 
#por heurística para llegar al nodo objetivo. 
#Luego, mientras haya nodos en la cola de prioridad, el algoritmo selecciona el nodo con la estimación heurística más baja y lo expande.
#Para cada sucesor del nodo actual, se calcula el costo total de llegar a ese sucesor sumando el costo actual del nodo al peso de la arista 
#que conecta el nodo actual con el sucesor. 
#Si el costo total es menor que el costo actual conocido del sucesor, el costo del sucesor se actualiza y se agrega a la cola de prioridad. 
#Si el sucesor ya ha sido visitado, su costo se actualiza sólo si el nuevo costo es menor. 
#El proceso continúa hasta que se llega al nodo objetivo o hasta que la cola de prioridad está vacía. 
#Cuando se llega al nodo objetivo, se devuelve el camino que se tomó para llegar a él.
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


#Implementación del algoritmo A* para encontrar el camino más corto desde un nodo inicial hasta un nodo objetivo en un grafo ponderado y dirigido.
#Toma los siguientes parámetros
#start: Nodo inicial.
#goal: Nodo objetivo.
#graph: Grafo ponderado y dirigido representado como un diccionario de diccionarios.
#contiene los nodos del grafo como claves, y los valores son otros diccionarios que representan los nodos adyacentes y sus respectivos costos. 
#Por ejemplo: {'A': {'B': 6, 'C': 8}, 'B': {'D': 3}, ...}
#heuristics: Diccionario que contiene las heurísticas (en este caso, las distancias haversine) para cada nodo.
#Las claves son los nodos del grafo y los valores son los valores de la heurística para cada nodo.
#Por ejemplo: {'A': 2442.076702837977, 'B': 2504.893734691217, 'C': 3334.165797319547, ...}
#return: Tupla que contiene dos diccionarios: el diccionario 'came_from', que contiene los nodos antecesores 
#de cada nodo en el camino más corto desde el nodo inicial hasta el nodo objetivo, y el diccionario 'cost_so_far'
#que contiene el costo acumulado para llegar a cada nodo en el camino más corto desde el nodo inicial.
#El algoritmo comienza inicializando un conjunto de diccionarios que almacenarán los nodos antecesores y los costos acumulados
#y una cola de prioridad (heap) que almacenará los nodos por los que iremos pasando durante la búsqueda. 
#La cola de prioridad está implementada como una lista de tuplas, donde el primer elemento de la tupla es la suma del costo acumulado 
#y la heurística del nodo, y el segundo elemento es el propio nodo. 
#Los elementos de la cola de prioridad se ordenan según la suma del costo acumulado y la heurística
#de manera que los nodos con menor suma estén al principio de la cola.
#Luego, el algoritmo comienza a iterar mientras la cola de prioridad tenga elementos. 
#En cada iteración, se obtiene el nodo actual de la cola de prioridad, es decir, el nodo con menor suma de costo acumulado y heurística.
#Si hemos llegado al nodo objetivo, terminamos la búsqueda y devolvemos los diccionarios de nodos antecesores y costos acumulados.
#Si no hemos llegado al nodo objetivo, iteramos sobre los nodos adyacentes al nodo actual y calculamos el costo acumulado 
#para llegar a cada uno de ellos desde el nodo inicial, sumando el costo acumulado para llegar al nodo actual y el costo del arco 
#entre el nodo actual y el vecino. Si el vecino no está en el diccionario de costos acumulados o si hemos encontrado un camino más corto 
#para llegar al vecino, actualizamos el diccionario de costos acumulados y el diccionario de nodos antecesores. 
#Luego, añadimos el vecino a la cola de prioridad con una prioridad que es la suma del costo acumulado y la heurística del vecino.
#Finalmente, devolvemos los diccionarios de nodos antecesores y costos acumulados.
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

#La función expand_fn recibe un nodo y devuelve una lista de tuplas que representan los estados y los costos para llegar a esos estados. 
#Esta función es utilizada en la búsqueda Beam, para expandir los nodos.
#Expande un estado, es decir, devuelve todos los hijos del estado actual con sus respectivos costos.
#Función de beam
def expand_fn(state): #Javier Vázquez Gurrola
    return list(graph.get(state, {}).items())


# Función de búsqueda beam search
#La función beam_search implementa la búsqueda en haz (beam search) para encontrar el camino más corto desde 
#el estado inicial hasta el estado objetivo. Los parámetros de la función son:
#start_state: el estado inicial del problema.
#goal_fn: una función que determina si un estado dado es el estado objetivo o no.
#beam_width: el ancho del haz de búsqueda, es decir, la cantidad de estados que se mantienen en la frontera.
#goal: el nodo objetivo del problema.
#heuristic: una función heurística que estima el costo para llegar al estado objetivo desde cualquier estado dado.
#La función comienza creando un haz de búsqueda que contiene el estado inicial con costo 0. 
#Luego, se inicializa un diccionario llamado paths que almacenará los caminos hacia los estados de la frontera. 
#El bucle principal se ejecuta mientras el haz de búsqueda tenga elementos. 
#Dentro del bucle, se itera sobre los estados en el haz actual, y para cada estado, se expande su frontera utilizando la función expand_fn. 
#Se calcula el costo y la heurística para cada estado hijo, y si se encuentra un estado objetivo, se devuelve el costo y el camino para llegar a él.
#Si no se encuentra el estado objetivo, se agregan los hijos al siguiente haz de búsqueda y se actualiza el diccionario paths con los nuevos caminos. 
#Finalmente, se actualiza el haz de búsqueda seleccionando los beam_width estados con menor costo y se continúa con la siguiente iteración. 
#Si no quedan elementos en el haz de búsqueda, la función devuelve None.
#La función también imprime el haz de búsqueda en cada iteración, lo que puede ser útil para entender cómo se va expandiendo durante la búsqueda.
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


#Parametros de la función:
#graph: diccionario de diccionarios que representa el grafo
#start: nodo de inicio
#end: nodo de destino
#heuristic_func: función heurística que estima el costo restante desde un nodo hasta el nodo de destino
#El código implementa el algoritmo de búsqueda Branch and Bound para encontrar el camino más corto desde un nodo de inicio hasta un nodo de destino 
#en un grafo. El algoritmo utiliza una función heurística para estimar el costo restante desde un nodo hasta el nodo de destino.
#En la primera parte del código, se definen la cola de prioridad (heap) y el diccionario de costos mínimos. 
#La cola de prioridad se utiliza para ordenar los nodos adyacentes según su costo total más una estimación heurística del costo restante 
#hasta el nodo destino. El diccionario de costos mínimos se utiliza para llevar un registro de los costos mínimos 
#conocidos para llegar a cada nodo del grafo.
#La cola de prioridad se inicializa con una tupla que contiene el costo total estimado desde el nodo de inicio hasta el nodo de destino, 
#el nodo de inicio, el camino recorrido hasta el momento (que inicialmente solo contiene el nodo de inicio) 
#y el costo total acumulado hasta el momento (que inicialmente es cero). 
#El costo total estimado se calcula sumando el costo actual y la estimación heurística del costo restante hasta el nodo de destino.
#El diccionario de costos mínimos se inicializa con un valor infinito para cada nodo del grafo, 
#excepto para el nodo de inicio, que se inicializa con cero más la estimación heurística del costo restante hasta el nodo de destino.
#Luego, se inicia un bucle que recorre la cola de prioridad hasta encontrar el camino más corto desde el nodo de inicio hasta el nodo de destino. 
#En cada iteración del bucle, se extrae el nodo con el costo total más bajo de la cola de prioridad y se exploran sus nodos adyacentes.
#Si el nodo extraído es el nodo de destino, se devuelve el camino recorrido hasta el momento y el costo total acumulado.
#Si el costo actual es mayor que el costo mínimo conocido para el nodo, se ignora el nodo actual y se continúa con el siguiente.
#Si el costo actual es menor que el costo mínimo conocido para el nodo adyacente, se actualiza el costo mínimo conocido y se agrega el nodo adyacente 
#a la cola de prioridad con el nuevo costo total estimado, el nuevo camino y el nuevo costo total acumulado.
#Si no se encuentra un camino desde el nodo de inicio hasta el nodo de destino, se devuelve None.
#En resumen, este código implementa el algoritmo de búsqueda Branch and Bound para encontrar el camino más corto desde 
#un nodo de inicio hasta un nodo de destino en un grafo, utilizando una función heurística para estimar el costo restante hasta el nodo de destino.
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

# Ejecutamos el algoritmo Branch and Bound
path, cost = branch_and_bound_shortest_path(graph, start, goal, heuristic)
print("Resultado de Branch and Bound")
print("Camino más corto:", path)
print("Costo total:", cost)
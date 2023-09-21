# Búsquedas_Informadas

El programa es un conjunto de funciones relacionadas con algoritmos de búsqueda y cálculo de heurísticas para la resolución de problemas en grafos. A continuación, se describe lo que el programa busca hacer, su importancia y posibles aplicaciones:

Cálculo de Distancia Haversine: La función heuristic(node) calcula la distancia Haversine entre dos coordenadas geográficas (en este caso, nodos del grafo). Esto es útil para estimar la distancia entre dos puntos en la superficie de la Tierra, lo que puede ser utilizado como heurística en algoritmos de búsqueda como A*.

Algoritmo Greedy Best-First Search: La función greedy(graph, start, goal) implementa el algoritmo Greedy Best-First Search para encontrar una ruta desde un nodo de inicio hasta un nodo objetivo en un grafo ponderado. Este algoritmo se utiliza para búsquedas no informadas y prioriza la expansión de los nodos que están más cerca del objetivo según la heurística definida.

Algoritmo Weighted A*: Las funciones edge_cost, successors, y weighted_astar implementan el algoritmo Weighted A* para encontrar el camino más corto en un grafo ponderado, teniendo en cuenta una heurística y un factor de peso (w) para equilibrar el costo real y la heurística. Este algoritmo es ampliamente utilizado en la planificación de rutas y navegación.

Importancia y posibles aplicaciones:

Estos algoritmos son fundamentales en campos como la inteligencia artificial, la planificación de rutas en sistemas de navegación, la robótica, la logística y la optimización de recursos. Permiten encontrar soluciones eficientes en grafos de gran escala.

La función de cálculo de distancia Haversine es crucial en aplicaciones de geolocalización y mapas en línea, ya que permite estimar distancias en la superficie de la Tierra de manera precisa.

Los algoritmos de búsqueda como A* y sus variantes son esenciales en la planificación de rutas para vehículos autónomos, robots móviles y sistemas de logística, ayudando a encontrar la mejor ruta para alcanzar un objetivo.

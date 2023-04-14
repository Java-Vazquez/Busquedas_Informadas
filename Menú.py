# Función para mostrar el menú y obtener la opción elegida por el usuario
def mostrar_menu():
    print("Menú:")
    print("1. Greedy best-first")
    print("2. A* con peso")
    print("3. A*")
    print("4. Beam")
    print("5. Branch and Bound")
    opcion = input("Ingrese el número de la opción que desea: ")
    return opcion
# variable para controlar el bucle del menú
opcion = ""

# bucle que muestra el menú y procesa la opción elegida por el usuario
while opcion != "6":
    opcion = mostrar_menu()
    if opcion == "1":
        start = input("Ingrese el nodo origen:")
        goal = input("Ingrese el nodo destino:")
        # Ejecutamos el algoritmo Greedy
        visited = greedy(start, goal)
        if visited is not None:
            print("Resultado Greedy")
            print(f"El camino más corto desde '{start}' hasta '{goal}' es: {visited}")
        else:
            print(f"No se pudo encontrar un camino válido desde '{start}' hasta '{goal}'.")

    elif opcion == "2":
       start = input("Ingrese el nodo origen:")
       goal = input("Ingrese el nodo destino:")
       # Ejecutamos el algoritmo A* con peso
       path = weighted_astar(start, goal, heuristic, successors, edge_cost, w=1.5)
       print("Resultado weighted A*")
       print(path)

    elif opcion == "3":
       start = input("Ingrese el nodo origen:")
       goal = input("Ingrese el nodo destino:")
       #Ejecutamos el algoritmo A*
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
                print(f"Costo total: {cost_so_far[goal]}")

    elif opcion == "4":
        start = input("Ingrese el nodo origen:")
        goal = input("Ingrese el nodo destino:")
        beam_width = str(input("Ingrese el beam width:"))
        # Ejecutamos el algoritmo Beam
        result = beam_search(start, lambda n: n == goal, expand_fn, beam_width, goal, heuristic)
        print("Resultado Beam")
        print(result)

    elif opcion == "5":
        start = input("Ingrese el nodo origen:")
        goal = input("Ingrese el nodo destino:")
        # Ejecutamos el algoritmo Branch and Bound
        path, cost = branch_and_bound_shortest_path(graph, start, goal, heuristic)
        print("Resultado de Branch and Bound")
        print("Camino más corto:", path)
        print("Costo total:", cost)

    else:
        print("Opción no válida. Por favor, elija una opción del menú.")


import heapq
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
import math

FREE = 255
UNKNOWN = 128
OCCUPIED = 0

# Vizinhança-8: (d_linha, d_coluna, custo)
NEIGHBORS = [
    (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
    (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)),
    (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2)),
]


class AStarPathfinder:
    ROBOT_RADIUS = 0.105
    SAFETY_MARGIN = 0.05
    MAP_RESOLUTION = 0.05  # metros por pixel

    COLLISION_RADIUS = ROBOT_RADIUS / MAP_RESOLUTION + 0.5
    # Abaixo deste raio o custo é alto, mas a passagem ainda é permitida
    SAFETY_RADIUS = (ROBOT_RADIUS + SAFETY_MARGIN) / MAP_RESOLUTION + 0.5
    SAFETY_PENALTY = 1000.0
    START_UNKNOWN_TOLERANCE = 5

    def __init__(self, map_array: np.array, start: tuple, goal: tuple, wall_influence=5.0, buffer_factor=2.0):
        """
        start/goal em (linha, coluna). wall_influence é o peso da proximidade
        das paredes e buffer_factor o alcance dessa influência.
        """
        self.start = start
        self.goal = goal
        self.wall_influence = wall_influence
        self.buffer_factor = buffer_factor
        self.GOAL_REACHEABLE = False

        self.map = map_array.copy()
        self.map_array = self.preprocess_map(map_array)
        self.potential_field = self.create_potential_field()

        self.target = tuple(int(v) for v in goal)
        self.allow_goal_approach()


    def preprocess_map(self, map_array: np.array) -> np.array:
        """Converte valores intermediários em obstáculo."""
        processed = map_array.copy()
        processed[(processed != FREE) & (processed != UNKNOWN)] = OCCUPIED
        return processed

    def create_potential_field(self) -> np.array:
        """Gera o custo extra de cada célula pela proximidade de obstáculos."""
        self.obstacle_distance = distance_transform_edt(self.map_array != OCCUPIED)

        # Células onde o robô encostaria num obstáculo
        self.blocked = (self.map_array == OCCUPIED) | (self.obstacle_distance < self.COLLISION_RADIUS)

        # Alto rente à parede, quase nulo no meio do corredor
        potential = self.wall_influence * np.exp(-self.obstacle_distance / self.buffer_factor)

        potential[self.obstacle_distance <= self.SAFETY_RADIUS] += self.SAFETY_PENALTY
        return potential

    def allow_goal_approach(self):
        """
        Desbloqueia a vizinhança do objetivo quando ele está rente à parede,
        senão o A* nunca o alcançaria. Só libera células com folga >= à do
        objetivo, e elas mantêm custo alto.
        """
        r, c = self.target
        rows, cols = self.map_array.shape
        if not (0 <= r < rows and 0 <= c < cols) or self.map_array[r, c] == OCCUPIED:
            return  # objetivo fora do mapa ou em cima de parede

        radius = self.COLLISION_RADIUS + 1
        rr, cc = np.ogrid[:rows, :cols]
        near_goal = (rr - r) ** 2 + (cc - c) ** 2 <= radius ** 2
        min_clearance = min(self.obstacle_distance[r, c], self.COLLISION_RADIUS)
        self.blocked[near_goal & (self.obstacle_distance >= min_clearance)] = False

    def heuristic(self, a: tuple, b: tuple) -> float:
        """Distância euclidiana (admissível, pois todo passo custa >= seu comprimento)."""
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def is_valid(self, node: tuple) -> bool:
        """Célula dentro do mapa e onde o robô cabe."""
        r, c = node
        rows, cols = self.map_array.shape
        return 0 <= r < rows and 0 <= c < cols and not self.blocked[r, c]

    def nearest_free(self, node: tuple) -> tuple:
        """Célula livre mais próxima, para quando a localização cai sobre/rente a uma parede."""
        rows, cols = self.map_array.shape
        r = min(max(int(node[0]), 0), rows - 1)
        c = min(max(int(node[1]), 0), cols - 1)
        if self.is_valid((r, c)):
            return (r, c)

        # Prefere célula livre conhecida; senão, qualquer não bloqueada
        for passable in ((self.map_array == FREE) & ~self.blocked, ~self.blocked):
            if passable.any():
                _, (idx_r, idx_c) = distance_transform_edt(~passable, return_indices=True)
                return (int(idx_r[r, c]), int(idx_c[r, c]))
        return (r, c)

    def walking_distance_to_goal(self) -> dict:
        """
        Dijkstra a partir do objetivo, sem atravessar paredes. Usado quando o
        objetivo é inalcançável, para não parar do outro lado de um painel.
        """
        rows, cols = self.map_array.shape
        seed = (min(max(self.target[0], 0), rows - 1), min(max(self.target[1], 0), cols - 1))
        seed_cost = math.hypot(seed[0] - self.target[0], seed[1] - self.target[1])
        if self.map_array[seed] == OCCUPIED:
            # Objetivo na parede: parte da célula não-parede mais próxima
            _, (idx_r, idx_c) = distance_transform_edt(self.map_array == OCCUPIED, return_indices=True)
            nearest = (int(idx_r[seed]), int(idx_c[seed]))
            seed_cost += math.hypot(nearest[0] - seed[0], nearest[1] - seed[1])
            seed = nearest

        dist = {seed: seed_cost}
        heap = [(seed_cost, seed)]
        while heap:
            d, node = heapq.heappop(heap)
            if d > dist[node]:
                continue
            for dr, dc, step_cost in NEIGHBORS:
                nb = (node[0] + dr, node[1] + dc)
                if (0 <= nb[0] < rows and 0 <= nb[1] < cols and self.map_array[nb] != OCCUPIED
                        and d + step_cost < dist.get(nb, math.inf)):
                    dist[nb] = d + step_cost
                    heapq.heappush(heap, (d + step_cost, nb))
        return dist

    def find_path(self):
        """Executa o A*. Retorna (came_from, nó final) ou (None, None)."""
        start = self.nearest_free(self.start)
        goal = self.target

        if start != tuple(int(v) for v in self.start):
            print(f"Início {self.start} fora do mapa ou rente a obstáculo, usando célula livre mais próxima {start}")
        if not self.is_valid(start):
            print("Ponto inicial inválido (mapa sem células livres)")
            return None, None

        # Heap ordenado por f = g + h; o contador desempata por ordem de inserção
        counter = 0
        open_heap = [(self.heuristic(start, goal), counter, start)]
        came_from = {}
        g_score = {start: 0.0}
        closed = set()

        while open_heap:
            _, _, current = heapq.heappop(open_heap)
            if current in closed:
                continue  # entrada obsoleta

            if current == goal:
                self.GOAL_REACHEABLE = True
                return came_from, current

            closed.add(current)

            for dr, dc, step_cost in NEIGHBORS:
                neighbor = (current[0] + dr, current[1] + dc)
                if neighbor in closed or not self.is_valid(neighbor):
                    continue
                # Não corta quina na diagonal
                if dr and dc and (not self.is_valid((current[0] + dr, current[1]))
                                  or not self.is_valid((current[0], current[1] + dc))):
                    continue

                tentative_g = g_score[current] + step_cost + self.potential_field[neighbor]
                if tentative_g < g_score.get(neighbor, math.inf):
                    g_score[neighbor] = tentative_g
                    came_from[neighbor] = current
                    counter += 1
                    heapq.heappush(open_heap,
                                   (tentative_g + self.heuristic(neighbor, goal), counter, neighbor))

        # Objetivo inalcançável: para no ponto explorado mais próximo dele (a pé).
        # O objetivo não é trocado; o próximo replanejamento tenta de novo.
        walking = self.walking_distance_to_goal()
        best_node = min(closed, key=lambda n: (walking.get(n, math.inf), self.heuristic(n, goal)))
        if best_node != start:
            print(f"Objetivo {self.goal} inalcançável agora, parando no ponto mais próximo {best_node}")
            return came_from, best_node

        print("Caminho não encontrado")
        return None, None

    def reconstruct_path(self, came_from: dict, current: tuple) -> list:
        """Reconstrói o caminho do início até current."""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def know_path(self, path: list) -> list:
        """Corta o caminho para o robô andar só no conhecido."""
        # Corta na primeira célula desconhecida, tolerando as próximas ao início
        start = path[0]
        near_start = lambda p: math.hypot(p[0] - start[0], p[1] - start[1]) <= self.START_UNKNOWN_TOLERANCE

        known = path
        for i, cell in enumerate(path):
            if self.map_array[cell] == UNKNOWN and not near_start(cell):
                known = path[:max(i, 1)]
                break

        # A borda do desconhecido pode esconder uma parede: recua o fim até
        # ficar a pelo menos SAFETY_RADIUS do desconhecido
        if known[-1] != self.target:
            unknown_distance = distance_transform_edt(self.map_array != UNKNOWN)
            while (len(known) > 1 and not near_start(known[-1])
                   and unknown_distance[known[-1]] < self.SAFETY_RADIUS):
                known = known[:-1]
        return known

    def simplify_path(self, path: list) -> list:
        """Reduz o caminho a poucos waypoints."""
        if len(path) < 3:
            return list(path)

        # 1) Mantém só os pontos onde a direção muda
        corners = [path[0]]
        prev_dir = (path[1][0] - path[0][0], path[1][1] - path[0][1])
        for i in range(1, len(path) - 1):
            direction = (path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
            if direction != prev_dir:
                corners.append(path[i])
            prev_dir = direction
        corners.append(path[-1])

        # 2) Remove as "escadinhas": liga cada ponto ao mais distante alcançável
        # em linha reta sem perder folga em relação ao caminho do A*
        index_of = {p: i for i, p in enumerate(path)}
        path_clearance = np.array([self.obstacle_distance[p] for p in path])
        simplified = [corners[0]]
        i = 0
        while i < len(corners) - 1:
            j = len(corners) - 1
            while j > i + 1 and not self.shortcut_ok(path, path_clearance,
                                                     index_of[corners[i]], index_of[corners[j]]):
                j -= 1
            simplified.append(corners[j])
            i = j
        return simplified

    def shortcut_ok(self, path: list, path_clearance: np.array, a: int, b: int) -> bool:
        """
        A reta path[a] -> path[b] mantém, ponto a ponto, a folga local do
        caminho do A*? (Comparar com a folga mínima do trecho todo deixaria o
        atalho raspar em pontas de painel.)
        """
        p, q = path[a], path[b]
        n = int(math.ceil(max(abs(q[0] - p[0]), abs(q[1] - p[1])) * 2)) + 1
        window = max(3, (b - a) // 4)
        for t in np.linspace(0, 1, n):
            cell = (int(round(p[0] + t * (q[0] - p[0]))), int(round(p[1] + t * (q[1] - p[1]))))
            k = a + int(round(t * (b - a)))
            local = path_clearance[max(a, k - window):min(b, k + window) + 1].min()
            if self.obstacle_distance[cell] < local:
                return False
        return True

    def segment_clearance(self, a: tuple, b: tuple) -> float:
        """Menor distância a obstáculo ao longo do segmento a -> b."""
        n = int(math.ceil(max(abs(b[0] - a[0]), abs(b[1] - a[1])) * 2)) + 1
        rows = np.rint(np.linspace(a[0], b[0], n)).astype(int)
        cols = np.rint(np.linspace(a[1], b[1], n)).astype(int)
        return float(self.obstacle_distance[rows, cols].min())

    def plot_path(self, path: list, simplified_path: list):
        """Mostra o mapa com o caminho completo e o simplificado."""
        simplified_path = self.simplify_path(path)

        plt.figure(figsize=(10, 10))
        plt.imshow(self.map, cmap='gray')
        plt.scatter(self.start[1], self.start[0], color='green', s=100, label='Início')
        plt.scatter(self.goal[1], self.goal[0], color='blue', s=100, label='Objetivo')

        if path:
            path_x, path_y = zip(*path)
            plt.plot(path_y, path_x, color='magenta', linewidth=1, label='Caminho Completo')
            simp_x, simp_y = zip(*simplified_path)
            plt.plot(simp_y, simp_x, color='red', linewidth=2, linestyle='--', label='Caminho Simplificado')
        else:
            plt.title("Caminho não encontrado")

        plt.legend()
        plt.axis('equal')
        plt.show()

    def run(self, show_path=False):
        """
        Ponto de entrada do navegador: busca, reconstrói, corta e simplifica o
        caminho. Retorna os waypoints ou None. show_path fica desligado por
        padrão porque plt.show() bloqueia o robô.
        """
        print("Iniciando busca pelo caminho...")
        came_from, final_node = self.find_path()

        if final_node:
            print("Reconstruindo caminho...")
            path = self.reconstruct_path(came_from, final_node)

            print("Robo não anda no disconhecido")
            path = self.know_path(path)

            print("Caminho encontrado, simplificando...")
            simplified_path = self.simplify_path(path)

            print("Plotando o caminho...")
            if show_path:
                self.plot_path(path, simplified_path)

            return simplified_path
        else:
            print("Nenhum caminho pôde ser encontrado.")
            return None


def prep_map(map_path: str) -> np.array:
    """Carrega o .pgm e converte para FREE/UNKNOWN/OCCUPIED."""
    map_array = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
    map_array[map_array == 0] = 0
    map_array[map_array == 205] = 128
    map_array[map_array == 254] = 255
    map_array[(map_array >= 60) & (map_array != 128) & (map_array != 255)] = 0
    map_array = map_array.astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    map_array = cv2.morphologyEx(map_array, cv2.MORPH_OPEN, kernel)
    map_array = np.flipud(map_array)
    map_array = np.pad(map_array, ((0, 200), (0, 200)), 'constant', constant_values=128)
    return map_array


def main():
    map_array = prep_map('map5.pgm')
    astar = AStarPathfinder(map_array, (60, 20), (60, 120), wall_influence=10.0, buffer_factor=3.0)
    astar.run(show_path=True)


if __name__ == '__main__':
    main()

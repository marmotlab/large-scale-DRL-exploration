import numpy as np
from sklearn.neighbors import NearestNeighbors
import copy

from graph import Graph


class Ground_truth_graph():
    def __init__(self, map_size, k_size, ground_truth, plot=False):
        self.k_size = k_size
        self.graph = Graph()
        self.map_x = map_size[1]
        self.map_y = map_size[0]
        self.plot = plot
        if self.plot:
            self.x = []
            self.y = []

        self.utility = None
        self.explored_signal = None

        self.uniform_points = self.generate_uniform_points()
        self.ground_truth = ground_truth
        ground_truth_free_area = self.free_area(self.ground_truth)
        ground_truth_free_area_to_check = ground_truth_free_area[:, 0] + ground_truth_free_area[:, 1] * 1j
        uniform_points_to_check = self.uniform_points[:, 0] + self.uniform_points[:, 1] * 1j
        _, _, candidate_indices = np.intersect1d(ground_truth_free_area_to_check, uniform_points_to_check, return_indices=True)
        self.ground_truth_node_coords = self.uniform_points[candidate_indices]

    def generate_graph(self, robot_belief, node_coords_in_planning_graph, utility_in_planning_graph):
        self.find_k_neighbor_all_nodes(self.ground_truth_node_coords, self.ground_truth)

        free_area = self.free_area(robot_belief)
        free_area_to_check = free_area[:, 0] + free_area[:, 1] * 1j
        ground_truth_node_coords_to_check = self.ground_truth_node_coords[:, 0] + self.ground_truth_node_coords[:, 1] * 1j
        _, _, candidate_indices = np.intersect1d(free_area_to_check, ground_truth_node_coords_to_check, return_indices=True)

        self.explored_signal = np.ones((self.ground_truth_node_coords.shape[0], 1))
        for index in candidate_indices:
            self.explored_signal[index] = 0

        self.utility = np.ones((self.ground_truth_node_coords.shape[0], 1))
        for node, utility in zip(node_coords_in_planning_graph, utility_in_planning_graph):
            index = np.argmin(np.linalg.norm(node - self.ground_truth_node_coords, axis=1))
            self.utility[index] = utility

        return self.ground_truth_node_coords, self.graph.edges, self.utility, self.explored_signal

    def update_graph(self, robot_belief, node_coords_in_planning_graph, utility_in_planning_graph):
        free_area = self.free_area(robot_belief)
        free_area_to_check = free_area[:, 0] + free_area[:, 1] * 1j
        ground_truth_node_coords_to_check = self.ground_truth_node_coords[:, 0] + self.ground_truth_node_coords[:,
                                                                                  1] * 1j
        _, _, candidate_indices = np.intersect1d(free_area_to_check, ground_truth_node_coords_to_check,
                                                 return_indices=True)
        self.explored_signal = np.ones((self.ground_truth_node_coords.shape[0], 1))
        for index in candidate_indices:
            self.explored_signal[index] = 0

        self.utility = -np.ones((self.ground_truth_node_coords.shape[0], 1))
        for node, utility in zip(node_coords_in_planning_graph, utility_in_planning_graph):
            index = np.argmin(np.linalg.norm(node - self.ground_truth_node_coords, axis=1))
            self.utility[index] = utility

        return self.ground_truth_node_coords, self.graph.edges, self.utility, self.explored_signal

    def unique_coords(self, coords):
        x = coords[:, 0] + coords[:, 1] * 1j
        indices = np.unique(x, return_index=True)[1]
        coords = np.array([coords[idx] for idx in sorted(indices)])
        return coords

    def find_k_neighbor_all_nodes(self, node_coords, robot_belief):
        X = node_coords
        if len(node_coords) >= self.k_size:
            knn = NearestNeighbors(n_neighbors=self.k_size)
        else:
            knn = NearestNeighbors(n_neighbors=len(node_coords))
        knn.fit(X)
        distances, indices = knn.kneighbors(X)

        for i, p in enumerate(X):
            for j, neighbour in enumerate(X[indices[i][:]]):
                start = p
                end = neighbour
                if not self.check_collision(start, end, robot_belief):
                    a = str(self.find_index_from_coords(node_coords, p))
                    b = str(self.find_index_from_coords(node_coords, neighbour))
                    self.graph.add_node(a)
                    self.graph.add_edge(a, b, distances[i, j])
                    
                    if self.plot:
                        self.x.append([p[0], neighbour[0]])
                        self.y.append([p[1], neighbour[1]])
                    

    def check_collision(self, start, end, robot_belief):
        # Bresenham line algorithm checking
        collision = False

        x0 = start[0].round()
        y0 = start[1].round()
        x1 = end[0].round()
        y1 = end[1].round()
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        x, y = x0, y0
        error = dx - dy
        x_inc = 1 if x1 > x0 else -1
        y_inc = 1 if y1 > y0 else -1
        dx *= 2
        dy *= 2

        while 0 <= x < robot_belief.shape[1] and 0 <= y < robot_belief.shape[0]:
            k = robot_belief.item(int(y), int(x))
            if x == x1 and y == y1:
                break
            if k == 1:
                collision = True
                break
            if k == 127:
                collision = True
                break
            if error > 0:
                x += x_inc
                error -= dy
            else:
                y += y_inc
                error += dx
        return collision

    def find_index_from_coords(self, node_coords, p):
        return np.where(np.linalg.norm(node_coords - p, axis=1) < 1e-5)[0][0]

    def generate_uniform_points(self):
        x = np.linspace(0, self.map_x - 1, 30).round().astype(int)
        y = np.linspace(0, self.map_y - 1, 30).round().astype(int)
        t1, t2 = np.meshgrid(x, y)
        points = np.vstack([t1.T.ravel(), t2.T.ravel()]).T
        return points

    def free_area(self, robot_belief):
        index = np.where(robot_belief == 255)
        free = np.asarray([index[1], index[0]]).T
        return free


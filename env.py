
from skimage import io
import matplotlib.pyplot as plt
import os
from skimage.measure import block_reduce

from sensor import *
from graph_generator import *
from node import *
from ground_truth_graph import *


class Env():
    def __init__(self, map_index, k_size=20, plot=False, test=False):
        # import environment ground truth from dungeon files
        self.test = test
        if self.test:
            self.map_dir = f'DungeonMaps/test'
        else:
            self.map_dir = f'DungeonMaps/train'
        self.map_list = os.listdir(self.map_dir)
        self.map_list.sort(reverse=True)
        self.map_index = map_index % np.size(self.map_list)
        self.ground_truth, self.start_position = self.import_ground_truth(
            self.map_dir + '/' + self.map_list[self.map_index])
        self.ground_truth_size = np.shape(self.ground_truth)  # (480, 640)

        # initialize robot_belief
        self.robot_belief = np.ones(self.ground_truth_size) * 127  # unexplored 127
        self.downsampled_belief = None
        self.old_robot_belief = copy.deepcopy(self.robot_belief)

        # initialize parameters
        self.resolution = 4
        self.sensor_range = 80
        self.explored_rate = 0

        # initialize graph generator
        self.graph_generator = Graph_generator(map_size=self.ground_truth_size, sensor_range=self.sensor_range, k_size=k_size, plot=plot)
        self.graph_generator.route_node.append(self.start_position)
        self.node_coords, self.graph, self.node_utility, self.guidepost = None, None, None, None
        self.frontiers = None

        # initialize ground truth graph
        self.ground_truth_graph_generator = Ground_truth_graph(map_size=self.ground_truth_size, k_size=k_size, ground_truth=self.ground_truth, plot=plot)
        self.ground_truth_node_coords, self.ground_truth_graph, self.ground_truth_utility, self.ground_truth_explored_signal = None, None, None, None

        self.begin()

        # plot related
        self.plot = plot
        self.frame_files = []
        if self.plot:
            # initialize the route
            self.xPoints = [self.start_position[0]]
            self.yPoints = [self.start_position[1]]

    def find_index_from_coords(self, position):
        index = np.argmin(np.linalg.norm(self.node_coords - position, axis=1))
        return index

    def find_index_from_ground_truth_coords(self, position):
        index = np.argmin(np.linalg.norm(self.ground_truth_node_coords - position, axis=1))
        return index

    def begin(self):
        self.robot_belief = self.update_robot_belief(self.start_position, self.sensor_range, self.robot_belief,
                                                     self.ground_truth)\

        # downsampled belief has lower resolution than robot belief
        self.downsampled_belief = block_reduce(self.robot_belief.copy(), block_size=(self.resolution, self.resolution),
                                               func=np.min)
        self.frontiers = self.find_frontier()
        self.old_robot_belief = copy.deepcopy(self.robot_belief)

        self.node_coords, self.graph, self.node_utility, self.guidepost = self.graph_generator.generate_graph(
            self.start_position, self.robot_belief, self.frontiers)

        self.ground_truth_node_coords, self.ground_truth_graph, self.ground_truth_utility, self.ground_truth_explored_signal = self.ground_truth_graph_generator.generate_graph(self.robot_belief, self.node_coords, self.node_utility)

    def step(self, robot_position, next_position, travel_dist):
        # move the robot to the selected position and update its belief
        dist = np.linalg.norm(robot_position - next_position)
        travel_dist += dist

        robot_position = next_position
        self.graph_generator.route_node.append(robot_position)
        next_node_index = self.find_index_from_coords(robot_position)
        self.graph_generator.nodes_list[next_node_index].set_visited()
        self.robot_belief = self.update_robot_belief(robot_position, self.sensor_range, self.robot_belief,
                                                     self.ground_truth)
        self.downsampled_belief = block_reduce(self.robot_belief.copy(), block_size=(self.resolution, self.resolution),
                                               func=np.min)

        frontiers = self.find_frontier()
        self.explored_rate = self.evaluate_exploration_rate()

        # calculate the reward associated with this action
        reward = self.calculate_reward(dist, frontiers)

        if self.plot:
            self.xPoints.append(robot_position[0])
            self.yPoints.append(robot_position[1])

        # update the graph
        self.node_coords, self.graph, self.node_utility, self.guidepost = self.graph_generator.update_graph(
            robot_position, self.robot_belief, self.old_robot_belief, frontiers, self.frontiers)
        self.old_robot_belief = copy.deepcopy(self.robot_belief)

        self.frontiers = frontiers

        # update the ground truth graph
        self.ground_truth_node_coords, self.ground_truth_graph, self.ground_truth_utility, self.ground_truth_explored_signal = self.ground_truth_graph_generator.update_graph(self.robot_belief, self.node_coords, self.node_utility)

        # check if done
        done = self.check_done()
        if done:
            reward += 20

        return reward, done, robot_position, travel_dist

    def import_ground_truth(self, map_index):
        # occupied 1, free 255, unexplored 127
        ground_truth = (io.imread(map_index, 1) * 255).astype(int)
        robot_location = np.nonzero(ground_truth == 208)
        robot_location = np.array([np.array(robot_location)[1, 127], np.array(robot_location)[0, 127]])
        ground_truth = (ground_truth > 150)
        ground_truth = ground_truth * 254 + 1
        return ground_truth, robot_location

    def free_cells(self):
        index = np.where(self.ground_truth == 255)
        free = np.asarray([index[1], index[0]]).T
        return free

    def update_robot_belief(self, robot_position, sensor_range, robot_belief, ground_truth):
        robot_belief = sensor_work(robot_position, sensor_range, robot_belief, ground_truth)
        return robot_belief

    def check_done(self):
        done = False
        if self.test:
            if np.sum(self.ground_truth == 255) - np.sum(self.robot_belief == 255) <= 250:
                done = True
        elif np.sum(self.node_utility) == 0:
            done = True
        return done

    def calculate_reward(self, dist, frontiers):
        reward = 0
        reward -= dist / 64

        # check the num of observed frontiers
        frontiers_to_check = frontiers[:, 0] + frontiers[:, 1] * 1j
        pre_frontiers_to_check = self.frontiers[:, 0] + self.frontiers[:, 1] * 1j
        frontiers_num = np.intersect1d(frontiers_to_check, pre_frontiers_to_check).shape[0]
        pre_frontiers_num = pre_frontiers_to_check.shape[0]
        delta_num = pre_frontiers_num - frontiers_num

        reward += delta_num / 50

        return reward

    def evaluate_exploration_rate(self):
        rate = np.sum(self.robot_belief == 255) / np.sum(self.ground_truth == 255)
        return rate

    def calculate_new_free_area(self):
        old_free_area = self.old_robot_belief == 255
        current_free_area = self.robot_belief == 255
        new_free_area = (current_free_area.astype(np.int) - old_free_area.astype(np.int)) * 255

        return new_free_area, np.sum(old_free_area)

    def calculate_path_length(self, path):
        dist = 0
        start = path[0]
        end = path[-1]
        for index in path:
            if index == end:
                break
            dist += np.linalg.norm(self.node_coords[start] - self.node_coords[index])
            start = index
        return dist

    def find_frontier(self):
        # find frontiers from downsampled_belief by checking nearby 8 cells for each cell
        y_len = self.downsampled_belief.shape[0]
        x_len = self.downsampled_belief.shape[1]
        mapping = self.downsampled_belief.copy()
        belief = self.downsampled_belief.copy()
        mapping = (mapping == 127) * 1
        mapping = np.lib.pad(mapping, ((1, 1), (1, 1)), 'constant', constant_values=0)
        fro_map = mapping[2:][:, 1:x_len + 1] + mapping[:y_len][:, 1:x_len + 1] + mapping[1:y_len + 1][:, 2:] + \
                  mapping[1:y_len + 1][:, :x_len] + mapping[:y_len][:, 2:] + mapping[2:][:, :x_len] + mapping[2:][:,
                                                                                                      2:] + \
                  mapping[:y_len][:, :x_len]
        ind_free = np.where(belief.ravel(order='F') == 255)[0]
        ind_fron_1 = np.where(1 < fro_map.ravel(order='F'))[0]
        ind_fron_2 = np.where(fro_map.ravel(order='F') < 8)[0]
        ind_fron = np.intersect1d(ind_fron_1, ind_fron_2)
        ind_to = np.intersect1d(ind_free, ind_fron)

        map_x = x_len
        map_y = y_len
        x = np.linspace(0, map_x - 1, map_x)
        y = np.linspace(0, map_y - 1, map_y)
        t1, t2 = np.meshgrid(x, y)
        points = np.vstack([t1.T.ravel(), t2.T.ravel()]).T

        f = points[ind_to]
        f = f.astype(int)
        f = f * self.resolution

        return f

    def plot_env(self, n, path, step, travel_dist):
        plt.switch_backend('agg')
        # plt.ion()
        plt.cla()
        plt.figure(figsize=(6, 3))
        plt.subplot(1, 2, 1)
        plt.imshow(self.robot_belief, cmap='gray')
        plt.axis('off')
        # plt.axis((0, self.ground_truth_size[1], self.ground_truth_size[0], 0))
        #for i in range(len(self.graph_generator.x)):
        #   plt.plot(self.graph_generator.x[i], self.graph_generator.y[i], 'tan', zorder=1)
        plt.scatter(self.node_coords[:, 0], self.node_coords[:, 1], c=self.node_utility, s=8, zorder=5)
        #plt.scatter(self.frontiers[:, 0], self.frontiers[:, 1], c='r', s=2, zorder=3)
        plt.plot(self.xPoints, self.yPoints, 'b', linewidth=2, zorder=9)
        plt.plot(self.xPoints[-1], self.yPoints[-1], 'mo', markersize=4, zorder=10)
        #plt.plot(self.xPoints[0], self.yPoints[0], 'co', markersize=8)

        plt.subplot(1, 2, 2)
        plt.imshow(self.ground_truth, cmap='gray')
        plt.axis('off')
        #for i in range(len(self.ground_truth_graph_generator.x)):
        #   plt.plot(self.ground_truth_graph_generator.x[i], self.ground_truth_graph_generator.y[i], 'tan', zorder=1)
        plt.scatter(self.ground_truth_node_coords[:, 0], self.ground_truth_node_coords[:, 1], c=self.ground_truth_utility, s=8, zorder=5)
        plt.plot(self.xPoints, self.yPoints, 'b', linewidth=2, zorder=9)
        plt.plot(self.xPoints[-1], self.yPoints[-1], 'mo', markersize=4, zorder=10)
        #plt.plot(self.xPoints[0], self.yPoints[0], 'co', markersize=8)

        # plt.pause(0.1)
        plt.suptitle('Explored ratio: {:.4g}  Travel distance: {:.4g}'.format(self.explored_rate, travel_dist))
        plt.tight_layout()
        plt.savefig('{}/{}_{}_samples.png'.format(path, n, step, dpi=300))
        # plt.show()
        frame = '{}/{}_{}_samples.png'.format(path, n, step)
        self.frame_files.append(frame)

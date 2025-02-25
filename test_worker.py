import os
import matplotlib.pyplot as plt
import torch
import numpy as np
from env import Env
from agent import Agent
from utils import *
from node_manager_quadtree import NodeManager
from test_parameter import *
from copy import deepcopy
from gat_model import GraphAttentionNetwork  

if not os.path.exists(gifs_path):
    os.makedirs(gifs_path)

class TestWorker:
    def __init__(self, meta_agent_id, policy_net, global_step, device='cpu', save_image=False, greedy=True, seed=123):
        self.meta_agent_id = meta_agent_id
        self.global_step = global_step
        self.save_image = save_image
        self.device = device
        self.greedy = greedy

        np.random.seed(seed)
        torch.manual_seed(seed)

        self.env = Env(global_step, n_agent=TEST_N_AGENTS, explore=EXPLORATION, plot=self.save_image, test=True)
        self.node_manager = NodeManager(self.env.ground_truth_coords, self.env.ground_truth_info, explore=EXPLORATION, plot=self.save_image)
        self.robot_list = [Agent(i, policy_net, self.node_manager, self.device, self.save_image) for i in range(self.env.n_agent)]
        self.perf_metrics = dict()

        self.steps_taken = [0] * self.env.n_agent  # Step count for each robot
        self.force_assignment = [False] * self.env.n_agent  # Whether to force assign target
        self.assigned_target_locations = [None] * self.env.n_agent  # Assigned target locations

        self.global_node_features = None
        self.adjacency_matrix = None

        # Load GAT model
        self.gat_model = self.load_gat_model()

    def load_gat_model(self):
        input_dim = 2  # Node feature dimension
        hidden_dim = 512
        num_heads = 8
        num_layers = 4

        gat_model = GraphAttentionNetwork(input_dim, hidden_dim, num_heads, num_layers)
        checkpoint = torch.load('gat_checkpoint.pth', map_location=self.device)
        gat_model.load_state_dict(checkpoint)
        gat_model.to(self.device)
        gat_model.eval()  # Set to evaluation mode
        return gat_model

    def run_episode(self):
        done = False
        # Initial update of each robot's state
        for robot in self.robot_list:
            robot.update_graph(self.env.belief_info, deepcopy(self.env.robot_locations[robot.id]))
            robot.update_safe_graph(self.env.safe_info, self.env.uncovered_safe_frontiers, self.env.counter_safe_info)
            robot.update_planning_state(self.env.robot_locations)
        if self.save_image:
            self.plot_local_env(-1)

        max_travel_dist = 0
        length_history = [max_travel_dist]
        safe_rate_history = [self.env.safe_rate]
        explored_rate_history = [self.env.explored_rate]

        step_thresholds = list(range(10, 151, 10))
        assignments_done = {threshold: False for threshold in step_thresholds}

        for i in range(MAX_EPISODE_STEP):
            selected_locations = []
            dist_list = []

            # Check if target assignment is needed (based on step thresholds and deadlock state)
            for threshold in step_thresholds:
                if not assignments_done[threshold]:
                    for idx in range(len(self.robot_list)):
                        # When a robot's step count reaches the threshold and the environment is in a deadlock state
                        if self.steps_taken[idx] == threshold and self.env.get_deadlock_status():
                            # Obtain global node features and adjacency matrix
                            (self.global_node_features, _, _, _, _, _, _, _, self.adjacency_matrix, _, _) = \
                                self.node_manager.get_all_node_graph(self.robot_list[0].location,
                                [robot.location for robot in self.robot_list])
                            if self.global_node_features is not None and self.adjacency_matrix is not None:
                                print(f"Agent {self.meta_agent_id}: Calling GAT target assignment at step {threshold} or due to deadlock condition.")
                                self.assign_targets_based_on_best_node()
                                assignments_done[threshold] = True
                            else:
                                print("Warning: adjacency_matrix or global_node_features is None, skipping GAT assignment.")
                            break
                    if assignments_done[threshold]:
                        break

            # Each robot performs one action step
            for idx, robot in enumerate(self.robot_list):
                self.steps_taken[idx] += 1
                print(f"Agent {self.meta_agent_id}, Robot {idx}: Executed step count {self.steps_taken[idx]}")

                if self.force_assignment[idx]:
                    if robot.current_target_path:
                        next_location = robot.current_target_path.pop(0)
                    else:
                        self.force_assignment[idx] = False
                        local_obs = robot.get_local_observation(pad=False)
                        next_location, _, _ = robot.select_next_waypoint(local_obs, self.greedy)
                else:
                    local_obs = robot.get_local_observation(pad=False)
                    next_location, _, _ = robot.select_next_waypoint(local_obs, self.greedy)

                selected_locations.append(next_location)
                dist_list.append(np.linalg.norm(next_location - robot.location))

            selected_locations = np.array(selected_locations).reshape(-1, 2)
            arriving_sequence = np.argsort(np.array(dist_list))
            selected_locations_in_seq = np.array(selected_locations)[arriving_sequence]

            # Resolve conflicts where multiple robots select the same location
            for j, selected_location in enumerate(selected_locations_in_seq):
                solved_locations = selected_locations_in_seq[:j]
                while selected_location[0] + selected_location[1] * 1j in \
                      solved_locations[:, 0] + solved_locations[:, 1] * 1j:
                    robot_id = arriving_sequence[j]
                    nearby_nodes = self.robot_list[robot_id].node_manager.local_nodes_dict.nearest_neighbors(selected_location.tolist(), 25)
                    for node in nearby_nodes:
                        coords = node.data.coords
                        if coords[0] + coords[1] * 1j not in \
                           solved_locations[:, 0] + solved_locations[:, 1] * 1j:
                            selected_location = coords
                            break
                    selected_locations_in_seq[j] = selected_location
                    selected_locations[robot_id] = selected_location

            self.env.decrease_safety(selected_locations)
            self.env.step(selected_locations)
            self.env.classify_safe_frontier(selected_locations)

            for robot in self.robot_list:
                robot.update_graph(self.env.belief_info, deepcopy(self.env.robot_locations[robot.id]))
                robot.update_safe_graph(self.env.safe_info, self.env.uncovered_safe_frontiers, self.env.counter_safe_info)
                robot.update_planning_state(self.env.robot_locations)

            max_travel_dist += np.max(dist_list)
            done = self.env.check_done()

            length_history.append(max_travel_dist)
            safe_rate_history.append(self.env.safe_rate)
            explored_rate_history.append(self.env.explored_rate)

            if self.save_image:
                self.plot_local_env(i)

            if max_travel_dist >= 1000:
                max_travel_dist = 1000
                break
            if done:
                break

        # Save performance metrics
        self.perf_metrics['travel_dist'] = max([robot.travel_dist for robot in self.robot_list])
        self.perf_metrics['max_travel_dist'] = max_travel_dist
        self.perf_metrics['explored_rate'] = self.env.explored_rate
        self.perf_metrics['safe_rate'] = self.env.safe_rate
        self.perf_metrics['success_rate'] = done
        self.perf_metrics['length_history'] = length_history
        self.perf_metrics['safe_rate_history'] = safe_rate_history
        self.perf_metrics['explored_rate_history'] = explored_rate_history

        if self.save_image:
            make_gif(gifs_path, self.global_step, self.env.frame_files, self.env.explored_rate)

    def generate_path(self, start_location, target_location):
        # Generate a path from start to target (excluding the starting point) using the A* algorithm
        path_coords, _ = self.node_manager.a_star(start_location, target_location)
        return path_coords

    def assign_targets_based_on_best_node(self):
        """Invoke the GAT model to predict the best node and assign target locations to the robots"""
        with torch.no_grad():
            feat_tensor = torch.tensor(self.global_node_features, dtype=torch.float32).unsqueeze(0).to(self.device)
            adj_tensor = torch.tensor(self.adjacency_matrix, dtype=torch.float32).unsqueeze(0).to(self.device)
            node_probabilities = self.gat_model(feat_tensor, adj_tensor)  # Output shape [1, num_nodes]
        node_probabilities = node_probabilities.squeeze(0).cpu().numpy()  # [num_nodes]
        best_node_index = np.argmax(node_probabilities)
        best_node_location = self.global_node_features[best_node_index]
        print(f"Best node location: {best_node_location}")

        # Select a set of available nodes near the best node
        assigned_nodes = self.get_nodes_within_radius(best_node_location, radius=3)
        if not assigned_nodes:
            print("No available nodes found within the radius of the best node")
            return
        for idx, robot in enumerate(self.robot_list):
            self.assigned_target_locations[idx] = assigned_nodes[idx % len(assigned_nodes)]
            self.force_assignment[idx] = True
            robot.current_target_path = self.generate_path(robot.location, self.assigned_target_locations[idx])

    def get_nodes_within_radius(self, center_node, radius):
        # Select all nodes within the radius of center_node
        nearby_nodes = []
        for node in self.global_node_features:
            if np.linalg.norm(node - center_node) <= radius:
                nearby_nodes.append(node)
        return nearby_nodes

    def plot_local_env(self, step, planned_paths=None):
        plt.switch_backend('agg')
        plt.figure(figsize=(9, 4))
        plt.subplot(1, 2, 2)
        plt.imshow(self.env.robot_belief, cmap='gray', vmin=0, alpha=0)
        plt.axis('off')
        color_list = ['r', 'b', 'g', 'y', 'm', 'c', 'k', 'w', (1, 0.5, 0.5), (0.2, 0.5, 0.7)]
        robot = self.robot_list[0]
        nodes = get_cell_position_from_coords(robot.local_node_coords, robot.safe_zone_info)
        plt.scatter(nodes[:, 0], nodes[:, 1], c=robot.safe_utility, s=5, zorder=2)
        for i in range(nodes.shape[0]):
            for j in range(i + 1, nodes.shape[0]):
                if robot.local_adjacent_matrix[i, j] == 0:
                    plt.plot([nodes[i, 0], nodes[j, 0]], [nodes[i, 1], nodes[j, 1]],
                             c=(0.988, 0.557, 0.675), linewidth=1.5, zorder=1)

        plt.subplot(1, 2, 1)
        plt.imshow(self.env.robot_belief, cmap='gray')
        self.env.classify_safe_frontier(self.env.robot_locations)
        covered_safe_frontier_cells = get_cell_position_from_coords(self.env.covered_safe_frontiers, self.env.safe_info).reshape(-1, 2)
        uncovered_safe_frontier_cells = get_cell_position_from_coords(self.env.uncovered_safe_frontiers, self.env.safe_info).reshape(-1, 2)
        if covered_safe_frontier_cells.shape[0] != 0:
            plt.scatter(covered_safe_frontier_cells[:, 0], covered_safe_frontier_cells[:, 1], c='g', s=1, zorder=6)
        if uncovered_safe_frontier_cells.shape[0] != 0:
            plt.scatter(uncovered_safe_frontier_cells[:, 0], uncovered_safe_frontier_cells[:, 1], c='r', s=1, zorder=6)

        n_segments = len(self.robot_list[0].trajectory_x) - 1
        alpha_values = np.linspace(0.3, 1, n_segments)
        for robot in self.robot_list:
            c = color_list[robot.id]
            if robot.id == 0:
                alpha_mask = robot.safe_zone_info.map / 255 / 3
                plt.imshow(robot.safe_zone_info.map, cmap='Greens', alpha=alpha_mask)
                plt.axis('off')
            robot_cell = get_cell_position_from_coords(robot.location, robot.safe_zone_info)
            plt.plot(robot_cell[0], robot_cell[1], c=c, marker='o', markersize=10, zorder=5)
            for i in range(n_segments):
                plt.plot((np.array(robot.trajectory_x[i:i + 2]) - robot.global_map_info.map_origin_x) / robot.cell_size,
                         (np.array(robot.trajectory_y[i:i + 2]) - robot.global_map_info.map_origin_y) / robot.cell_size,
                         c, linewidth=2, alpha=alpha_values[i], zorder=3)

        if planned_paths is not None and planned_paths[0] is not None:
            best_node_location = planned_paths[0]
            best_node_cell = get_cell_position_from_coords(best_node_location, robot.safe_zone_info)
            plt.scatter(best_node_cell[0], best_node_cell[1], c='gold', s=100, marker='*', zorder=10, label='Best Node')
        else:
            print("No best node available to mark")

        plt.axis('off')
        plt.suptitle('Explored rate: {:.4g} | Safe rate: {:.4g} | Trajectory length: {:.4g}'.format(
            self.env.explored_rate,
            self.env.safe_rate,
            max([robot.travel_dist for robot in self.robot_list])
        ))
        plt.tight_layout()
        plt.savefig('{}/{}_{}_samples.png'.format(gifs_path, self.global_step, step))
        plt.close()
        frame = '{}/{}_{}_samples.png'.format(gifs_path, self.global_step, step)
        self.env.frame_files.append(frame)

if __name__ == '__main__':
    from model import PolicyNet
    net = PolicyNet(8, 128)
    ckp = torch.load(f'{model_path}/checkpoint.pth', map_location=torch.device('cpu'))
    net.load_state_dict(ckp['policy_model'])
    test_worker = TestWorker(0, net, 0, save_image=True, greedy=True)
    test_worker.run_episode()

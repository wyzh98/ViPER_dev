import numpy as np
import torch
from utils.utils import *
from parameter import *


class Agent:
    def __init__(self, id, policy_net, node_manager, device='cpu', plot=False):
        self.id = id
        self.device = device
        self.plot = plot
        self.policy_net = policy_net

        # map related parameters
        self.location = None
        self.map_info = None
        self.safe_zone_info = None
        self.local_map_info = None
        self.extended_local_map_info = None

        self.cell_size = CELL_SIZE
        self.downsample_size = NODE_RESOLUTION  # cell
        self.downsampled_cell_size = self.cell_size * self.downsample_size  # meter
        self.local_map_size = LOCAL_MAP_SIZE  # meter
        self.extended_local_map_size = EXTENDED_LOCAL_MAP_SIZE

        # frontiers
        self.explore_frontier = None
        self.safe_frontier = None

        # managers
        self.node_manager = node_manager

        # local graph
        (self.local_node_coords, self.explore_utility, self.safe_utility, self.uncovered_safe_utility, self.guidepost,
         self.signal, self.occupancy) = None, None, None, None, None, None, None
        self.current_local_index, self.local_adjacent_matrix, self.local_neighbor_indices, self.traversable_indices = (
            None, None, None, None)

        # ground truth graph (only for critic)
        self.true_node_coords, self.true_adjacent_matrix = None, None

        self.travel_dist = 0

        self.episode_buffer = {}
        self.add = lambda d, k, v: d.setdefault(k, []).extend(v)

        if self.plot:
            self.trajectory_x = []
            self.trajectory_y = []

    def update_map(self, map_info):
        self.map_info = map_info

    def update_safe_zone(self, safe_zone_info):
        self.safe_zone_info = safe_zone_info

    def update_local_map(self):
        self.local_map_info = self.get_local_map(self.location, self.map_info)
        self.extended_local_map_info = self.get_extended_local_map(self.location, self.map_info)

    def update_location(self, location):
        if self.location is None:
            self.location = location

        dist = np.linalg.norm(self.location - location)
        self.travel_dist += dist

        self.location = location
        node = self.node_manager.local_nodes_dict.find((location[0], location[1]))
        if node:
            node.data.set_visited()
        if self.plot:
            self.trajectory_x.append(location[0])
            self.trajectory_y.append(location[1])

    def update_explore_frontiers(self):
        self.explore_frontier = get_explore_frontier(self.extended_local_map_info)

    def update_safe_frontiers(self):
        self.safe_frontier = get_safe_zone_frontier(self.safe_zone_info, self.map_info)

    def update_graph(self, map_info, location):
        self.update_map(map_info)
        self.update_location(location)
        self.update_local_map()
        self.update_explore_frontiers()
        self.node_manager.update_local_explore_graph(self.location, self.explore_frontier, self.local_map_info,
                                                     self.extended_local_map_info)

    def update_safe_graph(self, safe_zone_info, uncovered_safe_frontiers):
        self.update_safe_zone(safe_zone_info)
        self.update_safe_frontiers()
        self.node_manager.update_safe_graph(self.location, self.safe_frontier, uncovered_safe_frontiers,
                                            self.safe_zone_info, self.extended_local_map_info)

    def update_planning_state(self, robot_locations):
        (self.local_node_coords, self.explore_utility, self.safe_utility, self.uncovered_safe_utility, self.guidepost, self.signal, self.occupancy, self.local_adjacent_matrix,
         self.current_local_index, self.local_neighbor_indices, self.traversable_indices) = self.node_manager.get_all_node_graph(self.location, robot_locations)
        self.clique_indices, self.topo_node_coords, self.topo_adjacent_matrix, self.current_topo_index \
            = self.node_manager.get_topological_node_graph(self.location, self.local_adjacent_matrix, self.local_node_coords)

    def update_underlying_state(self):
        self.true_node_coords, self.true_adjacent_matrix = self.node_manager.get_underlying_node_graph(self.local_node_coords)
        self.true_clique_indices, self.true_topo_node_coords, self.true_topo_adjacent_matrix, self.true_current_topo_index \
            = self.node_manager.get_topological_node_graph(self.location, self.true_adjacent_matrix, self.true_node_coords)

    def get_observation(self, pad=True):
        local_node_coords = self.local_node_coords
        local_node_safe_utility = self.safe_utility.reshape(-1, 1)
        local_node_uncovered_safe_utility = self.uncovered_safe_utility.reshape(-1, 1)
        local_node_guidepost = self.guidepost.reshape(-1, 1)
        local_node_occupancy = self.occupancy.reshape(-1, 1)
        local_node_signal = self.signal.reshape(-1, 1)
        current_local_index = self.current_local_index
        # local_edge_mask = self.local_adjacent_matrix
        current_local_edge = self.local_neighbor_indices
        local_traversable_edge = self.traversable_indices

        current_local_node_coords = local_node_coords[self.current_local_index]
        local_node_coords = np.concatenate((local_node_coords[:, 0].reshape(-1, 1) - current_local_node_coords[0],
                                            local_node_coords[:, 1].reshape(-1, 1) - current_local_node_coords[1]),
                                           axis=-1) / LOCAL_MAP_SIZE
        local_node_safe_utility = local_node_safe_utility / 30
        local_node_uncovered_safe_utility = local_node_uncovered_safe_utility / 30
        local_node_inputs = np.concatenate((local_node_coords, local_node_safe_utility, local_node_uncovered_safe_utility,
                                            local_node_guidepost, local_node_signal, local_node_occupancy), axis=1)
        local_node_inputs = local_node_inputs[current_local_edge, :]  # extract local neighbors only
        local_node_inputs = torch.FloatTensor(local_node_inputs).unsqueeze(0).to(self.device)
        current_local_index_in_edge = np.where(current_local_edge == current_local_index)[0]
        current_local_index_in_edge = torch.tensor(current_local_index_in_edge).reshape(1, 1, 1).to(self.device)
        k_size = current_local_edge.shape[0]

        if pad:
            padding = torch.nn.ZeroPad2d((0, 0, 0, LOCAL_K_PADDING_SIZE - k_size))
            local_node_inputs = padding(local_node_inputs)

        current_local_edge = torch.tensor(current_local_edge).unsqueeze(0).to(self.device)
        current_traversable_edge = torch.tensor(local_traversable_edge).unsqueeze(0).to(self.device)
        local_edge_padding_mask = torch.ones_like(current_local_edge).to(self.device)
        local_edge_padding_mask[torch.isin(current_local_edge, current_traversable_edge)] = 0
        if pad:
            padding = torch.nn.ConstantPad1d((0, LOCAL_K_PADDING_SIZE - k_size), 1)
            local_edge_padding_mask = padding(local_edge_padding_mask)
        local_edge_padding_mask = local_edge_padding_mask.unsqueeze(0)

        # topological graph
        topo_node_coords = self.topo_node_coords
        topo_node_safe_utility = np.asarray([local_node_safe_utility.flatten()[indices].max() for indices in self.clique_indices]).reshape(-1, 1)
        topo_node_uncovered_safe_utility = np.asarray([local_node_uncovered_safe_utility.flatten()[indices].max() for indices in self.clique_indices]).reshape(-1, 1)
        topo_node_guidepost = np.asarray([local_node_guidepost.flatten()[indices].any() for indices in self.clique_indices]).reshape(-1, 1)
        topo_node_signal = np.asarray([local_node_signal.flatten()[indices].all() for indices in self.clique_indices]).reshape(-1, 1)
        topo_node_occupancy = np.asarray([local_node_occupancy.flatten()[indices].any() for indices in self.clique_indices]).reshape(-1, 1)
        topo_node_coords = np.concatenate((topo_node_coords[:, 0].reshape(-1, 1) - current_local_node_coords[0],
                                            topo_node_coords[:, 1].reshape(-1, 1) - current_local_node_coords[1]),
                                            axis=-1) / LOCAL_MAP_SIZE
        n_topo_node = topo_node_coords.shape[0]
        topo_node_inputs = np.concatenate((topo_node_coords, topo_node_safe_utility, topo_node_uncovered_safe_utility,
                                           topo_node_guidepost, topo_node_signal, topo_node_occupancy), axis=1)
        topo_node_inputs = torch.FloatTensor(topo_node_inputs).unsqueeze(0).to(self.device)
        if pad:
            assert topo_node_coords.shape[0] < TOPOLOGICAL_NODE_PADDING_SIZE, print(topo_node_coords.shape[0])
            padding = torch.nn.ZeroPad2d((0, 0, 0, TOPOLOGICAL_NODE_PADDING_SIZE - n_topo_node))
            topo_node_inputs = padding(topo_node_inputs)
        topo_node_padding_mask = torch.zeros((1, 1, n_topo_node), dtype=torch.int16).to(self.device)
        if pad:
            topo_node_padding = torch.ones((1, 1, TOPOLOGICAL_NODE_PADDING_SIZE - n_topo_node), dtype=torch.int16).to(self.device)
            topo_node_padding_mask = torch.cat((topo_node_padding_mask, topo_node_padding), dim=-1)

        current_topo_index = torch.tensor([self.current_topo_index]).reshape(1, 1, 1).to(self.device)

        topo_edge_mask = torch.tensor(self.topo_adjacent_matrix).unsqueeze(0).to(self.device)
        if pad:
            padding = torch.nn.ConstantPad2d((0, TOPOLOGICAL_NODE_PADDING_SIZE - n_topo_node, 0, TOPOLOGICAL_NODE_PADDING_SIZE - n_topo_node), 1)
            topo_edge_mask = padding(topo_edge_mask)

        return [topo_node_inputs, topo_node_padding_mask, topo_edge_mask, current_topo_index, local_node_inputs, current_local_index_in_edge, local_edge_padding_mask]

    def get_state(self):
        true_node_coords = self.true_node_coords
        true_node_safe_utility = self.safe_utility.reshape(-1, 1)
        true_node_uncovered_safe_utility = self.uncovered_safe_utility.reshape(-1, 1)
        true_node_guidepost = self.guidepost.reshape(-1, 1)
        true_node_occupancy = self.occupancy.reshape(-1, 1)
        true_node_signal = self.signal.reshape(-1, 1)
        # state_edge_mask = self.true_adjacent_matrix
        n_true_node = true_node_coords.shape[0]
        n_padding = n_true_node - self.local_node_coords.shape[0]

        true_node_safe_utility = np.pad(true_node_safe_utility, ((0, n_padding), (0, 0)), mode='constant', constant_values=-30)
        true_node_uncovered_safe_utility = np.pad(true_node_uncovered_safe_utility, ((0, n_padding), (0, 0)), mode='constant', constant_values=-30)
        true_node_guidepost = np.pad(true_node_guidepost, ((0, n_padding), (0, 0)), mode='constant', constant_values=0)
        true_node_occupancy = np.pad(true_node_occupancy, ((0, n_padding), (0, 0)), mode='constant', constant_values=0)
        true_node_signal = np.pad(true_node_signal, ((0, n_padding), (0, 0)), mode='constant', constant_values=0)

        current_node_coords = true_node_coords[self.current_local_index]
        true_node_safe_utility = true_node_safe_utility / 30
        true_node_uncovered_safe_utility = true_node_uncovered_safe_utility / 30

        # topological graph
        true_topo_node_coords = self.true_topo_node_coords
        true_topo_node_safe_utility = np.asarray([true_node_safe_utility.flatten()[indices].max() for indices in self.true_clique_indices]).reshape(-1, 1)
        true_topo_node_uncovered_safe_utility = np.asarray([true_node_uncovered_safe_utility.flatten()[indices].max() for indices in self.true_clique_indices]).reshape(-1, 1)
        true_topo_node_guidepost = np.asarray([true_node_guidepost.flatten()[indices].any() for indices in self.true_clique_indices]).reshape(-1, 1)
        true_topo_node_signal = np.asarray([true_node_signal.flatten()[indices].all() for indices in self.true_clique_indices]).reshape(-1, 1)
        true_topo_node_occupancy = np.asarray([true_node_occupancy.flatten()[indices].any() for indices in self.true_clique_indices]).reshape(-1, 1)

        true_topo_node_coords = np.concatenate((true_topo_node_coords[:, 0].reshape(-1, 1) - current_node_coords[0],
                                                true_topo_node_coords[:, 1].reshape(-1, 1) - current_node_coords[1]),
                                                axis=-1) / LOCAL_MAP_SIZE
        n_topo_node = true_topo_node_coords.shape[0]
        state_topo_node_inputs = np.concatenate((true_topo_node_coords, true_topo_node_safe_utility, true_topo_node_uncovered_safe_utility,
                                                true_topo_node_guidepost, true_topo_node_signal, true_topo_node_occupancy), axis=1)
        state_topo_node_inputs = torch.FloatTensor(state_topo_node_inputs).unsqueeze(0).to(self.device)

        assert true_topo_node_coords.shape[0] < TOPOLOGICAL_NODE_PADDING_SIZE, print(true_topo_node_coords.shape[0])
        padding = torch.nn.ZeroPad2d((0, 0, 0, TOPOLOGICAL_NODE_PADDING_SIZE - n_topo_node))
        state_topo_node_inputs = padding(state_topo_node_inputs)

        state_topo_node_padding_mask = torch.zeros((1, 1, n_topo_node), dtype=torch.int16).to(self.device)
        topo_node_padding = torch.ones((1, 1, TOPOLOGICAL_NODE_PADDING_SIZE - n_topo_node), dtype=torch.int16).to(self.device)
        state_topo_node_padding_mask = torch.cat((state_topo_node_padding_mask, topo_node_padding), dim=-1)

        state_current_topo_index = torch.tensor([self.true_current_topo_index]).reshape(1, 1, 1).to(self.device)

        state_topo_edge_mask = torch.tensor(self.true_topo_adjacent_matrix).unsqueeze(0).to(self.device)

        padding = torch.nn.ConstantPad2d((0, TOPOLOGICAL_NODE_PADDING_SIZE - n_topo_node, 0, TOPOLOGICAL_NODE_PADDING_SIZE - n_topo_node), 1)
        state_topo_edge_mask = padding(state_topo_edge_mask)

        return [state_topo_node_inputs, state_current_topo_index, state_topo_node_padding_mask, state_topo_edge_mask]

    def select_next_waypoint(self, local_observation, greedy=False):
        with torch.no_grad():
            logp = self.policy_net(*local_observation)

        if greedy:
            action_index = torch.argmax(logp, dim=1).long()
        else:
            action_index = torch.multinomial(logp.exp(), 1).long().squeeze(1)

        next_node_index = self.local_neighbor_indices[action_index.item()]
        next_position = self.local_node_coords[next_node_index]

        return next_position, next_node_index, action_index

    def get_local_map(self, location, map_info):
        local_map_origin_x = (location[
                                  0] - self.local_map_size / 2) // self.downsampled_cell_size * self.downsampled_cell_size
        local_map_origin_y = (location[
                                  1] - self.local_map_size / 2) // self.downsampled_cell_size * self.downsampled_cell_size
        local_map_top_x = local_map_origin_x + self.local_map_size + NODE_RESOLUTION
        local_map_top_y = local_map_origin_y + self.local_map_size + NODE_RESOLUTION

        min_x = map_info.map_origin_x
        min_y = map_info.map_origin_y
        max_x = map_info.map_origin_x + self.cell_size * map_info.map.shape[1]
        max_y = map_info.map_origin_y + self.cell_size * map_info.map.shape[0]

        if local_map_origin_x < min_x:
            local_map_origin_x = min_x
        if local_map_origin_y < min_y:
            local_map_origin_y = min_y
        if local_map_top_x > max_x:
            local_map_top_x = max_x
        if local_map_top_y > max_y:
            local_map_top_y = max_y

        local_map_origin_x = np.around(local_map_origin_x, 1)
        local_map_origin_y = np.around(local_map_origin_y, 1)
        local_map_top_x = np.around(local_map_top_x, 1)
        local_map_top_y = np.around(local_map_top_y, 1)

        local_map_origin = np.array([local_map_origin_x, local_map_origin_y])
        local_map_origin_in_global_map = get_cell_position_from_coords(local_map_origin, map_info)

        local_map_top = np.array([local_map_top_x, local_map_top_y])
        local_map_top_in_global_map = get_cell_position_from_coords(local_map_top, map_info)

        local_map = map_info.map[
                    local_map_origin_in_global_map[1]:local_map_top_in_global_map[1],
                    local_map_origin_in_global_map[0]:local_map_top_in_global_map[0]]

        local_map_info = Map_info(local_map, local_map_origin_x, local_map_origin_y, self.cell_size)

        return local_map_info

    def get_extended_local_map(self, location, map_info):
        # expanding local map to involve all related frontiers
        local_map_origin_x = (location[
                                  0] - self.extended_local_map_size / 2) // self.downsampled_cell_size * self.downsampled_cell_size
        local_map_origin_y = (location[
                                  1] - self.extended_local_map_size / 2) // self.downsampled_cell_size * self.downsampled_cell_size
        local_map_top_x = local_map_origin_x + self.extended_local_map_size + 2 * NODE_RESOLUTION
        local_map_top_y = local_map_origin_y + self.extended_local_map_size + 2 * NODE_RESOLUTION

        min_x = map_info.map_origin_x
        min_y = map_info.map_origin_y
        max_x = map_info.map_origin_x + self.cell_size * map_info.map.shape[1]
        max_y = map_info.map_origin_y + self.cell_size * map_info.map.shape[0]

        if local_map_origin_x < min_x:
            local_map_origin_x = min_x
        if local_map_origin_y < min_y:
            local_map_origin_y = min_y
        if local_map_top_x > max_x:
            local_map_top_x = max_x
        if local_map_top_y > max_y:
            local_map_top_y = max_y

        local_map_origin_x = np.around(local_map_origin_x, 1)
        local_map_origin_y = np.around(local_map_origin_y, 1)
        local_map_top_x = np.around(local_map_top_x, 1)
        local_map_top_y = np.around(local_map_top_y, 1)

        local_map_origin = np.array([local_map_origin_x, local_map_origin_y])
        local_map_origin_in_global_map = get_cell_position_from_coords(local_map_origin, map_info)

        local_map_top = np.array([local_map_top_x, local_map_top_y])
        local_map_top_in_global_map = get_cell_position_from_coords(local_map_top, map_info)

        local_map = map_info.map[
                    local_map_origin_in_global_map[1]:local_map_top_in_global_map[1],
                    local_map_origin_in_global_map[0]:local_map_top_in_global_map[0]]

        local_map_info = Map_info(local_map, local_map_origin_x, local_map_origin_y, self.cell_size)

        return local_map_info

    def save_observation(self, local_observation):
        (topo_node_inputs, topo_node_padding_mask, topo_edge_mask, current_topo_index,
         local_node_inputs, current_local_index_in_edge, local_edge_padding_mask) = local_observation
        self.add(self.episode_buffer, 'node_inputs', local_node_inputs)
        self.add(self.episode_buffer, 'current_index_in_edge', current_local_index_in_edge)
        self.add(self.episode_buffer, 'edge_padding_mask', local_edge_padding_mask.bool())
        self.add(self.episode_buffer, 'topo_node_inputs', topo_node_inputs)
        self.add(self.episode_buffer, 'topo_node_padding_mask', topo_node_padding_mask.bool())
        self.add(self.episode_buffer, 'topo_edge_mask', topo_edge_mask.bool())
        self.add(self.episode_buffer, 'current_topo_index', current_topo_index)

    def save_action(self, action_index):
        self.add(self.episode_buffer, 'action', action_index.reshape(1, 1, 1).to(self.device))

    def save_reward(self, reward):
        self.add(self.episode_buffer, 'reward', torch.FloatTensor([reward]).reshape(1, 1, 1).to(self.device))

    def save_done(self, done):
        self.add(self.episode_buffer, 'done', torch.tensor([int(done)]).reshape(1, 1, 1).to(self.device))

    def save_next_observations(self, local_observation):
        self.episode_buffer['next_node_inputs'] = copy.deepcopy(self.episode_buffer['node_inputs'])[1:]
        self.episode_buffer['next_current_index_in_edge'] = copy.deepcopy(self.episode_buffer['current_index_in_edge'])[1:]
        self.episode_buffer['next_edge_padding_mask'] = copy.deepcopy(self.episode_buffer['edge_padding_mask'])[1:]
        self.episode_buffer['next_topo_node_inputs'] = copy.deepcopy(self.episode_buffer['topo_node_inputs'])[1:]
        self.episode_buffer['next_topo_node_padding_mask'] = copy.deepcopy(self.episode_buffer['topo_node_padding_mask'])[1:]
        self.episode_buffer['next_topo_edge_mask'] = copy.deepcopy(self.episode_buffer['topo_edge_mask'])[1:]
        self.episode_buffer['next_current_topo_index'] = copy.deepcopy(self.episode_buffer['current_topo_index'])[1:]

        (topo_node_inputs, topo_node_padding_mask, topo_edge_mask, current_topo_index,
         local_node_inputs, current_local_index_in_edge, local_edge_padding_mask) = local_observation
        self.episode_buffer['next_node_inputs'] += local_node_inputs
        self.episode_buffer['next_current_index_in_edge'] += current_local_index_in_edge
        self.episode_buffer['next_edge_padding_mask'] += local_edge_padding_mask.bool()
        self.episode_buffer['next_topo_node_inputs'] += topo_node_inputs
        self.episode_buffer['next_topo_node_padding_mask'] += topo_node_padding_mask.bool()
        self.episode_buffer['next_topo_edge_mask'] += topo_edge_mask.bool()
        self.episode_buffer['next_current_topo_index'] += current_topo_index

    def save_state(self, state):
        state_topo_node_inputs, state_current_topo_index, state_topo_node_padding_mask, state_topo_edge_mask = state
        self.add(self.episode_buffer, 'state_topo_node_inputs', state_topo_node_inputs)
        self.add(self.episode_buffer, 'state_current_topo_index', state_current_topo_index)
        self.add(self.episode_buffer, 'state_topo_node_padding_mask', state_topo_node_padding_mask.bool())
        self.add(self.episode_buffer, 'state_topo_edge_mask', state_topo_edge_mask.bool())

    def save_next_state(self, state):
        self.episode_buffer['next_state_topo_node_inputs'] = copy.deepcopy(self.episode_buffer['state_topo_node_inputs'])[1:]
        self.episode_buffer['next_state_current_topo_index'] = copy.deepcopy(self.episode_buffer['state_current_topo_index'])[1:]
        self.episode_buffer['next_state_topo_node_padding_mask'] = copy.deepcopy(self.episode_buffer['state_topo_node_padding_mask'])[1:]
        self.episode_buffer['next_state_topo_edge_mask'] = copy.deepcopy(self.episode_buffer['state_topo_edge_mask'])[1:]

        state_topo_node_inputs, state_current_topo_index, state_topo_node_padding_mask, state_topo_edge_mask = state
        self.episode_buffer['next_state_topo_node_inputs'] += state_topo_node_inputs
        self.episode_buffer['next_state_current_topo_index'] += state_current_topo_index
        self.episode_buffer['next_state_topo_node_padding_mask'] += state_topo_node_padding_mask.bool()
        self.episode_buffer['next_state_topo_edge_mask'] += state_topo_edge_mask.bool()


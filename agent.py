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
        self.current_local_index, self.local_adjacent_matrix, self.local_neighbor_indices, self.traversable_indices = None, None, None, None

        # topological graph
        self.cliques, self.local_node_type, self.topological_node_coords, self.topological_adjacent_matrix_padded = None, None, None, None

        # hybrid graph
        self.hybrid_node_coords, self.current_true_hybrid_index, self.hybrid_adjacent_matrix, self.hybrid_node_safe_utility = None, None, None, None

        # ground truth graph (only for critic)
        self.true_node_coords, self.true_hybrid_node_coords = None, None
        self.true_adjacent_matrix, self.true_hybrid_adjacent_matrix, self.true_topological_adjacent_matrix_padded = None, None, None
        self.true_cliques, self.true_node_type, self.true_topological_node_coords = None, None, None

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
        self.cliques, self.topological_node_coords, self.topological_adjacent_matrix_padded = self.node_manager.get_topological_node_graph(self.local_adjacent_matrix, self.local_node_coords)
        self.local_node_type = self.node_manager.get_hybrid_node_graph(robot_locations, self.local_node_coords, self.topological_node_coords, self.cliques)

    def update_underlying_state(self, robot_locations):
        self.true_node_coords, self.true_adjacent_matrix = self.node_manager.get_underlying_node_graph(self.local_node_coords)
        self.true_cliques, self.true_topological_node_coords, self.true_topological_adjacent_matrix_padded = (
            self.node_manager.get_topological_node_graph(self.true_adjacent_matrix, self.true_node_coords))
        self.true_node_type = self.node_manager.get_hybrid_node_graph(robot_locations, self.true_node_coords, self.true_topological_node_coords, self.true_cliques)

    def get_observation(self, pad=True):
        hybrid_node_coords = []
        hybrid_node_safe_utility = []
        hybrid_node_uncovered_safe_utility = []
        hybrid_node_guidepost = []
        hybrid_node_signal = []
        hybrid_node_occupancy = []
        hybrid_node_clique_center = []
        for i, coords in enumerate(self.local_node_coords):
            if self.local_node_type[i] == 0:  # keep node
                hybrid_node_coords.append(coords)
                hybrid_node_safe_utility.append(self.safe_utility[i])
                hybrid_node_uncovered_safe_utility.append(self.uncovered_safe_utility[i])
                hybrid_node_guidepost.append(self.guidepost[i])
                hybrid_node_signal.append(self.signal[i])
                hybrid_node_occupancy.append(self.occupancy[i])
                hybrid_node_clique_center.append(0)
            elif self.local_node_type[i] == 1:  # clique center
                clique_index = next(j for j, clique in enumerate(self.cliques) if i in clique)
                safe_utility_clique = self.safe_utility[self.cliques[clique_index]].max()
                uncovered_safe_utility_clique = self.uncovered_safe_utility[self.cliques[clique_index]].max()
                guidepost_clique = self.guidepost[self.cliques[clique_index]].any()
                signal_clique = self.signal[self.cliques[clique_index]].all()
                occupancy_clique = self.occupancy[self.cliques[clique_index]].any()  # always 0
                hybrid_node_coords.append(coords)
                hybrid_node_safe_utility.append(safe_utility_clique)
                hybrid_node_uncovered_safe_utility.append(uncovered_safe_utility_clique)
                hybrid_node_guidepost.append(guidepost_clique)
                hybrid_node_signal.append(signal_clique)
                hybrid_node_occupancy.append(occupancy_clique)
                hybrid_node_clique_center.append(1)
            else:  # remove non-robot-neighbor non-clique-center nodes
                pass
        hybrid_node_coords = np.array(hybrid_node_coords).reshape(-1, 2)
        self.hybrid_node_coords = hybrid_node_coords
        hybrid_node_safe_utility = np.array(hybrid_node_safe_utility).reshape(-1, 1)
        self.hybrid_node_safe_utility = hybrid_node_safe_utility
        hybrid_node_uncovered_safe_utility = np.array(hybrid_node_uncovered_safe_utility).reshape(-1, 1)
        hybrid_node_guidepost = np.array(hybrid_node_guidepost).reshape(-1, 1)
        hybrid_node_signal = np.array(hybrid_node_signal).reshape(-1, 1)
        hybrid_node_occupancy = np.array(hybrid_node_occupancy).reshape(-1, 1)
        hybrid_node_clique_center = np.array(hybrid_node_clique_center).reshape(-1, 1)

        remove_indices = np.argwhere(np.array(self.local_node_type) == -1).flatten()
        hybrid_edge_mask = np.delete(self.local_adjacent_matrix, remove_indices, axis=0)  # locally connected
        hybrid_edge_mask = np.delete(hybrid_edge_mask, remove_indices, axis=1)

        current_local_node_coords = self.local_node_coords[self.current_local_index]
        current_hybrid_index = np.argwhere(np.all(hybrid_node_coords == current_local_node_coords, axis=1)).flatten()[0]

        current_local_edge = np.argwhere(hybrid_edge_mask[current_hybrid_index] == 0).flatten()
        old_to_new_neighbor_map = {old: new for old, new in zip(self.local_neighbor_indices, current_local_edge)}
        local_traversable_edge = [old_to_new_neighbor_map[old] for old in self.traversable_indices]

        self.hybrid_adjacent_matrix = (self.local_adjacent_matrix > 0) & (self.topological_adjacent_matrix_padded > 0)  # both locally and topologically connected
        self.hybrid_adjacent_matrix = np.delete(self.hybrid_adjacent_matrix, remove_indices, axis=0)
        self.hybrid_adjacent_matrix = np.delete(self.hybrid_adjacent_matrix, remove_indices, axis=1)
        hybrid_edge_mask = self.hybrid_adjacent_matrix

        n_hybrid_node = hybrid_node_coords.shape[0]

        hybrid_node_coords = np.concatenate((hybrid_node_coords[:, 0].reshape(-1, 1) - current_local_node_coords[0],
                                            hybrid_node_coords[:, 1].reshape(-1, 1) - current_local_node_coords[1]),
                                           axis=-1) / LOCAL_MAP_SIZE
        hybrid_node_safe_utility = hybrid_node_safe_utility / 30
        hybrid_node_uncovered_safe_utility = hybrid_node_uncovered_safe_utility / 30

        node_inputs = np.concatenate((hybrid_node_coords, hybrid_node_safe_utility, hybrid_node_uncovered_safe_utility,
                                      hybrid_node_guidepost, hybrid_node_signal, hybrid_node_occupancy, hybrid_node_clique_center), axis=1)
        node_inputs = torch.FloatTensor(node_inputs).unsqueeze(0).to(self.device)

        if pad:
            assert hybrid_node_coords.shape[0] < LOCAL_NODE_PADDING_SIZE, print(hybrid_node_coords.shape[0])
            padding = torch.nn.ZeroPad2d((0, 0, 0, LOCAL_NODE_PADDING_SIZE - n_hybrid_node))
            node_inputs = padding(node_inputs)

        hybrid_node_padding_mask = torch.zeros((1, 1, n_hybrid_node), dtype=torch.int16).to(self.device)

        if pad:
            local_node_padding = torch.ones((1, 1, LOCAL_NODE_PADDING_SIZE - n_hybrid_node), dtype=torch.int16).to(self.device)
            hybrid_node_padding_mask = torch.cat((hybrid_node_padding_mask, local_node_padding), dim=-1)

        current_hybrid_index = torch.tensor([current_hybrid_index]).reshape(1, 1, 1).to(self.device)

        hybrid_edge_mask = torch.tensor(hybrid_edge_mask).unsqueeze(0).to(self.device)

        if pad:
            padding = torch.nn.ConstantPad2d((0, LOCAL_NODE_PADDING_SIZE - n_hybrid_node, 0, LOCAL_NODE_PADDING_SIZE - n_hybrid_node), 1)
            hybrid_edge_mask = padding(hybrid_edge_mask)

        current_local_edge = torch.tensor(current_local_edge).unsqueeze(0).to(self.device)
        k_size = current_local_edge.size()[-1]
        current_traversable_edge = torch.tensor(local_traversable_edge).unsqueeze(0).to(self.device)
        local_edge_padding_mask = torch.ones_like(current_local_edge).to(self.device)
        local_edge_padding_mask[torch.isin(current_local_edge, current_traversable_edge)] = 0
        if pad:
            padding0 = torch.nn.ConstantPad1d((0, LOCAL_K_PADDING_SIZE - k_size), 0)
            current_local_edge = padding0(current_local_edge)
            padding1 = torch.nn.ConstantPad1d((0, LOCAL_K_PADDING_SIZE - k_size), 1)
            local_edge_padding_mask = padding1(local_edge_padding_mask)
        current_local_edge = current_local_edge.unsqueeze(-1)
        local_edge_padding_mask = local_edge_padding_mask.unsqueeze(0)

        return [node_inputs, hybrid_node_padding_mask, hybrid_edge_mask, current_hybrid_index, current_local_edge, local_edge_padding_mask]

    def get_state(self):
        n_true_node = len(self.true_node_coords)
        n_padding = n_true_node - self.local_node_coords.shape[0]
        safe_utility_padded = np.pad(self.safe_utility, (0, n_padding), mode='constant', constant_values=-30)
        uncovered_safe_utility_padded = np.pad(self.uncovered_safe_utility, (0, n_padding), mode='constant', constant_values=-30)
        guidepost_padded = np.pad(self.guidepost, (0, n_padding), mode='constant', constant_values=0)
        signal_padded = np.pad(self.signal, (0, n_padding), mode='constant', constant_values=0)
        occupancy_padded = np.pad(self.occupancy, (0, n_padding), mode='constant', constant_values=0)

        true_hybrid_node_coords = []
        true_hybrid_node_safe_utility = []
        true_hybrid_node_uncovered_safe_utility = []
        true_hybrid_node_guidepost = []
        true_hybrid_node_signal = []
        true_hybrid_node_occupancy = []
        true_hybrid_node_clique_center = []
        for i, coords in enumerate(self.true_node_coords):
            if self.true_node_type[i] == 0:  # keep node
                true_hybrid_node_coords.append(coords)
                true_hybrid_node_safe_utility.append(safe_utility_padded[i])
                true_hybrid_node_uncovered_safe_utility.append(uncovered_safe_utility_padded[i])
                true_hybrid_node_guidepost.append(guidepost_padded[i])
                true_hybrid_node_signal.append(signal_padded[i])
                true_hybrid_node_occupancy.append(occupancy_padded[i])
                true_hybrid_node_clique_center.append(0)
            elif self.true_node_type[i] == 1:  # clique center
                clique_index = next(j for j, clique in enumerate(self.true_cliques) if i in clique)
                safe_utility_clique = safe_utility_padded[self.true_cliques[clique_index]].max()
                uncovered_safe_utility_clique = uncovered_safe_utility_padded[self.true_cliques[clique_index]].max()
                guidepost_clique = guidepost_padded[self.true_cliques[clique_index]].any()
                signal_clique = signal_padded[self.true_cliques[clique_index]].all()
                occupancy_clique = occupancy_padded[self.true_cliques[clique_index]].any()  # always 0
                true_hybrid_node_coords.append(coords)
                true_hybrid_node_safe_utility.append(safe_utility_clique)
                true_hybrid_node_uncovered_safe_utility.append(uncovered_safe_utility_clique)
                true_hybrid_node_guidepost.append(guidepost_clique)
                true_hybrid_node_signal.append(signal_clique)
                true_hybrid_node_occupancy.append(occupancy_clique)
                true_hybrid_node_clique_center.append(1)
            else:  # remove non-robot-neighbor non-clique-center nodes
                pass

        self.true_hybrid_node_coords = np.array(true_hybrid_node_coords).reshape(-1, 2)
        true_hybrid_node_coords = self.true_hybrid_node_coords
        true_hybrid_node_safe_utility = np.array(true_hybrid_node_safe_utility).reshape(-1, 1)
        true_hybrid_node_uncovered_safe_utility = np.array(true_hybrid_node_uncovered_safe_utility).reshape(-1, 1)
        true_hybrid_node_guidepost = np.array(true_hybrid_node_guidepost).reshape(-1, 1)
        true_hybrid_node_signal = np.array(true_hybrid_node_signal).reshape(-1, 1)
        true_hybrid_node_occupancy = np.array(true_hybrid_node_occupancy).reshape(-1, 1)
        true_hybrid_node_clique_center = np.array(true_hybrid_node_clique_center).reshape(-1, 1)

        remove_indices = np.argwhere(np.array(self.true_node_type) == -1).flatten()
        true_hybrid_edge_mask = np.delete(self.true_adjacent_matrix, remove_indices, axis=0)
        true_hybrid_edge_mask = np.delete(true_hybrid_edge_mask, remove_indices, axis=1)

        current_node_coords = self.true_node_coords[self.current_local_index]
        current_true_hybrid_index = np.argwhere(np.all(true_hybrid_node_coords == current_node_coords, axis=1)).flatten()[0]
        self.current_true_hybrid_index = current_true_hybrid_index

        current_true_local_edge = np.argwhere(true_hybrid_edge_mask[current_true_hybrid_index] == 0).flatten()

        self.true_hybrid_adjacent_matrix = (self.true_adjacent_matrix > 0) & (self.true_topological_adjacent_matrix_padded > 0)
        self.true_hybrid_adjacent_matrix = np.delete(self.true_hybrid_adjacent_matrix, remove_indices, axis=0)
        self.true_hybrid_adjacent_matrix = np.delete(self.true_hybrid_adjacent_matrix, remove_indices, axis=1)
        true_hybrid_edge_mask = self.true_hybrid_adjacent_matrix

        n_true_hybrid_node = true_hybrid_node_coords.shape[0]

        true_hybrid_node_coords = np.concatenate((true_hybrid_node_coords[:, 0].reshape(-1, 1) - current_node_coords[0],
                                                 true_hybrid_node_coords[:, 1].reshape(-1, 1) - current_node_coords[1]),
                                                 axis=-1) / LOCAL_MAP_SIZE
        true_hybrid_node_safe_utility = true_hybrid_node_safe_utility / 30
        true_hybrid_node_uncovered_safe_utility = true_hybrid_node_uncovered_safe_utility / 30
        state_node_inputs = np.concatenate((true_hybrid_node_coords, true_hybrid_node_safe_utility,
                                            true_hybrid_node_uncovered_safe_utility,true_hybrid_node_guidepost,
                                            true_hybrid_node_signal, true_hybrid_node_occupancy, true_hybrid_node_clique_center), axis=1)
        state_node_inputs = torch.FloatTensor(state_node_inputs).unsqueeze(0).to(self.device)

        padding = torch.nn.ZeroPad2d((0, 0, 0, LOCAL_NODE_PADDING_SIZE - n_true_hybrid_node))
        state_node_inputs = padding(state_node_inputs)

        state_node_padding_mask = torch.zeros((1, 1, n_true_hybrid_node), dtype=torch.int16).to(self.device)
        global_node_padding = torch.ones((1, 1, LOCAL_NODE_PADDING_SIZE - n_true_hybrid_node), dtype=torch.int16).to(self.device)
        state_node_padding_mask = torch.cat((state_node_padding_mask, global_node_padding), dim=-1)

        current_true_hybrid_index = torch.tensor([current_true_hybrid_index]).reshape(1, 1, 1).to(self.device)

        state_edge_mask = torch.tensor(true_hybrid_edge_mask).unsqueeze(0).to(self.device)

        padding = torch.nn.ConstantPad2d((0, LOCAL_NODE_PADDING_SIZE - n_true_hybrid_node, 0,
                                          LOCAL_NODE_PADDING_SIZE - n_true_hybrid_node), 1)
        state_edge_mask = padding(state_edge_mask)

        current_true_local_edge = torch.tensor(current_true_local_edge).unsqueeze(0).to(self.device)
        k_size = current_true_local_edge.size()[-1]
        padding = torch.nn.ConstantPad1d((0, LOCAL_K_PADDING_SIZE - k_size), 0)
        state_current_local_edge = padding(current_true_local_edge).unsqueeze(-1)

        return [state_node_inputs, state_node_padding_mask, state_edge_mask, current_true_hybrid_index, state_current_local_edge]

    def select_next_waypoint(self, local_observation, greedy=False):
        _, _, _, _, current_local_edge, _ = local_observation
        with torch.no_grad():
            logp = self.policy_net(*local_observation)

        if greedy:
            action_index = torch.argmax(logp, dim=1).long()
        else:
            action_index = torch.multinomial(logp.exp(), 1).long().squeeze(1)

        next_node_index = current_local_edge[0, action_index.item(), 0].item()
        next_position = self.hybrid_node_coords[next_node_index]

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
        local_node_inputs, local_node_padding_mask, local_edge_mask, current_local_index, current_local_edge, local_edge_padding_mask = local_observation
        self.add(self.episode_buffer, 'node_inputs', local_node_inputs)
        self.add(self.episode_buffer, 'node_padding_mask', local_node_padding_mask.bool())
        self.add(self.episode_buffer, 'edge_mask', local_edge_mask.bool())
        self.add(self.episode_buffer, 'current_index', current_local_index)
        self.add(self.episode_buffer, 'current_edge', current_local_edge)
        self.add(self.episode_buffer, 'edge_padding_mask', local_edge_padding_mask.bool())

    def save_action(self, action_index):
        self.add(self.episode_buffer, 'action', action_index.reshape(1, 1, 1).to(self.device))

    def save_reward(self, reward):
        self.add(self.episode_buffer, 'reward', torch.FloatTensor([reward]).reshape(1, 1, 1).to(self.device))

    def save_done(self, done):
        self.add(self.episode_buffer, 'done', torch.tensor([int(done)]).reshape(1, 1, 1).to(self.device))

    def save_all_indices(self, all_agent_curr_indices):
        self.add(self.episode_buffer, 'all_agent_indices', torch.tensor(all_agent_curr_indices).reshape(1, -1, 1).to(self.device))

    def save_next_observations(self, local_observation, next_node_index_list):
        self.episode_buffer['next_node_inputs'] = copy.deepcopy(self.episode_buffer['node_inputs'])[1:]
        self.episode_buffer['next_node_padding_mask'] = copy.deepcopy(self.episode_buffer['node_padding_mask'])[1:]
        self.episode_buffer['next_edge_mask'] = copy.deepcopy(self.episode_buffer['edge_mask'])[1:]
        self.episode_buffer['next_current_index'] = copy.deepcopy(self.episode_buffer['current_index'])[1:]
        self.episode_buffer['next_current_edge'] = copy.deepcopy(self.episode_buffer['current_edge'])[1:]
        self.episode_buffer['next_edge_padding_mask'] = copy.deepcopy(self.episode_buffer['edge_padding_mask'])[1:]
        self.episode_buffer['all_agent_next_indices'] = copy.deepcopy(self.episode_buffer['all_agent_indices'])[1:]

        local_node_inputs, local_node_padding_mask, local_edge_mask, current_local_index, current_local_edge, local_edge_padding_mask = local_observation
        self.episode_buffer['next_node_inputs'] += local_node_inputs
        self.episode_buffer['next_node_padding_mask'] += local_node_padding_mask.bool()
        self.episode_buffer['next_edge_mask'] += local_edge_mask.bool()
        self.episode_buffer['next_current_index'] += current_local_index
        self.episode_buffer['next_current_edge'] += current_local_edge
        self.episode_buffer['next_edge_padding_mask'] += local_edge_padding_mask.bool()
        self.episode_buffer['all_agent_next_indices'] += torch.tensor(next_node_index_list).reshape(1, -1, 1).to(self.device)
        self.episode_buffer['next_all_agent_next_indices'] = copy.deepcopy(self.episode_buffer['all_agent_next_indices'])[1:]
        self.episode_buffer['next_all_agent_next_indices'] += copy.deepcopy(self.episode_buffer['all_agent_next_indices'])[-1:]

    def save_state(self, state):
        state_node_inputs, state_node_padding_mask, state_edge_mask, current_true_hybrid_index, state_current_local_edge = state

        self.add(self.episode_buffer, 'state_node_inputs', state_node_inputs)
        self.add(self.episode_buffer, 'state_node_padding_mask', state_node_padding_mask.bool())
        self.add(self.episode_buffer, 'state_edge_mask', state_edge_mask.bool())
        self.add(self.episode_buffer, 'current_state_index', current_true_hybrid_index)
        self.add(self.episode_buffer, 'current_state_edge', state_current_local_edge)

    def save_next_state(self, state):
        self.episode_buffer['next_state_node_inputs'] = copy.deepcopy(self.episode_buffer['state_node_inputs'])[1:]
        self.episode_buffer['next_state_node_padding_mask'] = copy.deepcopy(self.episode_buffer['state_node_padding_mask'])[1:]
        self.episode_buffer['next_state_edge_mask'] = copy.deepcopy(self.episode_buffer['state_edge_mask'])[1:]
        self.episode_buffer['next_current_state_index'] = copy.deepcopy(self.episode_buffer['current_state_index'])[1:]
        self.episode_buffer['next_current_state_edge'] = copy.deepcopy(self.episode_buffer['current_state_edge'])[1:]

        state_node_inputs, state_node_padding_mask, state_edge_mask, current_true_hybrid_index, state_current_local_edge = state
        self.episode_buffer['next_state_node_inputs'] += state_node_inputs
        self.episode_buffer['next_state_node_padding_mask'] += state_node_padding_mask.bool()
        self.episode_buffer['next_state_edge_mask'] += state_edge_mask.bool()
        self.episode_buffer['next_current_state_index'] += current_true_hybrid_index
        self.episode_buffer['next_current_state_edge'] += state_current_local_edge


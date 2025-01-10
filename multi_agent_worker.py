import matplotlib.pyplot as plt
from copy import deepcopy
from env import Env
from agent import Agent
from model import PolicyNet
from utils.utils import *
from utils.node_manager_quadtree import NodeManager

if not os.path.exists(gifs_path):
    os.makedirs(gifs_path)


class Multi_agent_worker:
    def __init__(self, meta_agent_id, policy_net, global_step, device='cpu', save_image=False):
        self.meta_agent_id = meta_agent_id
        self.global_step = global_step
        self.save_image = save_image
        self.device = device

        self.env = Env(global_step, explore=EXPLORATION, plot=self.save_image)
        self.n_agent = N_AGENTS
        self.node_manager = NodeManager(self.env.ground_truth_coords, self.env.ground_truth_info, explore=EXPLORATION, plot=self.save_image)

        self.robot_list = [Agent(i, policy_net, self.node_manager, self.device, self.save_image) for i in range(self.n_agent)]

        self.episode_buffer = dict()
        self.perf_metrics = dict()

    def run_episode(self):
        done = False
        for robot in self.robot_list:
            robot.update_graph(self.env.belief_info, deepcopy(self.env.robot_locations[robot.id]))
        for robot in self.robot_list:
            robot.update_safe_graph(self.env.safe_info, self.env.uncovered_safe_frontiers)
        for robot in self.robot_list:
            robot.update_planning_state(self.env.robot_locations)
            robot.update_underlying_state()

        safe_increase_log = []
        max_travel_dist = 0
        for i in range(MAX_EPISODE_STEP):
            selected_locations = []
            dist_list = []
            next_node_index_list = []
            for robot in self.robot_list:
                observation = robot.get_observation()
                state = robot.get_state()
                robot.save_observation(observation)
                robot.save_state(state)

                next_location, next_node_index, action_index = robot.select_next_waypoint(observation)
                robot.save_action(action_index)

                selected_locations.append(next_location)
                dist_list.append(np.linalg.norm(next_location - robot.location))
                next_node_index_list.append(next_node_index)

            selected_locations = self.solve_path_confict(selected_locations, dist_list)

            curr_node_indices = np.array([robot.current_local_index for robot in self.robot_list])

            self.env.step(selected_locations, i, self.robot_list)

            for robot in self.robot_list:
                robot.update_graph(self.env.belief_info, deepcopy(self.env.robot_locations[robot.id]))
            for robot in self.robot_list:
                robot.update_safe_graph(self.env.safe_info, self.env.uncovered_safe_frontiers)

            done = self.env.check_done()

            indiv_reward, safety_increase = self.env.calculate_reward(dist_list)

            max_travel_dist += np.max(dist_list)
            if safety_increase > 0:
                safe_increase_log.append(1)
            else:
                safe_increase_log.append(0)

            for robot, reward in zip(self.robot_list, indiv_reward):
                robot.save_all_indices(np.array(curr_node_indices))
                robot.save_reward(reward)
                robot.save_done(done)
                robot.update_planning_state(self.env.robot_locations)
                robot.update_underlying_state()

            if done:
                break

        # save metrics
        self.perf_metrics['travel_dist'] = max([robot.travel_dist for robot in self.robot_list])
        self.perf_metrics['max_travel_dist'] = max_travel_dist
        self.perf_metrics['explored_rate'] = self.env.explored_rate
        self.perf_metrics['safe_rate'] = self.env.safe_rate
        self.perf_metrics['success_rate'] = done
        self.perf_metrics['safe_increase_rate'] = np.mean(safe_increase_log)

        # save episode buffer
        for robot in self.robot_list:
            observation = robot.get_observation()
            state = robot.get_state()
            robot.save_next_observations(observation, next_node_index_list)
            robot.save_next_state(state)

            for k in robot.episode_buffer:
                if k not in self.episode_buffer:
                    self.episode_buffer[k] = []
                self.episode_buffer[k] += robot.episode_buffer[k]

        # save gif
        if self.save_image:
            make_gif(gifs_path, self.global_step, self.env.frame_files, self.env.safe_rate)

    def solve_path_confict(self, selected_locations, dist_list):
        selected_locations = np.array(selected_locations).reshape(-1, 2)
        arriving_sequence = np.argsort(np.array(dist_list))
        selected_locations_in_arriving_sequence = np.array(selected_locations)[arriving_sequence]

        for j, selected_location in enumerate(selected_locations_in_arriving_sequence):
            solved_locations = selected_locations_in_arriving_sequence[:j]
            while selected_location[0] + selected_location[1] * 1j in solved_locations[:, 0] + solved_locations[:, 1] * 1j:
                id = arriving_sequence[j]
                nearby_nodes = self.robot_list[id].node_manager.local_nodes_dict.nearest_neighbors(
                    selected_location.tolist(), 25)
                for node in nearby_nodes:
                    coords = node.data.coords
                    if coords[0] + coords[1] * 1j in solved_locations[:, 0] + solved_locations[:, 1] * 1j:
                        continue
                    selected_location = coords
                    break

                selected_locations_in_arriving_sequence[j] = selected_location
                selected_locations[id] = selected_location

        return selected_locations


if __name__ == '__main__':
    from parameter import *
    import torch
    policynet = PolicyNet(NODE_INPUT_DIM, EMBEDDING_DIM)
    # ckp = torch.load('model/viper/checkpoint.pth', map_location='cpu')
    # policynet.load_state_dict(ckp['policy_model'])
    worker = Multi_agent_worker(0, policynet, 0, 'cpu', False)
    worker.run_episode()

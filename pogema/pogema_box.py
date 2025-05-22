from pogema import GridConfig
from pogema.envs import Pogema
import numpy as np
from pogema.envs import GridConfig
import gymnasium
from pogema.integrations.sample_factory import AutoResetWrapper, IsMultiAgentWrapper, MetricsForwardingWrapper

class PogemaBox(Pogema):
    def base_construct(self, grid_config=GridConfig(num_agents=3), num_boxes=1):  # Default to 1 box and 2 agents
        super().__init__(grid_config)
        if num_boxes >= grid_config.num_agents:
            raise ValueError("Number of boxes must be less than total number of agents")
        
        self.agent_indices = list(range(0, self.grid_config.num_agents - num_boxes))
        self.box_agent_indices = list(range(self.grid_config.num_agents - num_boxes, self.grid_config.num_agents))  # First num_boxes agents are boxes
        # self.agent_indices = list(range(num_boxes, self.grid_config.num_agents))  # Rest are actual agents
        
        # Modify observation space to include box channel
        full_size = self.grid_config.obs_radius * 2 + 1
        if self.grid_config.observation_type == 'default':
            self.observation_space = gymnasium.spaces.Box(-1.0, 1.0, shape=(4, full_size, full_size))  # Added box channel
        elif self.grid_config.observation_type == 'POMAPF':
            self.observation_space: gymnasium.spaces.Dict = gymnasium.spaces.Dict(
                obstacles=gymnasium.spaces.Box(0.0, 1.0, shape=(full_size, full_size)),
                agents=gymnasium.spaces.Box(0.0, 1.0, shape=(full_size, full_size)),
                boxes=gymnasium.spaces.Box(0.0, 1.0, shape=(full_size, full_size)),  # Added box channel
                xy=gymnasium.spaces.Box(low=-1024, high=1024, shape=(2,), dtype=int),
                target_xy=gymnasium.spaces.Box(low=-1024, high=1024, shape=(2,), dtype=int),
            )
        elif self.grid_config.observation_type == 'MAPF':
            self.observation_space: gymnasium.spaces.Dict = gymnasium.spaces.Dict(
                obstacles=gymnasium.spaces.Box(0.0, 1.0, shape=(full_size, full_size)),
                agents=gymnasium.spaces.Box(0.0, 1.0, shape=(full_size, full_size)),
                boxes=gymnasium.spaces.Box(0.0, 1.0, shape=(full_size, full_size)),  # Added box channel
                xy=gymnasium.spaces.Box(low=-1024, high=1024, shape=(2,), dtype=int),
                target_xy=gymnasium.spaces.Box(low=-1024, high=1024, shape=(2,), dtype=int),
            )
        
    def __init__(self, grid_config=GridConfig(num_agents=3), num_boxes=1, integration = None):
        self.base_construct(grid_config, num_boxes)
        if integration is not None:
            if integration == 'SampleFactory':
                env = PogemaBox(grid_config, num_boxes)
                env = MetricsForwardingWrapper(env)
                env = IsMultiAgentWrapper(env)
                if grid_config.auto_reset is None or grid_config.auto_reset:
                    env = AutoResetWrapper(env)
                self = env
                print('built sample factory pogema')
            else:
                raise KeyError(integration)

    def _get_adjacent_agents(self, box_pos):
        """Get agents that are adjacent to the box position."""
        up_agent = None
        down_agent = None
        left_agent = None
        right_agent = None

        for agent_idx in self.agent_indices:
            agent_pos = self.grid.positions_xy[agent_idx]
            # up
            if (agent_pos[0] == (box_pos[0] - 1)) and agent_pos[1] == box_pos[1]:
                up_agent = agent_idx
            # down
            elif (agent_pos[0] == (box_pos[0] + 1)) and agent_pos[1] == box_pos[1]:
                down_agent = agent_idx
            # left
            elif (agent_pos[0] == box_pos[0]) and (agent_pos[1] == (box_pos[1] - 1)):
                left_agent = agent_idx
            # right
            elif (agent_pos[0] == box_pos[0]) and (agent_pos[1] == (box_pos[1] + 1)):
                right_agent = agent_idx
        
        return up_agent, down_agent, left_agent, right_agent

    def step(self, action: list):
        """Modified step function to handle box-specific rewards and termination."""
        assert len(action) == len(self.agent_indices)  # Actions only for regular agents

        agents_moved = set()
        for box_idx in self.box_agent_indices:
            box_pos = self.grid.positions_xy[box_idx]
            up_agent, down_agent, left_agent, right_agent = self._get_adjacent_agents(box_pos)
            adj_agents_idx = [up_agent, down_agent, left_agent, right_agent]

            if sum(x is not None for x in adj_agents_idx) >= 2:
                adj_actions = list(action[idx] for idx in adj_agents_idx if idx is not None)

                action_counts = {}
                for action_ in adj_actions:
                    if action_ not in action_counts:
                        action_counts[action_] = 0
                    action_counts[action_] += 1
                most_common_action = max(action_counts.items(), key=lambda x: x[1])

                if most_common_action[1] < 2 or most_common_action[0] == 0: # less than 2 agents or wait action
                    continue

                # up
                if most_common_action[0] == 1:
                    # up agent
                    if up_agent is not None and action[up_agent] == 1:
                        self.grid.move(up_agent, 1)
                        agents_moved.add(up_agent)

                    # box
                    if self.grid.is_active[box_idx]:
                        self.grid.move(box_idx, 1)

                    # other agents
                    for agent_idx in [down_agent, left_agent, right_agent]:
                        if agent_idx is not None and action[agent_idx] == 1:
                            self.grid.move(agent_idx, 1)
                            agents_moved.add(agent_idx)
                
                # down
                elif most_common_action[0] == 2:
                    # down agents
                    if down_agent is not None and action[down_agent] == 2:
                        self.grid.move(down_agent, 2)
                        agents_moved.add(down_agent)

                    # box
                    if self.grid.is_active[box_idx]:
                        self.grid.move(box_idx, 2)

                    # other agents
                    for agent_idx in [up_agent, left_agent, right_agent]:
                        if agent_idx is not None and action[agent_idx] == 2:
                            self.grid.move(agent_idx, 2)
                            agents_moved.add(agent_idx)
                
                # left
                elif most_common_action[0] == 3:
                    # left agent
                    if left_agent is not None and action[left_agent] == 3:
                        self.grid.move(left_agent, 3)
                        agents_moved.add(left_agent)

                    # box
                    if self.grid.is_active[box_idx]:
                        self.grid.move(box_idx, 3)

                    # other agents
                    for agent_idx in [up_agent, down_agent, right_agent]:
                        if agent_idx is not None and action[agent_idx] == 3:
                            self.grid.move(agent_idx, 3)
                            agents_moved.add(agent_idx)
                
                # right
                elif most_common_action[0] == 4:
                    # right agent
                    if right_agent is not None and action[right_agent] == 4:
                        self.grid.move(right_agent, 4)
                        agents_moved.add(right_agent)

                    # box
                    if self.grid.is_active[box_idx]:
                        self.grid.move(box_idx, 4)

                    # other agents
                    for agent_idx in [up_agent, down_agent, left_agent]:
                        if agent_idx is not None and action[agent_idx] == 4:
                            self.grid.move(agent_idx, 4)
                            agents_moved.add(agent_idx)
                else:
                    raise ValueError(f"Invalid action: {most_common_action[0]}")
        
        for agent_idx in self.agent_indices:
            if agent_idx not in agents_moved:
                self.grid.move(agent_idx, action[agent_idx])
                agents_moved.add(agent_idx)
        
        # self.update_was_on_goal()

        # All boxes reaching their targets is the main goal
        all_boxes_on_goal = all(self.grid.on_goal(box_idx) for box_idx in self.box_agent_indices)

        rewards = [0.0 for _ in range(len(self.agent_indices))]
        for box_idx in self.box_agent_indices:
            box_pos = self.grid.positions_xy[box_idx]
            up_agent, down_agent, left_agent, right_agent = self._get_adjacent_agents(box_pos)

            for agent_idx in [up_agent, down_agent, left_agent, right_agent]:
                if agent_idx is not None:
                    if self.grid.on_goal(box_idx):
                        rewards[agent_idx] = 1.0
                    else:
                        rewards[agent_idx] = 0.1

        # Hide agents that reached their goals
        # for agent_idx in range(self.grid_config.num_agents):
        #     if self.grid.on_goal(agent_idx):
        #         self.grid.hide_agent(agent_idx)
        #         self.grid.is_active[agent_idx] = False

        # Hide boxes that reached their goals
        for box_idx in self.box_agent_indices:
            if self.grid.on_goal(box_idx):
                self.grid.hide_agent(box_idx)

        infos = self._get_infos()
        observations = self._obs()
        # Filter observations to only include regular agents
        observations = [obs for i, obs in enumerate(observations) if i in self.agent_indices]
        terminated = all_boxes_on_goal
        truncated = [False for _ in self.agent_indices]
        return observations, rewards, terminated, truncated, infos

    def _get_agents_obs(self, agent_id=0):
        """
        Returns the observation of the agent with the given id.
        Now includes separate channels for agents and boxes.
        """
        # Get base observation
        base_obs = super()._get_agents_obs(agent_id)
        
        # Create separate channels for agents and boxes
        agent_positions = np.zeros_like(base_obs[1])  # Channel for agents
        box_positions = np.zeros_like(base_obs[1])    # Channel for boxes
        
        # Get observing agent's position and observation radius
        obs_agent_x, obs_agent_y = self.grid.positions_xy[agent_id]
        r = self.grid.config.obs_radius
        
        # Fill agent positions (excluding boxes)
        for idx in self.agent_indices:
            if self.grid.is_active[idx]:
                x, y = self.grid.positions_xy[idx]
                # Convert to local coordinates
                local_x = x - obs_agent_x + r
                local_y = y - obs_agent_y + r
                # Check if within observation window
                if 0 <= local_x < 2*r + 1 and 0 <= local_y < 2*r + 1:
                    agent_positions[local_x, local_y] = 1.0
                
        # Fill box positions
        for box_idx in self.box_agent_indices:
            if self.grid.is_active[box_idx]:
                x, y = self.grid.positions_xy[box_idx]
                # Convert to local coordinates
                local_x = x - obs_agent_x + r
                local_y = y - obs_agent_y + r
                # Check if within observation window
                if 0 <= local_x < 2*r + 1 and 0 <= local_y < 2*r + 1:
                    box_positions[local_x, local_y] = 1.0
            
        # Combine channels
        return np.concatenate([
            base_obs[0][None],  # obstacles
            agent_positions[None],  # agents
            box_positions[None],    # boxes
            base_obs[2][None]       # targets
        ])

    def _pomapf_obs(self):
        """Modified POMAPF observation to include separate box channel."""
        results = []
        agents_xy_relative = self.grid.get_agents_xy_relative()
        targets_xy_relative = self.grid.get_targets_xy_relative()

        for agent_idx in self.agent_indices:
            # Get observing agent's position and observation radius
            obs_agent_x, obs_agent_y = self.grid.positions_xy[agent_idx]
            r = self.grid.config.obs_radius
            
            # Create separate channels for agents and boxes
            agent_positions = np.zeros_like(self.grid.get_positions(agent_idx))
            box_positions = np.zeros_like(self.grid.get_positions(agent_idx))
            
            # Fill agent positions (excluding boxes)
            for idx in self.agent_indices:
                if self.grid.is_active[idx]:
                    x, y = self.grid.positions_xy[idx]
                    # Convert to local coordinates
                    local_x = x - obs_agent_x + r
                    local_y = y - obs_agent_y + r
                    # Check if within observation window
                    if 0 <= local_x < 2*r + 1 and 0 <= local_y < 2*r + 1:
                        agent_positions[local_x, local_y] = 1.0
                    
            # Fill box positions
            for box_idx in self.box_agent_indices:
                if self.grid.is_active[box_idx]:
                    x, y = self.grid.positions_xy[box_idx]
                    # Convert to local coordinates
                    local_x = x - obs_agent_x + r
                    local_y = y - obs_agent_y + r
                    # Check if within observation window
                    if 0 <= local_x < 2*r + 1 and 0 <= local_y < 2*r + 1:
                        box_positions[local_x, local_y] = 1.0

            result = {
                'obstacles': self.grid.get_obstacles_for_agent(agent_idx),
                'agents': agent_positions,
                'boxes': box_positions,
                'xy': agents_xy_relative[agent_idx],
                'target_xy': targets_xy_relative[agent_idx]
            }
            results.append(result)
        return results
    
def SampleFactoryPogemaBox(grid_config, num_boxes=1):
    env = PogemaBox(grid_config, num_boxes, integration='SampleFactory')
    return env
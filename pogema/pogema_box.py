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
        
        self.box_agent_indices = list(range(num_boxes))  # First num_boxes agents are boxes
        self.agent_indices = list(range(num_boxes, self.grid_config.num_agents))  # Rest are actual agents
        
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
        adjacent_agents = []
        for agent_idx in self.agent_indices:
            if not self.grid.is_active[agent_idx]:
                continue
            agent_pos = self.grid.positions_xy[agent_idx]
            if abs(agent_pos[0] - box_pos[0]) + abs(agent_pos[1] - box_pos[1]) == 1:  # Manhattan distance = 1
                adjacent_agents.append(agent_idx)
        return adjacent_agents

    def _can_move_box(self, box_pos, actions):
        """Check if box can be moved based on adjacent agents' actions."""
        # Get agents adjacent to box
        adjacent_agents = self._get_adjacent_agents(box_pos)
        if len(adjacent_agents) < 2:
            return False, 0  # Return False and no-op action
            
        # Get actions of adjacent agents
        adjacent_actions = [actions[idx] for idx in adjacent_agents]
        
        # Check if at least 2 agents are taking the same action
        action_counts = {}
        for action in adjacent_actions:
            action_counts[action] = action_counts.get(action, 0) + 1
            
        # Find the most common action
        most_common_action = max(action_counts.items(), key=lambda x: x[1])
        if most_common_action[1] >= 2:  # At least 2 agents taking the same action
            # Check if the target position is free
            dx, dy = self.grid_config.MOVES[most_common_action[0]]
            target_pos = (box_pos[0] + dx, box_pos[1] + dy)
            if not self.grid.has_obstacle(target_pos[0], target_pos[1]):
                return True, most_common_action[0]
                
        return False, 0  # Return False and no-op action
    def move_agents(self, actions):
            """Modified movement logic to handle cooperative box movement."""
            if self.grid.config.collision_system == 'priority':
                # Then try to move boxes if possible
                for box_idx in self.box_agent_indices:
                    if self.grid.is_active[box_idx]:
                        box_pos = self.grid.positions_xy[box_idx]
                        can_move, box_action = self._can_move_box(box_pos, actions)
                        if can_move:
                            self.grid.move(box_idx, box_action)
                # First move regular agents
                for agent_idx in self.agent_indices:
                    if self.grid.is_active[agent_idx]:
                        self.grid.move(agent_idx, actions[agent_idx])
            
            else:
                # For other collision systems, use the same logic but with box movement check
                used_cells = {}
                agents_xy = self.grid.get_agents_xy()
                
                # First process regular agents
                for agent_idx in self.agent_indices:
                    if self.grid.is_active[agent_idx]:
                        x, y = agents_xy[agent_idx]
                        dx, dy = self.grid_config.MOVES[actions[agent_idx]]
                        used_cells[x + dx, y + dy] = 'blocked' if (x + dx, y + dy) in used_cells else 'visited'
                        used_cells[x, y] = 'blocked'
                
                # Then try to move boxes
                for box_idx in self.box_agent_indices:
                    if self.grid.is_active[box_idx]:
                        x, y = agents_xy[box_idx]
                        can_move, box_action = self._can_move_box((x, y), actions)
                        if can_move:
                            dx, dy = self.grid_config.MOVES[box_action]
                            used_cells[x + dx, y + dy] = 'blocked' if (x + dx, y + dy) in used_cells else 'visited'
                            used_cells[x, y] = 'blocked'
            
                # Apply movements
                for agent_idx in range(self.grid_config.num_agents):
                    if self.grid.is_active[agent_idx]:
                        x, y = agents_xy[agent_idx]
                        if agent_idx in self.box_agent_indices:
                            can_move, box_action = self._can_move_box((x, y), actions)
                            if can_move:
                                dx, dy = self.grid_config.MOVES[box_action]
                                if used_cells.get((x + dx, y + dy), None) != 'blocked':
                                    self.grid.move(agent_idx, box_action)
                        else:
                            dx, dy = self.grid_config.MOVES[actions[agent_idx]]
                            if used_cells.get((x + dx, y + dy), None) != 'blocked':
                                self.grid.move(agent_idx, actions[agent_idx])

    def step(self, action: list):
        """Modified step function to handle box-specific rewards and termination."""
        assert len(action) == len(self.agent_indices)  # Actions only for regular agents
        rewards = []
        terminated = []

        # Create full action list including boxes (boxes don't have actions)
        full_actions = [0] * self.grid_config.num_agents  # Initialize with no-op actions
        for i, agent_idx in enumerate(self.agent_indices):
            full_actions[agent_idx] = action[i]

        # Store previous positions to check if boxes were moved
        # prev_box_positions = {box_idx: self.grid.positions_xy[box_idx] for box_idx in self.box_agent_indices}
        
        self.move_agents(full_actions)
        self.update_was_on_goal()
# All boxes reaching their targets is the main goal
        all_boxes_on_goal = all(self.grid.on_goal(box_idx) for box_idx in self.box_agent_indices)
        
        # Only return rewards for regular agents
        for agent_idx in self.agent_indices:
            if self.grid.is_active[agent_idx]:
                agent_pos = self.grid.positions_xy[agent_idx]
                reward = 0.0
                
                # Check if agent helped move any box to its goal
                for box_idx in self.box_agent_indices:
                    if self.grid.is_active[box_idx]:
                        box_pos = self.grid.positions_xy[box_idx]
                        # prev_box_pos = prev_box_positions[box_idx]
                        
                        # If box is on goal and was moved this step
                        if self.grid.on_goal(box_idx):
                            # Check if agent was adjacent to box's previous position
                            if agent_idx in self._get_adjacent_agents(box_pos):
                                reward += 1.0  # Large reward for helping move box to goal
                
                # Small reward for being adjacent to any box
                adjacent_agents = self._get_adjacent_agents(agent_pos)
                if any(idx in self.box_agent_indices for idx in adjacent_agents):
                    reward += 0.1  # Small reward for being near a box
                
                rewards.append(reward)
            else:
                rewards.append(0.0)
            
            # Episode terminates when all boxes reach their targets
            terminated.append(all_boxes_on_goal)

        # Hide agents that reached their goals
        # for agent_idx in range(self.grid_config.num_agents):
        #     if self.grid.on_goal(agent_idx):
        #         self.grid.hide_agent(agent_idx)
        #         self.grid.is_active[agent_idx] = False

        infos = self._get_infos()
        observations = self._obs()
        # Filter observations to only include regular agents
        observations = [obs for i, obs in enumerate(observations) if i in self.agent_indices]
        truncated = [False] * len(self.agent_indices)
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
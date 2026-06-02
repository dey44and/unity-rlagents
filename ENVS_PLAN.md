## MAPPO Sketch With `mlagents_envs` Low-Level API

This is a compact training-loop sketch for cooperative MARL with centralized training and decentralized execution (CTDE):

- Actor input: each agent's local observation.
- Critic input: local observation + global observation.
- Action application: only for agents present in `DecisionSteps`.

### Assumptions

- You use one behavior name for one team of agents.
- Actions are continuous (replace with `discrete` branch if needed).
- `decision_steps.obs[0]` is local observation.
- Global observation comes from:
  - Option A: `decision_steps.obs[1]` (preferred, exported by Unity), or
  - Option B: concatenation of all local observations at the same step.

### Pseudocode

```python
import numpy as np
import torch

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple


def build_global_obs(local_obs: np.ndarray, agent_ids: np.ndarray) -> np.ndarray:
    # local_obs: [N, local_dim]
    # Global state by concatenating local obs in stable agent-id order.
    order = np.argsort(agent_ids)
    flat_global = local_obs[order].reshape(1, -1)          # [1, N * local_dim]
    return np.repeat(flat_global, local_obs.shape[0], 0)   # [N, global_dim]


env = UnityEnvironment(file_name="path/to/YourEnv", no_graphics=True, seed=1)
env.reset()
behavior_name = list(env.behavior_specs.keys())[0]
spec = env.behavior_specs[behavior_name]

rollout_buffer = []

while training:
    decision_steps, terminal_steps = env.get_steps(behavior_name)

    # Agents that need actions now.
    agent_ids = decision_steps.agent_id
    n_agents = len(agent_ids)

    if n_agents > 0:
        local_obs = decision_steps.obs[0]  # [N, local_dim]

        # Option A: global_obs = decision_steps.obs[1]
        # Option B:
        global_obs = build_global_obs(local_obs, agent_ids)  # [N, global_dim]

        # Actor uses local obs only.
        local_obs_t = torch.as_tensor(local_obs, dtype=torch.float32, device=device)
        action_t, logp_t = actor.sample(local_obs_t)  # action_t: [N, act_dim]

        # Critic uses centralized input.
        global_obs_t = torch.as_tensor(global_obs, dtype=torch.float32, device=device)
        critic_in = torch.cat([local_obs_t, global_obs_t], dim=-1)
        value_t = critic(critic_in)  # [N, 1]

        # IMPORTANT: rows must match decision_steps agent order.
        action_np = action_t.detach().cpu().numpy().astype(np.float32)
        env.set_actions(behavior_name, ActionTuple(continuous=action_np))

        # Store transition pieces keyed by agent id for MAPPO update later.
        rollout_buffer.append(
            {
                "agent_ids": agent_ids.copy(),
                "local_obs": local_obs.copy(),
                "global_obs": global_obs.copy(),
                "actions": action_np.copy(),
                "logp": logp_t.detach().cpu().numpy().copy(),
                "values": value_t.detach().cpu().numpy().copy(),
            }
        )

    env.step()

    # Read next step to get rewards/dones for GAE targets.
    next_decision_steps, next_terminal_steps = env.get_steps(behavior_name)

    # terminal_steps contains ended agents this step.
    # Use this to mark done flags and stop bootstrap for those agents.
    # MAPPO update is then done from rollout_buffer (advantages, returns, PPO loss).
```

### Practical Notes

- Keep a stable mapping `agent_id -> trajectory` if agents can spawn/despawn.
- If not all agents request decisions at each step, handle partial batches with masks.
- During inference/deployment, critic/global input is not required; actor with local obs is enough.
- If Unity can expose explicit team/global stats as a separate sensor, prefer that over concatenation.

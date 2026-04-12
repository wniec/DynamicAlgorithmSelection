import torch
import numpy as np
import matplotlib.pyplot as plt

from dynamicalgorithmselection.agents.agent_state import ELA_FEATURES
from dynamicalgorithmselection.agents.ppo_utils import Actor

actor = Actor(n_actions=3, input_size=33)
weights_path = "/home/wladek/IdeaProjects/DynamicAlgorithmSelection/DAS_CV_G3PCX_LMCMAES_SPSO_PG_MULTIDIMENSIONAL_CV-LOIO_CDB2.1_SEED34_final.pth"
state_dict = torch.load(weights_path, weights_only=False)
actor.load_state_dict(state_dict["actor_parameters"])

actor.eval()
ACTIONS = ["G3PCX", "LMCMAES", "SPSO"]
feature_names = ELA_FEATURES + [
    *(f"last_action_is_{i}" for i in ACTIONS),
    "same_action_counter",
    *(f"{i}_choice_frequency" for i in ACTIONS),
    "choice_entropy",
    "problem_dimensionality",
    "function_evaluations",
    "stagnation_count",
]

# --- Directional Gradient Method ---


def calculate_directional_impact(model, states_tensor, num_actions=3):
    states_tensor.requires_grad_(True)
    action_outputs = model(states_tensor)

    directional_impacts = np.zeros((num_actions, states_tensor.shape[1]))
    importance_magnitudes = np.zeros((num_actions, states_tensor.shape[1]))

    for action_idx in range(num_actions):
        target_outputs = action_outputs[:, action_idx].sum()
        model.zero_grad()
        if states_tensor.grad is not None:
            states_tensor.grad.zero_()

        target_outputs.backward(retain_graph=True)
        gradients = states_tensor.grad.detach().numpy()

        # Calculate raw average (Impact Direction)
        directional_impacts[action_idx] = np.mean(gradients, axis=0)
        # Calculate absolute average (Importance Magnitude)
        importance_magnitudes[action_idx] = np.mean(np.abs(gradients), axis=0)

    return directional_impacts, importance_magnitudes


# Generate dummy normalized states (Mean 0, Variance 1 matches torch.randn perfectly)
# Replace with real sampled states from your buffer for exact production results!
num_samples = 10_000
sample_states = torch.randn((num_samples, 33), dtype=torch.float32)

impacts, magnitudes = calculate_directional_impact(actor, sample_states, num_actions=3)

# --- Visualization ---

fig, axes = plt.subplots(1, 3, figsize=(18, 10))
fig.suptitle("Feature Impact on Action Probability", fontsize=16)

for action_idx, action in enumerate(ACTIONS):
    ax = axes[action_idx]

    # Sort by MAGNITUDE so the most influential features are still at the top,
    # but we will plot the RAW IMPACT on the x-axis.
    mags = magnitudes[action_idx]
    raw_impacts = impacts[action_idx]

    # Sort descending by magnitude (Absolute Importance)
    sorted_indices = np.argsort(mags)[::-1]

    sorted_impacts = []
    sorted_names = []
    colors = []

    for i in sorted_indices:
        impact_val = raw_impacts[i]
        sorted_impacts.append(impact_val)

        sorted_names.append(feature_names[i])
        # Color code: Green for positive correlation, Red for negative
        colors.append("seagreen" if impact_val >= 0 else "indianred")
    sorted_names.reverse()
    sorted_impacts.reverse()
    colors.reverse()

    # Plotting
    y_pos = np.arange(len(sorted_names))
    ax.barh(y_pos, sorted_impacts, align="center", color=colors)
    ax.axvline(0, color="black", linewidth=1.2)

    # Formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_names)
    ax.set_xlabel("Average Gradient (Impact per 1 Std Dev)")
    ax.set_title(f"Action: {action}")
    ax.xaxis.grid(True, linestyle="--", alpha=0.7)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

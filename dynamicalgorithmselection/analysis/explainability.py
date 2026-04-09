import torch
import numpy as np

from dynamicalgorithmselection.agents.agent_state import ELA_FEATURES
from dynamicalgorithmselection.agents.ppo_utils import Actor

# --- 1. Your Original Setup ---
actor = Actor(n_actions=3, input_size=33)
weights_path = "/home/wladek/IdeaProjects/DynamicAlgorithmSelection/DAS_CV_G3PCX_CMAES_MADDE_PG_CV-LOPO_CDB1.5_DIM10_SEED123_final.pth"
state_dict = torch.load(weights_path, weights_only=False)
actor.load_state_dict(state_dict["actor_parameters"])

# Ensure the model is in evaluation mode (disables dropout, fixes batch norm)
actor.eval()
ACTIONS = ["G3PCX", "CMAES", "MADDE"]
feature_names = ELA_FEATURES + [
    "last_action_encoded_1",
    "last_action_encoded_2",
    "same_action_counter",
    *(f"{i}_choice_frequency" for i in ACTIONS),
    "choice_entropy",
    "problem_dimensionality",
    "function_evaluations",
    "stagnation_count",
]

# --- 2. Gradient Feature Importance Method ---


def calculate_averaged_gradients(model, states_tensor, num_actions=3):
    """
    Calculates the average absolute gradient of each action output
    with respect to the input state over a batch of samples.
    """
    # Enable gradient tracking on the inputs
    states_tensor.requires_grad_(True)

    # Forward pass
    # NOTE: If your Actor returns a tuple (e.g., action, log_prob) or a Distribution,
    # you will need to extract just the raw probabilities/logits tensor here.
    action_outputs = model(states_tensor)

    # Store the importance scores (num_actions x input_size)
    feature_importances = np.zeros((num_actions, states_tensor.shape[1]))

    for action_idx in range(num_actions):
        # Sum the outputs for the target action across all samples.
        # This allows us to do a single backward pass for the whole batch.
        target_outputs = action_outputs[:, action_idx].sum()

        # Zero out any existing gradients in the model and the input tensor
        model.zero_grad()
        if states_tensor.grad is not None:
            states_tensor.grad.zero_()

        # Backward pass to calculate gradients
        target_outputs.backward(retain_graph=True)

        # Extract gradients (shape: [num_samples, 33])
        gradients = states_tensor.grad.detach().numpy()

        # Calculate the mean of the ABSOLUTE gradients across the batch.
        # We use absolute values because negative gradients still indicate high importance
        # (meaning an increase in the feature decreases the action probability).
        avg_abs_gradients = np.mean(np.abs(gradients), axis=0)
        feature_importances[action_idx] = avg_abs_gradients

    return feature_importances


# --- 3. Execution ---

# Generate or load a batch of multiple sample states.
# IMPORTANT: For the most accurate results, replace torch.randn with a batch
# of REAL states collected from your environment's observation space!
num_samples = 1000
sample_states = torch.randn((num_samples, 33), dtype=torch.float32)

# Calculate the importances
importances = calculate_averaged_gradients(actor, sample_states, num_actions=3)

# Print the top 5 most influential features for each action
print(f"Evaluated over {num_samples} samples.\n")
for action_idx, action in enumerate(ACTIONS):
    print(f"--- Top Features for {action} choice ---")

    # Get the indices of the features sorted by importance (descending)
    sorted_indices = np.argsort(importances[action_idx])[::-1]

    for rank in range(5):  # Show top 5
        feat_idx = sorted_indices[rank]
        score = importances[action_idx][feat_idx]
        print(
            f"Rank {rank + 1}: {feature_names[feat_idx]} | Importance Score = {score:.6f}"
        )
    print()

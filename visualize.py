import json
import matplotlib.pyplot as plt

with open("actor_losses.json", "r") as f:
    actor_losses = json.load(f)
with open("critic_losses.json", "r") as f:
    critic_losses = json.load(f)

fig, ax = plt.subplots(2)


ax[0].plot([i for i in range(len(actor_losses))], actor_losses)
ax[1].plot([i for i in range(len(critic_losses))], critic_losses)

ax[0].set_title("Actor Losses")

ax[1].set_title("Critic Losses")

plt.show()

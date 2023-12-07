import comet_ml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from comet_ml.integration.gymnasium import CometLogger
from stable_baselines3 import A2C
import gymnasium as gym

class Encoder(nn.Module):
    def __init__(self, input_dim=11, latent_dim=5):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2_mean = nn.Linear(64, latent_dim)
        self.fc2_logvar = nn.Linear(64, latent_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mean = self.fc2_mean(x)
        logvar = self.fc2_logvar(x)
        return mean, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim=5, output_dim=11):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 64)
        self.fc2 = nn.Linear(64, output_dim)

    def forward(self, z):
        z = F.relu(self.fc1(z))
        x_hat = self.fc2(z)
        return x_hat

class Translator(nn.Module):
    def __init__(self, latent_dim=5, output_dim=3):
        super(Translator, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 64)
        self.fc2 = nn.Linear(64, 2)
        self.fc3 = nn.Linear(64, 11) 

    def forward(self, z):
        z = F.relu(self.fc1(z))
        psi_params_array = self.fc3(z)
        psi_params_scalar = self.fc2(z)
        return psi_params_array, psi_params_scalar

class VAE(nn.Module):
    def __init__(self, input_dim=11, latent_dim=5, output_dim=3):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)
        self.translator = Translator(latent_dim, output_dim)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        x_hat = self.decoder(z)
        psi_params_array, psi_params_scalar = self.translator(z)
        return x_hat, mean, logvar, psi_params_array, psi_params_scalar

# Instantiate the GMR model
vae = VAE()
optimizer = optim.Adam(vae.parameters(), lr=0.001)

# Define hyperparameters
alpha = 0.1
beta = 0.01

# Define loss functions
mse_loss = torch.nn.MSELoss()

# Training loop
num_epochs = 10

env = gym.make('InvertedDoublePendulum-v4')
model = A2C.load("/work/06970/scr2543/ls6/gym/models/a2c2")
model.set_env(env)
num_episodes = 100

states = []
actions = []
for episode in range(num_episodes):
    state, _  = env.reset()  # Reset the environment at the beginning of each episode
    total_reward = 0

    while True:
        action, _ = model.predict(state, deterministic=True)
        states.append(state)
        actions.append(action)
        # Perform the action and observe the next state and reward
        next_state, reward, done, _, _ = env.step(action)
        # Update the state for the next iteration
        state = next_state

        if done:
            break
    
for epoch in range(num_epochs):
    total_loss = 0.0

    for i in range(len(actions)):  # iterate over your dataset
        optimizer.zero_grad()
        x = states[i]
        x = torch.tensor(x, dtype=torch.float32)
        u = actions[i]
        u = torch.tensor(u, dtype=torch.float32)
        # Forward pass
        x_hat, mean, logvar, psi_params_pred_array, psi_params_pred_scalar = vae(x)

        # Compute losses
        reconstruction_loss = mse_loss(x, x_hat)
        kl_divergence = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        pred_u = torch.normal(mean=(psi_params_pred_array @ x + psi_params_pred_scalar[0]), std=torch.abs(psi_params_pred_scalar[1]))
        control_loss = mse_loss(pred_u, u)
        regularization_loss = sum(p.pow(2).sum() for p in vae.parameters())

        gmr_loss = (reconstruction_loss + alpha * kl_divergence + control_loss + beta * regularization_loss)

        # Backward pass
        gmr_loss.backward()
        optimizer.step()

        total_loss += gmr_loss.item()

    # Print average loss for the epoch
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(actions)}")

# Set the model in evaluation mode
vae.eval()

comet_ml.init(project_name="GMR_log")
experiment = comet_ml.Experiment()
env = CometLogger(env, experiment)
num_episodes = 100
for episode in range(num_episodes):
    state, _  = env.reset()  # Reset the environment at the beginning of each episode
    while True:
        state = torch.tensor(state, dtype=torch.float32)
        x_hat, mean, logvar, psi_params_pred_array, psi_params_pred_scalar = vae(state)
        action = torch.normal(mean=(psi_params_pred_array @ x + psi_params_pred_scalar[0]), std=torch.abs(psi_params_pred_scalar[1]))
        # Perform the action and observe the next state and reward
        next_state, reward, done, _, _ = env.step([action.item()])


        # Update the state for the next iteration
        state = next_state

        if done:
            break
env.close()
experiment.end()



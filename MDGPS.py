import numpy as np
from scipy.optimize import minimize
import comet_ml
from comet_ml.integration.gymnasium import CometLogger
import gymnasium as gym

# Mujoco double inverted pyramid environment and training parameters
state_dim = 11
action_dim = 1
num_trajectories = 100
trajectory_length = 100
num_iterations = 10

# Define the initial policy
policy_mean = np.zeros((trajectory_length, action_dim))
policy_covariance = np.eye(action_dim)

# Lagrange multipliers
lambda_values = np.ones(trajectory_length)

# Epsilon for the KL-divergence constraint
epsilon = 0.01

# LQR parameters
Q_terminal = np.eye(state_dim)  # Terminal state cost
R = np.eye(action_dim)  # Action cost
gamma = 0.99  # Discount factor

goal_state = np.zeros(state_dim)

env = gym.make('InvertedDoublePendulum-v4')

def kl_divergence(mean1, cov1, mean2, cov2):
    """
    Calculate KL divergence between two multivariate normal distributions.

    Parameters:
        mean1 (numpy.ndarray): Mean of the first distribution.
        cov1 (numpy.ndarray): Covariance matrix of the first distribution.
        mean2 (numpy.ndarray): Mean of the second distribution.
        cov2 (numpy.ndarray): Covariance matrix of the second distribution.

    Returns:
        float: KL divergence between the two distributions.
    """
    dim = len(mean1)
    
    # Determinants of the covariance matrices
    det_cov1 = np.linalg.det(cov1)
    det_cov2 = np.linalg.det(cov2)
    
    # Inverse of the covariance matrices
    inv_cov1 = np.linalg.inv(cov1)
    inv_cov2 = np.linalg.inv(cov2)
    
    # KL divergence calculation
    kl = 0.5 * (np.trace(inv_cov2 @ cov1) + np.dot((mean2 - mean1).T, np.dot(np.squeeze(inv_cov2), (mean2 - mean1))) -
                dim + np.log(det_cov2 / det_cov1))
    
    return kl

def reward_model(states, actions, policy_mean, policy_covariance, lambdas):
    # Compute the original reward term J(τ)
    original_reward = np.sum(states) + np.sum(actions)

    # Compute the KL-divergence regularized term
    kld_values = kl_divergence(policy_mean, policy_covariance, policy_mean, policy_covariance)
    kld_term = np.sum(lambdas * kld_values)

    # Combine the terms to get the final reward model
    total_reward = original_reward + kld_term

    return total_reward

def constraint_violation(kld_values):
    return np.max(kld_values) - epsilon


def lqr_backward(trajectories, policy_mean, policy_covariance, lambda_values):
    # Initialize the value function for the terminal state
    V_terminal = np.zeros(state_dim)

    # Initialize Q-function for the terminal state
    Q_terminal = np.zeros((state_dim, action_dim, action_dim))

    # Placeholder for the LQR gains
    K_list = []
    # Backward pass
    for t in range(trajectory_length - 1, -1, -1):
        states = trajectories['states'][:, t, :]
        actions = trajectories['actions'][:, t, :]

        # Compute the state-action value (Q-function)
        Q_t = reward_model(states, actions, policy_mean, policy_covariance, lambda_values) + gamma * V_terminal
        # Compute the LQR gains
        Q_t = Q_t.reshape(-1, 1, 1)

        K = np.linalg.inv(R + gamma * Q_terminal) @  Q_t
        K_list.insert(0, K)

        # Update the Q-function for the current time step
        Q_terminal = Q_t + R * K - K * (gamma * Q_terminal)
        
    return K_list


def lqr_forward(trajectories, K_list):
    # Placeholder for the updated trajectories
    updated_trajectories = {'states': np.zeros((num_trajectories, trajectory_length, state_dim)),
                            'actions': np.zeros((num_trajectories, trajectory_length, action_dim))}

    # Forward pass
    for t in range(trajectory_length):
        states = trajectories['states'][:, t, :]
        actions = trajectories['actions'][:, t, :]

        # Update the action using the LQR gain
        updated_actions = actions + (np.squeeze(K_list[t]) @ (states - goal_state).T)[:, None]
        
        # Clip the actions to be within the valid range, if necessary
        updated_actions = np.clip(updated_actions, -1, 1)
        updated_trajectories['actions'][:, t, :] = updated_actions

        # Execute the action in the environment and observe the next state
    for i in range(num_trajectories):
        state = np.zeros((trajectory_length, state_dim))
        next_state, _ = env.reset()
        t = 0
        
        while True:
            state[t] = next_state
            action = updated_trajectories['actions'][i][t]
            t += 1
            next_state, reward, done, _, _ = env.step(action)
            if done:
                break
        state[t] = next_state

        # Update the trajectories
        updated_trajectories['states'][i] = state

    return updated_trajectories

# KL-divergence adjustment
def adjust_lambda(old_policy, old_policy_covariance, lambdas):
    # Perform the adjustment until the constraint violation is met
    while True:
        # Perform Mirror Descent trajectory optimization to update the policy
        def optimization_objective(params, old_policy_mean, old_policy_covariance):
            policy_mean_new = np.reshape(params[:trajectory_length * action_dim], (trajectory_length, action_dim))
            policy_covariance_new = np.reshape(params[-1], (action_dim, action_dim))
            total_cost = 0
            for t in range(trajectory_length):
                total_cost += reward_model(
                trajectories['states'][:, t, :],
                trajectories['actions'][:, t, :],
                policy_mean_new,
                policy_covariance_new,
                lambda_values)
            
            # Calculate the KL divergence between the old and new policies
            kld = kl_divergence(old_policy_mean, old_policy_covariance, policy_mean_new, policy_covariance_new)
            old_policy_mean = policy_mean_new
            old_policy_covariance = policy_covariance_new
            # Apply the loss term with KL-divergence regularization
            loss_term = total_cost + np.sum(lambdas * kld)
            objective = loss_term
            
            return objective

        # Initialize the optimization parameters
        initial_params = np.concatenate([old_policy.flatten(), old_policy_covariance.flatten()])
        # Perform the optimization
        result = minimize(optimization_objective, initial_params, args=(old_policy, old_policy_covariance), method='L-BFGS-B')
        
        # Update the policy with the optimized parameters
        new_policy = np.reshape(result.x[:trajectory_length * action_dim], (trajectory_length, action_dim))
        new_policy_covariance = np.reshape(result.x[-1], (action_dim,action_dim))
        
        # Check the constraint violation
        kld_values = kl_divergence(old_policy, old_policy_covariance, new_policy, new_policy_covariance)
        violation = constraint_violation(kld_values)
        old_policy = new_policy
        old_policy_covariance = new_policy_covariance
        # Update lambda values
        lambdas = lambdas + violation  # You may need to adjust this based on your specific requirements

        # Break the loop if the constraint violation is below the threshold
        if violation <= 0:
            break

    return lambdas

# Main Mirror Descent GPS loop with LQR steps
for iteration in range(num_iterations):
    # Collect trajectories by sampling from the current policy
    trajectories = {'states': np.zeros((num_trajectories, trajectory_length, state_dim)),
                    'actions': np.zeros((num_trajectories, trajectory_length, action_dim))}
    for i in range(num_trajectories):
        state = np.zeros((trajectory_length, state_dim))
        actions = np.zeros((trajectory_length, action_dim))
        next_state, _ = env.reset()
        t = 0
        
        while True:
            state[t] = next_state
            action = np.random.multivariate_normal(policy_mean[t], policy_covariance, 1)
            actions[t] = action[0]
            t += 1
            next_state, reward, done, _, _ = env.step(action[0])
            if done:
                break
        state[t] = next_state

        trajectories['states'][i] = state
        trajectories['actions'][i] = actions
    
    # Perform c1-step LQR backward
    lqr_gains = lqr_backward(trajectories, policy_mean, policy_covariance, lambda_values)

    # Perform c2-step LQR forward
    trajectories = lqr_forward(trajectories, lqr_gains)

    # Perform c3-step adjust lambda
    lambda_values = adjust_lambda(policy_mean, policy_covariance, lambda_values)

    # Perform Mirror Descent trajectory optimization to update the policy
    def optimization_objective(params):
            policy_mean_new = np.reshape(params[:trajectory_length * action_dim], (trajectory_length, action_dim))
            policy_covariance_new = np.reshape(params[-1], (action_dim, action_dim))
            total_cost = 0
            for t in range(trajectory_length):
                total_cost += reward_model(
                trajectories['states'][:, t, :],
                trajectories['actions'][:, t, :],
                policy_mean_new,
                policy_covariance_new,
                lambda_values)

            # Calculate the KL divergence between the old and new policies
            kld = kl_divergence(policy_mean, policy_covariance, policy_mean_new, policy_covariance_new)
            # Apply the loss term with KL-divergence regularization
            loss_term = total_cost + np.sum(lambda_values * kld)
            objective = loss_term

            return objective
    # Initialize the optimization parameters
    initial_params = np.concatenate([policy_mean.flatten(), policy_covariance.flatten()])

    # Perform the optimization
    result = minimize(optimization_objective, initial_params, method='L-BFGS-B')
    
    # Update the policy with the optimized parameters
    policy_mean = np.reshape(result.x[:trajectory_length * action_dim], (trajectory_length, action_dim))
    policy_covariance = np.reshape(result.x[-1], (action_dim, action_dim))
    
    # Print the total cost for monitoring purposes
    print(f"Iteration {iteration+1}, Total Cost: {-result.fun}")

# After the loop, you can use the final policy for deployment
comet_ml.init(project_name="MDGPS_log")
experiment = comet_ml.Experiment()
env = CometLogger(env, experiment)
num_episodes = 100
for episode in range(num_episodes):
    state, _  = env.reset()  # Reset the environment at the beginning of each episode
    total_reward = 0
    t = 0
    while True:
        action = np.random.multivariate_normal(policy_mean[t], policy_covariance, 1)
        print(action, t)
        t += 1
        # Perform the action and observe the next state and reward
        next_state, reward, done, _, _ = env.step(action[0])


        # Update the state for the next iteration
        state = next_state

        total_reward += reward

        if done:
            break


env.close()
experiment.end()

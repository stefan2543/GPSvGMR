import comet_ml
from comet_ml.integration.gymnasium import CometLogger
from stable_baselines3 import A2C
import gymnasium as gym
from pyvirtualdisplay import Display

display = Display(visible=0, size=(1400, 900))
display.start()
#comet_ml.init(project_name="A2C_log_test")
env = gym.make('InvertedDoublePendulum-v4', render_mode='human')
#experiment = comet_ml.Experiment()
#state_space = env.observation_space
#experiment.log_parameter("state_space", state_space)
#action_space = env.action_space
#experiment.log_parameter("action_space", action_space)
#env = CometLogger(env, experiment)

model = A2C("MlpPolicy", env, verbose=0)
#model.learn(total_timesteps=1000)
num_episodes = 1

for episode in range(num_episodes):
    state, _  = env.reset()  # Reset the environment at the beginning of each episode
    total_reward = 0

    while True:
        #env.render()
        action, _ = model.predict(state, deterministic=True)
        # Perform the action and observe the next state and reward
        next_state, reward, done, _, _ = env.step(action)

        # Log state and action for each timestep
        #experiment.log_other("state", state)
        #experiment.log_other("action", action)

        # Update the state for the next iteration
        state = next_state

        total_reward += reward

        if done:
            # Log total reward for the episode
            #experiment.log_metric("total_reward", total_reward, step=episode)
            break


env.close()
#experiment.end()
display.stop()

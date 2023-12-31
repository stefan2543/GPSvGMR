import comet_ml
from comet_ml.integration.gymnasium import CometLogger
from stable_baselines3 import A2C
import gymnasium as gym

comet_ml.init(project_name="A2C_log_train")
env = gym.make('InvertedDoublePendulum-v4')
experiment = comet_ml.Experiment()
env = CometLogger(env, experiment)

#model = A2C("MlpPolicy", env, verbose=0)
model = A2C.load("/work/06970/scr2543/ls6/gym/models/a2c2")
model.set_env(env)
model.learn(total_timesteps=100000)
model.save("/work/06970/scr2543/ls6/gym/models/a2c2")
#num_episodes = 100

#for episode in range(num_episodes):
    #state, _  = env.reset()  # Reset the environment at the beginning of each episode
    #total_reward = 0

    #while True:
        #action, _ = model.predict(state, deterministic=True)

        # Perform the action and observe the next state and reward
        #next_state, reward, done, _, _ = env.step(action)

        # Log state and action for each timestep
        #experiment.log_other("state", state)
        #experiment.log_other("action", action)

        # Update the state for the next iteration
        #state = next_state

        #total_reward += reward

        #if done:
            # Log total reward for the episode
            #experiment.log_metric("total_reward", total_reward, step=episode)
            #break


env.close()
experiment.end()

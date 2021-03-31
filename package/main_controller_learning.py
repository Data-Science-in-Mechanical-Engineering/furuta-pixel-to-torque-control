from gym_brt.data.config.configuration import FREQUENCY
from statetoinput.ppo_learner import PPOLearner

def learn_from_encoder():
    ############################################
    # Train a PPO RL agent from built in encoders
    model_id = -1 # specify a model from data/statetoinput or -1 to start a new one

    env = "QubeBeginDownEnv"
    # QubeBalanceEnv or QubeSwingupEnv for normal states: [alpha, theta, alpha_dot, theta_dot]
    # QubeBeginUpEnv or QubeBeginDownEnv for xy states: [cos(alpha), sin(alpha), cos(theta), sin(theta), alpha_dot, theta_dot]

    n_steps = 2048 # number of steps per episode
    batch_size = 256 # batch size for learning
    episodes = 1000 # number of episodes to be trained
    net_arch = [64, dict(vf=[64, 12], pi=[64, 12])] # specify a network architecture
    ############################################

    with PPOLearner(model_id, env, False, 100, n_steps, batch_size, simulation_mode='', vision_model_data_id=-1,
                    vision_model_model_name='') as learner:
        learner.train(n_steps * episodes, net_arch=net_arch, save_interval=n_steps)

def learn_from_se():
    ############################################
    # Train a PPO RL agent based on a specified state estimator
    model_id = -1 # specify a model from data/statetoinput or -1 to start a new one

    env = "VtSQubeBeginDownEnv"
    vision_model_data_id = 0 # specifiy dataID from data/visiontostate
    vision_model_model_name = 'model000' # specifiy model

    n_steps = 2048 # number of steps per episode
    batch_size = 256 # batch size for learning
    episodes = 10000 # number of episodes to be trained
    net_arch = [64, dict(vf=[64, 12], pi=[64, 12])] # specify a network architecture
    ############################################

    with PPOLearner(model_id, env, False, 100, n_steps, batch_size, simulation_mode='', vision_model_data_id=vision_model_data_id,
                    vision_model_model_name=vision_model_model_name) as learner:
        learner.train(n_steps * episodes, net_arch=net_arch, save_interval=n_steps)

def predict_from_enocders():
    ############################################
    # Run a trained PPO RL agent based on encoder states
    model_id = 0 # specify a model from data/statetoinput

    env = "QubeBeginDownEnv" # same as trained on

    episodes = 10 # number of episodes to run
    duration = 20 # duration of one episode in sec
    net_arch = [64, dict(vf=[64, 12], pi=[64, 12])] # network architecture the agent was trained on
    ############################################

    with PPOLearner(model_id, env, False, FREQUENCY, 2048, 256, simulation_mode='', vision_model_data_id=-1,
                    vision_model_model_name='') as learner:
        for _ in range(episodes):
            learner.predict(duration * FREQUENCY, net_arch=net_arch)

def predict_from_se():
    ############################################
    # Run a trained PPO RL agent based on a specified state estimator
    model_id = 9 # specify a model from data/statetoinput or -1 to start a new one

    env = "VtSQubeBeginDownEnv"
    # QubeBalanceEnv or QubeSwingupEnv for normal states: [alpha, theta, alpha_dot, theta_dot]
    # QubeBeginUpEnv or QubeBeginDownEnv for xy states: [cos(alpha), sin(alpha), cos(theta), sin(theta), alpha_dot, theta_dot]
    vision_model_data_id = 32 # specifiy dataID from data/visiontostate
    vision_model_model_name = 'model009' # specifiy model

    episodes = 10 # number of episodes to run
    duration = 40 # duration of one episode in sec
    net_arch = [64, dict(vf=[64, 12], pi=[64, 12])] # network architecture the agent was trained on
    ############################################

    with PPOLearner(model_id, env, False, FREQUENCY, 2048, 256, simulation_mode='', vision_model_data_id=vision_model_data_id,
                    vision_model_model_name=vision_model_model_name) as learner:
        for _ in range(episodes):
            learner.predict(duration * FREQUENCY, net_arch=net_arch)


if __name__ == '__main__':
    learn_from_encoder()


import argparse

"""
Here are the param for the training

"""


def get_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for single agent environments")
    # Environment
    parser.add_argument("--scenario-name", type=str, default="cart_pole", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=50000, help="maximum episode length")
    parser.add_argument("--time-steps", type=int, default=10, help="number of time steps")
    # Core training parameters
    parser.add_argument("--lr-actor", type=float, default=1e-4, help="learning rate of actor")
    parser.add_argument("--lr-critic", type=float, default=1e-3, help="learning rate of critic")
    parser.add_argument("--epsilon", type=float, default=0.1, help="epsilon greedy")
    parser.add_argument("--noise_rate", type=float, default=0.5, help="noise rate for sampling from a standard normal distribution")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--tau", type=float, default=0.01, help="parameter for updating the target network")
    parser.add_argument("--buffer-size", type=int, default=int(1e5), help="number of transitions can be stored in buffer")
    parser.add_argument("--batch-size", type=int, default=65, help="number of episodes to optimize at the same time")
    parser.add_argument("--learn_rate", type=int, default=4, help="number of episodes to optimize at the same time")
    # Checkpointing
    parser.add_argument("--save-dir", type=str, default="model", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=20, help="save model once every time this many episodes are completed")
    parser.add_argument("--model-dir", type=str, default="", help="directory in which training state and model are loaded")

    # Evaluate
    parser.add_argument("--evaluate-episodes", type=int, default=500, help="number of episodes for evaluating")
    parser.add_argument("--evaluate-episode-len", type=int, default=200, help="length of episodes for evaluating")
    parser.add_argument("--evaluate", type=bool, default=1, help="whether to evaluate the model")
    parser.add_argument("--evaluate-rate", type=int, default=100, help="how often to evaluate model")

    args = parser.parse_args()

    return args

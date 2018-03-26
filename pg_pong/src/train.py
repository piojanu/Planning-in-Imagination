import argparse

def parse_arguments():
    argparser = argparse.ArgumentParser(description='Script for training an agent to play Pong '
                                                    'game using the Policy Gradient method.')
    argparser.add_argument('-hs', '--hidden_size', type=int, default=50,
                           help='Number of neurons in hidden layer.')
    argparser.add_argument('-bs', '--batch_size', type=int, default=4,
                           help='Number of episodes in batch.')
    argparser.add_argument('-lr', '--learning_rate', type=float, default=1e-4,
                           help='Learning rate.')
    argparser.add_argument('-d', '--discount', type=int, default=0.99,
                           help='Discount factor for reward.')
    argparser.add_argument('-r', '--render', action='store_true', default=False,
                           help='If True, Pong\'s board is displayed')
    return argparser.parse_args()


def main():
    args = parse_arguments()
    print(f'Arguments: {args}')


if __name__ == "__main__":
    main()
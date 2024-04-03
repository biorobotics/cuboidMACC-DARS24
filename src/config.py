import argparse


def get_parser():
    parser = argparse.ArgumentParser(description='MACC Environment')

    '''Environment'''
    parser.add_argument('--size', type=int, default=16, help='Size of the environment')
    parser.add_argument('--height', type=int, default=8, help='Height of the environment')
    parser.add_argument('--input',  type=str, help='Terrain Filename', required=False)
    parser.add_argument('--output', type=str, help='Goal Structure Filename', required=True)

    return parser

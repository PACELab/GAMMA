import sys

sys.path.append("..")

from client import Environment
from aimd import AIMD

if __name__=="__main__":
    # environment for getting states and peforming actions
    env = Environment()

    # init agent
    agent = AIMD(env)
    
    agent.start()

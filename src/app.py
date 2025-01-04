import helpers.utils as utils
from core.model_tester import test, ann
from core.model_trainer import train

if __name__ == "__main__":
    device = utils.get_device()
    train(device)
    test(device)
    ann('1798年')

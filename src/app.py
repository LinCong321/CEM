import utils
from core.model_tester import test
from core.model_trainer import train

if __name__ == "__main__":
    device = utils.get_device()
    train(device)
    test(device)

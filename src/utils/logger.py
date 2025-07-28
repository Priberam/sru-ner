import logging


# Taken from
# https://hackernoon.com/yet-another-lightning-hydra-template-for-ml-experiments
def get_pylogger(name=__name__) -> logging.Logger:
    logger = logging.getLogger(name)
    return logger

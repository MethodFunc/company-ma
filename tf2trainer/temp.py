import logging.config

logging.config.fileConfig('logging.conf')
logger = logging.getLogger('dataset_maker')

n = 800
s = 100

if n < s:
    logger.warning(f'n is smaller and s.')
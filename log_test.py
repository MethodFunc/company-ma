import logging
import logging.config

logging.config.fileConfig("test_config.conf")
logger = logging.getLogger("test")

# logger = logging.getLogger()
#
# logger.setLevel(logging.INFO)
#
# formatter = logging.Formatter("%(asctime)s %(name)s : %(levelname)s %(message)s")
#
# stream_handler = logging.StreamHandler()
# stream_handler.setFormatter(formatter)
# logger.addHandler(stream_handler)
#
#
# file_handler = logging.FileHandler("test.log")
# file_handler.setFormatter(formatter)
# logger.addHandler(file_handler)
#
#
for i in range(20):
    logger.info(f"{i} logs test")
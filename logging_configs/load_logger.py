import json
import logging
from logging.config import dictConfig


LOG_CONF = "logging.json"


def configure_logging(logging_config):
    with open(logging_config) as log_file_conf:
        json_conf = json.load(log_file_conf)
    logging.config.dictConfig(json_conf)

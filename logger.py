import logging

file_handler = logging.FileHandler('log.txt', 'w', 'utf-8')
console_handler = logging.StreamHandler()

logging.basicConfig(
    handlers=(file_handler, console_handler),
    format='%(message)s',
    level=logging.INFO,
)

logger = logging.getLogger(__name__)


def log(function):
    def wrapper(*args, **kwargs):
        function_result = function(*args, **kwargs)
        logger.info(f'{function.__name__}: {function_result}')
        return function_result

    return wrapper

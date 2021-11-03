import logging

file_handler = logging.FileHandler('log.txt', 'w', 'utf-8')
console_handler = logging.StreamHandler()

logging.basicConfig(
    handlers=(file_handler, console_handler),
    format='%(message)s',
    level=logging.INFO,
)

logger = logging.getLogger(__name__)


def log(message):
    def decorator(function):
        def wrapper(*args, **kwargs):
            function_result = function(*args, **kwargs)
            logger.info(message.format(args=args, kwargs=kwargs, result=function_result))
            return function_result
        return wrapper
    return decorator

import logging

file_handler = logging.FileHandler('log.txt', 'w', 'utf-8')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(message)s'))

console_handler = logging.StreamHandler()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)


def log(message):
    def decorator(function):
        def wrapper(*args, **kwargs):
            function_result = function(*args, **kwargs)
            logger.info(message.format(args=args, kwargs=kwargs, result=function_result))
            return function_result
        return wrapper
    return decorator

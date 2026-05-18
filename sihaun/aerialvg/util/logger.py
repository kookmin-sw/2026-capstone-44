try:
    from groundingdino.util.logger import *  # noqa: F401,F403
except ModuleNotFoundError:
    import logging

    def setup_logger(output=None, distributed_rank=0, color=False, name="aerialvg"):
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
            logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return logger

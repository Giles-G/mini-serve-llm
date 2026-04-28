"""日志模块

提供统一的 logger 工厂函数，全局配置日志格式。
"""

import logging


def get_logger(name: str) -> logging.Logger:
    """获取一个带统一格式的 Logger 实例

    Args:
        name: logger 名称，通常用 __name__

    Returns:
        配置好的 Logger 实例
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    return logging.getLogger(name)

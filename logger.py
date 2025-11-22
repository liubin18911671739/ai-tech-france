"""
统一日志管理
"""
import sys
from pathlib import Path
from loguru import logger
from config import config


def setup_logger(
    log_level: str = None,
    log_file: str = None,
    rotation: str = "100 MB",
    retention: str = "10 days"
):
    """
    配置全局日志
    
    Args:
        log_level: 日志级别 (DEBUG/INFO/WARNING/ERROR)
        log_file: 日志文件路径,None则只输出到控制台
        rotation: 日志轮转大小
        retention: 日志保留时间
    """
    # 移除默认handler
    logger.remove()
    
    # 日志级别
    level = log_level or config.LOG_LEVEL
    
    # 控制台输出
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=level,
        colorize=True
    )
    
    # 文件输出
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level=level,
            rotation=rotation,
            retention=retention,
            encoding="utf-8"
        )
    
    return logger


# 默认logger实例
default_logger = setup_logger()


def get_logger(name: str = None, log_file: str = None):
    """
    获取logger实例
    
    Args:
        name: logger名称
        log_file: 可选的日志文件
    
    Returns:
        logger实例
    """
    if log_file:
        return setup_logger(log_file=log_file)
    
    if name:
        return logger.bind(name=name)
    
    return logger


if __name__ == "__main__":
    # 测试日志
    test_logger = get_logger("test_module")
    
    test_logger.debug("这是调试信息")
    test_logger.info("这是普通信息")
    test_logger.warning("这是警告信息")
    test_logger.error("这是错误信息")
    
    # 测试文件日志
    file_logger = get_logger("file_test", log_file="logs/test.log")
    file_logger.info("写入文件的日志")

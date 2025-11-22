"""
消融实验模块

系统化评测不同组件对检索性能的影响
"""
from .run_ablation import AblationExperiment

__all__ = ["AblationExperiment"]

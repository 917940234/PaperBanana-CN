# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
统一日志配置 — 为 PaperBanana 提供一致的日志格式和级别管理。
"""

import logging
import sys


def setup_logging(level: str = "INFO") -> None:
    """
    配置全局日志格式和级别。

    Args:
        level: 日志级别（DEBUG, INFO, WARNING, ERROR）
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    # 使用 UTF-8 编码的 StreamHandler，避免 Windows 终端编码问题
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(formatter)

    # 配置根 logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # 避免重复添加 handler
    if not root_logger.handlers:
        root_logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """
    获取指定名称的 logger。

    Args:
        name: logger 名称，通常使用模块名（如 "PlannerAgent"）

    Returns:
        配置好的 Logger 实例
    """
    return logging.getLogger(name)

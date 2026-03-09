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
绘图代码执行器 — 在子进程中安全执行 matplotlib 代码并返回 base64 图片。
"""

import base64
import io
import re
import logging

import matplotlib.pyplot as plt

logger = logging.getLogger("PlotExecutor")


def execute_plot_code(code_text: str, dpi: int = 300) -> str | None:
    """
    在独立进程中执行绘图代码并返回 JPEG base64 字符串。

    步骤：
    1. 从 markdown 代码块中提取 Python 代码
    2. 执行绘图
    3. 以 JPEG 格式返回 base64 编码的图片

    Args:
        code_text: 包含 matplotlib 绘图代码的文本（可含 ```python 标记）
        dpi: 输出图像的 DPI，默认 300

    Returns:
        base64 编码的 JPEG 图片字符串，失败时返回 None
    """
    match = re.search(r"```python(.*?)```", code_text, re.DOTALL)
    code_clean = match.group(1).strip() if match else code_text.strip()

    plt.switch_backend("Agg")
    plt.close("all")
    plt.rcdefaults()

    try:
        exec_globals = {}
        exec(code_clean, exec_globals)

        if plt.get_fignums():
            buf = io.BytesIO()
            plt.savefig(buf, format="jpeg", bbox_inches="tight", dpi=dpi)
            plt.close("all")

            buf.seek(0)
            img_bytes = buf.read()
            return base64.b64encode(img_bytes).decode("utf-8")
        else:
            return None

    except Exception as e:
        logger.error(f"❌ 执行绘图代码出错: {e}")
        return None

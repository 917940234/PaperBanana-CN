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
PaperBanana 并行 Streamlit 演示
接受用户文本输入，复制 10 份，并行处理以生成多个图表候选方案供比较。
"""

import streamlit as st
import asyncio
import math
import base64
import json
import time
import re
import html
from io import BytesIO
from PIL import Image
from pathlib import Path
import sys
import os
from datetime import datetime
from typing import Callable, Optional

# 将项目根目录添加到路径
sys.path.insert(0, str(Path(__file__).parent))

print("调试：正在导入代理模块...")
try:
    from agents.planner_agent import PlannerAgent
    print("调试：已导入 PlannerAgent")
    from agents.visualizer_agent import VisualizerAgent
    from agents.stylist_agent import StylistAgent
    from agents.critic_agent import CriticAgent
    from agents.retriever_agent import RetrieverAgent
    from agents.vanilla_agent import VanillaAgent
    from agents.polish_agent import PolishAgent
    print("调试：已导入所有代理模块")
    from utils import config
    from utils.paperviz_processor import PaperVizProcessor
    print("调试：已导入工具模块")

    import yaml
    config_path = Path(__file__).parent / "configs" / "model_config.yaml"
    model_config_data = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            model_config_data = yaml.safe_load(f) or {}

    def get_config_val(section, key, env_var, default=""):
        val = os.getenv(env_var)
        if not val and section in model_config_data:
            val = model_config_data[section].get(key)
        return val or default

except ImportError as e:
    print(f"调试：导入错误：{e}")
    import traceback
    traceback.print_exc()
    raise e
except Exception as e:
    print(f"调试：导入过程中发生异常：{e}")
    import traceback
    traceback.print_exc()
    raise e

st.set_page_config(
    layout="wide",
    page_title="PaperBanana 并行演示",
    page_icon="🍌"
)

def clean_text(text):
    """清理文本，移除无效的 UTF-8 代理字符。"""
    if not text:
        return text
    if isinstance(text, str):
        # 移除导致 UnicodeEncodeError 的代理字符
        return text.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
    return text

def base64_to_image(b64_str):
    """将 base64 字符串转换为 PIL 图像。"""
    if not b64_str:
        return None
    try:
        if "," in b64_str:
            b64_str = b64_str.split(",")[1]
        image_data = base64.b64decode(b64_str)
        return Image.open(BytesIO(image_data))
    except Exception:
        return None


def safe_log_text(value, max_len=2000):
    """将任意日志文本转换为可安全打印的字符串。"""
    try:
        text = value if isinstance(value, str) else str(value)
    except Exception:
        text = repr(value)
    text = text.replace("\x00", "\\x00")
    try:
        safe = text.encode("utf-8", errors="backslashreplace").decode("utf-8", errors="ignore")
    except Exception:
        safe = repr(text)
    if len(safe) > max_len:
        return safe[:max_len] + f"...(truncated {len(safe)-max_len} chars)"
    return safe


def get_refine_request_timeout_seconds(provider: str) -> float:
    """获取精修单次请求超时（秒），避免某次 API 请求无限挂起。"""
    env_val = os.getenv("REFINE_REQUEST_TIMEOUT_SEC", "").strip()
    if env_val:
        try:
            return max(float(env_val), 30.0)
        except ValueError:
            pass
    if provider == "gemini":
        return 240.0
    return 180.0


def extract_retry_delay_seconds(error_text: str) -> Optional[float]:
    """从错误文本中提取建议重试等待时间（秒）。"""
    if not error_text:
        return None
    lowered = error_text.lower()

    patterns = [
        r"retry in\s*([0-9]+(?:\.[0-9]+)?)s",
        r"retrydelay['\"]?\s*[:=]\s*['\"]?([0-9]+(?:\.[0-9]+)?)s",
    ]
    for pattern in patterns:
        match = re.search(pattern, lowered)
        if match:
            try:
                return max(float(match.group(1)), 1.0)
            except ValueError:
                continue
    return None


def emit_refine_status(status_callback: Optional[Callable[[str], None]], message: str):
    """向 UI 发送精修实时状态。"""
    if not status_callback or not message:
        return
    try:
        status_callback(message)
    except Exception as cb_error:
        try:
            print(f"[DEBUG] [WARN] 精修状态回调失败: {safe_log_text(cb_error)}")
        except Exception:
            pass


def normalize_image_mime_type(mime_type: Optional[str]) -> str:
    """将上传 MIME 归一化为 Gemini/Evolink 可接受值。"""
    if not mime_type:
        return "image/png"
    lowered = mime_type.strip().lower()
    if lowered in ("image/jpg", "image/pjpeg"):
        return "image/jpeg"
    if lowered in ("image/jpeg", "image/png"):
        return lowered
    return "image/png"


COMMON_ASPECT_RATIOS = [
    "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"
]


GEMINI_TEXT_MODELS = [
    "gemini-3.1-pro-preview",
    "gemini-3.1-flash-lite-preview",
    "gemini-3-flash-preview",
]


def create_sample_inputs(method_content, caption, diagram_type="Pipeline", aspect_ratio="16:9", num_copies=10, max_critic_rounds=3, image_resolution="2K"):
    """创建多份输入数据副本用于并行处理。"""
    base_input = {
        "filename": "demo_input",
        "caption": caption,
        "content": method_content,
        "visual_intent": caption,
        "additional_info": {
            "rounded_ratio": aspect_ratio,
            "image_resolution": image_resolution  # 添加图像分辨率参数
        },
        "max_critic_rounds": max_critic_rounds  # 添加评审轮次控制
    }

    # 创建 num_copies 份相同的输入，每份带有唯一标识符
    inputs = []
    for i in range(num_copies):
        input_copy = base_input.copy()
        input_copy["filename"] = f"demo_input_candidate_{i}"
        input_copy["candidate_id"] = i
        inputs.append(input_copy)

    return inputs

def compute_effective_concurrency(concurrency_mode: str, max_concurrent: int, total_candidates: int) -> int:
    """计算有效并发数（不改变业务逻辑，仅控制并发上限）。"""
    safe_max = max(1, int(max_concurrent))
    safe_total = max(1, int(total_candidates))

    if concurrency_mode == "auto":
        return min(safe_max, safe_total)
    return min(safe_max, safe_total)


async def process_parallel_candidates(
    data_list,
    exp_mode="dev_planner_critic",
    retrieval_setting="auto",
    model_name="",
    image_model_name="",
    provider="evolink",
    api_key="",
    concurrency_mode="auto",
    max_concurrent=20,
    progress_callback: Optional[Callable[[int, int, int], None]] = None,
    status_callback: Optional[Callable[[str], None]] = None,
):
    """使用 PaperVizProcessor 并行处理多个候选方案。"""
    total_candidates = len(data_list)
    effective_concurrent = compute_effective_concurrency(
        concurrency_mode=concurrency_mode,
        max_concurrent=max_concurrent,
        total_candidates=total_candidates,
    )

    print(f"\n{'='*60}")
    print(f"[DEBUG] process_parallel_candidates 开始")
    print(f"[DEBUG]   provider={provider}, model={model_name}, image_model={image_model_name}")
    print(f"[DEBUG]   exp_mode={exp_mode}, retrieval={retrieval_setting}, candidates={total_candidates}")
    print(f"[DEBUG]   concurrency_mode={concurrency_mode}, max_concurrent={max_concurrent}, effective={effective_concurrent}")
    print(f"[DEBUG]   api_key={'已设置 (' + api_key[:8] + '...)' if api_key else '未设置'}")
    print(f"{'='*60}")

    if progress_callback:
        try:
            progress_callback(0, total_candidates, effective_concurrent)
        except Exception as cb_error:
            print(f"[DEBUG] [WARN] 进度回调失败(初始化): {cb_error}")
    if status_callback:
        try:
            status_callback(
                f"任务启动：候选数={total_candidates}, 并发={effective_concurrent}, 流水线={exp_mode}"
            )
        except Exception as cb_error:
            print(f"[DEBUG] [WARN] 状态回调失败(初始化): {cb_error}")

    # 使用界面传入的 API Key 初始化 Provider
    from utils import generation_utils

    if api_key:
        if provider == "evolink":
            generation_utils.init_evolink_provider(api_key)
        elif provider == "gemini":
            generation_utils.init_gemini_client(api_key)
    else:
        print(f"[DEBUG] [WARN] 未提供 API Key，Provider 可能无法正常工作")

    # 创建实验配置
    exp_config = config.ExpConfig(
        dataset_name="Demo",
        split_name="demo",
        exp_mode=exp_mode,
        retrieval_setting=retrieval_setting,
        concurrency_mode=concurrency_mode,
        max_concurrent=int(max_concurrent),
        model_name=model_name,
        image_model_name=image_model_name,
        provider=provider,
        work_dir=Path(__file__).parent,
    )
    print(f"[DEBUG] ExpConfig 已创建: provider={exp_config.provider}, model={exp_config.model_name}, image_model={exp_config.image_model_name}")

    # 初始化处理器及所有代理
    processor = PaperVizProcessor(
        exp_config=exp_config,
        vanilla_agent=VanillaAgent(exp_config=exp_config),
        planner_agent=PlannerAgent(exp_config=exp_config),
        visualizer_agent=VisualizerAgent(exp_config=exp_config),
        stylist_agent=StylistAgent(exp_config=exp_config),
        critic_agent=CriticAgent(exp_config=exp_config),
        retriever_agent=RetrieverAgent(exp_config=exp_config),
        polish_agent=PolishAgent(exp_config=exp_config),
    )

    # 并行处理所有候选方案（并发量由处理器控制）
    results = []
    concurrent_num = effective_concurrent

    try:
        generation_utils.set_runtime_status_hook(status_callback)
        async for result_data in processor.process_queries_batch(
            data_list,
            max_concurrent=concurrent_num,
            do_eval=False,
            status_callback=status_callback,
        ):
            results.append(result_data)
            if progress_callback:
                try:
                    progress_callback(len(results), total_candidates, effective_concurrent)
                except Exception as cb_error:
                    print(f"[DEBUG] [WARN] 进度回调失败(更新): {cb_error}")
    finally:
        generation_utils.set_runtime_status_hook(None)
        # 关闭 Evolink Provider 的共享 session，避免资源泄漏
        if generation_utils.evolink_provider and hasattr(generation_utils.evolink_provider, 'close'):
            await generation_utils.evolink_provider.close()

    return results, effective_concurrent

async def refine_image_with_nanoviz(
    image_bytes,
    edit_prompt,
    aspect_ratio="21:9",
    image_size="2K",
    api_key="",
    provider="evolink",
    task_id: int = 1,
    status_callback: Optional[Callable[[str], None]] = None,
    input_mime_type: str = "image/png",
):
    """
    使用图像编辑 API 精修图像，支持 Evolink 和 Gemini 两种 Provider。

    参数：
        image_bytes: 图像字节数据
        edit_prompt: 描述所需修改的文本
        aspect_ratio: 输出宽高比 (21:9, 16:9, 3:2)
        image_size: 输出分辨率 (2K 或 4K)
        api_key: API 密钥
        provider: "evolink" 或 "gemini"

    返回：
        元组 (编辑后的图像字节数据, 成功消息)
    """
    from utils import generation_utils

    attempt = 0
    sleep_seconds = 2.0
    timeout_seconds = get_refine_request_timeout_seconds(provider)
    task_prefix = f"task#{task_id}"
    normalized_mime_type = normalize_image_mime_type(input_mime_type)

    # 精修功能：持续重试直到成功（按用户要求）
    while True:
        attempt += 1
        try:
            if provider == "gemini":
                # ====== Gemini 路径：多模态 API，直接传图片字节 ======
                if generation_utils.gemini_client is None:
                    await asyncio.sleep(min(sleep_seconds, 10.0))
                    sleep_seconds = min(sleep_seconds * 1.2, 15.0)
                    continue

                from google.genai import types

                contents = [
                    types.Part.from_text(text=edit_prompt),
                    types.Part.from_bytes(mime_type=normalized_mime_type, data=image_bytes),
                ]
                config = types.GenerateContentConfig(
                    temperature=1.0,
                    max_output_tokens=8192,
                    response_modalities=["IMAGE"],
                    image_config=types.ImageConfig(
                        aspect_ratio=aspect_ratio,
                        image_size=image_size,
                    ),
                )

                selected_model = st.session_state.get("tab1_image_model_name", "gemini-3-pro-image-preview")
                gemini_model_sequence = [selected_model]
                if selected_model != "gemini-3.1-flash-image-preview":
                    gemini_model_sequence.append("gemini-3.1-flash-image-preview")

                # 每 5 次失败切换一次模型（循环）
                model_index = ((attempt - 1) // 5) % len(gemini_model_sequence)
                image_model = gemini_model_sequence[model_index]

                emit_refine_status(
                    status_callback,
                    f"[精修][{task_prefix}] attempt={attempt} model={image_model} timeout={int(timeout_seconds)}s",
                )
                response = await asyncio.wait_for(
                    generation_utils.gemini_client.aio.models.generate_content(
                        model=image_model,
                        contents=contents,
                        config=config,
                    ),
                    timeout=timeout_seconds,
                )

                if response and response.candidates and response.candidates[0].content.parts:
                    for part in response.candidates[0].content.parts:
                        if hasattr(part, "inline_data") and part.inline_data:
                            edited_image_data = part.inline_data.data
                            if isinstance(edited_image_data, bytes) and edited_image_data:
                                emit_refine_status(
                                    status_callback,
                                    f"[精修][{task_prefix}] success on attempt={attempt} model={image_model}",
                                )
                                return edited_image_data, f"✅ 图像精修成功！（第 {attempt} 次尝试）"
                            if isinstance(edited_image_data, str) and edited_image_data:
                                emit_refine_status(
                                    status_callback,
                                    f"[精修][{task_prefix}] success on attempt={attempt} model={image_model}",
                                )
                                return base64.b64decode(edited_image_data), f"✅ 图像精修成功！（第 {attempt} 次尝试）"

                raise RuntimeError("Gemini 未返回有效图像数据")

            else:
                # ====== Evolink 路径：上传图片获取 URL → image_urls ======
                if generation_utils.evolink_provider is None:
                    await asyncio.sleep(min(sleep_seconds, 10.0))
                    sleep_seconds = min(sleep_seconds * 1.2, 15.0)
                    continue

                image_model = st.session_state.get("tab1_image_model_name", "nano-banana-2-lite")
                emit_refine_status(
                    status_callback,
                    f"[精修][{task_prefix}] attempt={attempt} model={image_model} timeout={int(timeout_seconds)}s",
                )

                # 步骤 1：上传原始图片到 Evolink 文件服务
                image_b64 = base64.b64encode(image_bytes).decode("utf-8")
                ref_image_url = await generation_utils.upload_image_to_evolink(
                    image_b64,
                    media_type=normalized_mime_type,
                )
                try:
                    print(f"[精修] attempt={attempt}, uploaded_ref={safe_log_text(ref_image_url[:80])}...")
                except Exception:
                    pass

                # 步骤 2：图像生成 API（传入参考图 URL）
                result = await asyncio.wait_for(
                    generation_utils.evolink_provider.generate_image(
                        model_name=image_model,
                        prompt=edit_prompt,
                        aspect_ratio=aspect_ratio,
                        quality=image_size,
                        image_urls=[ref_image_url],
                        max_attempts=1,
                        retry_delay=3,
                    ),
                    timeout=timeout_seconds,
                )

                if result and result[0] and result[0] != "Error":
                    edited_image_data = base64.b64decode(result[0])
                    emit_refine_status(
                        status_callback,
                        f"[精修][{task_prefix}] success on attempt={attempt} model={image_model}",
                    )
                    return edited_image_data, f"✅ 图像精修成功！（第 {attempt} 次尝试）"

                raise RuntimeError("Evolink 未返回有效图像数据")

        except asyncio.TimeoutError:
            err_text = (
                f"{provider} request timed out after {int(timeout_seconds)}s "
                f"(attempt={attempt}, {task_prefix})"
            )
            delay = min(max(sleep_seconds, 3.0), 20.0)
            emit_refine_status(
                status_callback,
                f"[精修][{task_prefix}] timeout, wait {delay:.1f}s then retry",
            )
            try:
                print(f"[精修][重试] {safe_log_text(err_text, max_len=1200)}")
            except Exception:
                pass
            await asyncio.sleep(delay)
            sleep_seconds = min(max(delay * 1.2, sleep_seconds * 1.25), 30.0)
        except Exception as e:
            # 不向前端抛错，持续重试直到成功
            error_text = safe_log_text(e, max_len=2000)
            lower_error = error_text.lower()

            # Windows 套接字异常自愈：重建 client/provider 后继续。
            if "winerror 10038" in lower_error:
                try:
                    if provider == "gemini" and api_key:
                        generation_utils.init_gemini_client(api_key)
                    elif provider == "evolink" and api_key:
                        generation_utils.init_evolink_provider(api_key)
                except Exception as reinit_error:
                    try:
                        print(f"[精修][WARN] socket 自愈重建失败: {safe_log_text(reinit_error)}")
                    except Exception:
                        pass

            suggested_delay = extract_retry_delay_seconds(error_text)
            delay = min(sleep_seconds, 20.0)
            if suggested_delay is not None:
                delay = min(max(delay, suggested_delay), 60.0)
            if "limit: 0" in lower_error:
                delay = max(delay, 30.0)
            emit_refine_status(
                status_callback,
                f"[精修][{task_prefix}] failed attempt={attempt}, wait {delay:.1f}s | {error_text[:160]}",
            )
            try:
                print(f"[精修][重试] attempt={attempt}, err={error_text}")
            except Exception:
                pass
            await asyncio.sleep(delay)
            sleep_seconds = min(max(delay * 1.1, sleep_seconds * 1.25), 30.0)


async def refine_images_with_count(
    image_bytes,
    edit_prompt,
    num_images=3,
    aspect_ratio="21:9",
    image_size="2K",
    api_key="",
    provider="evolink",
    input_mime_type="image/png",
    progress_callback: Optional[Callable[[int, int], None]] = None,
    status_callback: Optional[Callable[[str], None]] = None,
):
    """并发精修多张图像，按完成先后收集，避免被单一慢任务阻塞观感。"""
    from utils import generation_utils

    safe_count = max(1, int(num_images))
    results = [None] * safe_count
    tasks = []

    # 每次精修批次只初始化一次 Provider/Client，避免并发重建导致底层连接异常。
    if provider == "gemini" and api_key:
        generation_utils.init_gemini_client(api_key)
    elif provider == "evolink" and api_key:
        generation_utils.init_evolink_provider(api_key)

    for idx in range(safe_count):
        variant_prompt = (
            f"{edit_prompt}\n\n"
            f"[Variant Request #{idx + 1}] Keep the semantics unchanged, "
            f"but provide a distinct visual variant."
        )
        async def run_one(task_idx=idx, prompt_text=variant_prompt):
            value = await refine_image_with_nanoviz(
                image_bytes=image_bytes,
                edit_prompt=prompt_text,
                aspect_ratio=aspect_ratio,
                image_size=image_size,
                api_key=api_key,
                provider=provider,
                task_id=task_idx + 1,
                status_callback=status_callback,
                input_mime_type=input_mime_type,
            )
            return task_idx, value

        task = asyncio.create_task(run_one())
        tasks.append(task)

    done_count = 0
    for future in asyncio.as_completed(tasks):
        task_idx, value = await future
        results[task_idx] = value
        done_count += 1
        if progress_callback:
            try:
                progress_callback(done_count, safe_count)
            except Exception as cb_error:
                try:
                    print(f"[DEBUG] [WARN] 精修进度回调失败: {safe_log_text(cb_error)}")
                except Exception:
                    pass
        emit_refine_status(
            status_callback,
            f"[精修] completed {done_count}/{safe_count} (task#{task_idx + 1})",
        )

    return [x for x in results if x is not None]


def get_evolution_stages(result, exp_mode):
    """从结果中提取所有演化阶段（图像和描述）。"""
    task_name = "diagram"
    stages = []

    # 阶段 1：规划器输出
    planner_img_key = f"target_{task_name}_desc0_base64_jpg"
    planner_desc_key = f"target_{task_name}_desc0"
    if planner_img_key in result and result[planner_img_key]:
        stages.append({
            "name": "📋 规划器",
            "image_key": planner_img_key,
            "desc_key": planner_desc_key,
            "description": "基于方法内容生成的初始图表规划"
        })

    # 阶段 2：风格化器输出（仅限 demo_full 模式）
    if exp_mode == "demo_full":
        stylist_img_key = f"target_{task_name}_stylist_desc0_base64_jpg"
        stylist_desc_key = f"target_{task_name}_stylist_desc0"
        if stylist_img_key in result and result[stylist_img_key]:
            stages.append({
                "name": "✨ 风格化器",
                "image_key": stylist_img_key,
                "desc_key": stylist_desc_key,
                "description": "经过风格优化的描述"
            })

    # 阶段 3+：评审迭代
    for round_idx in range(4):  # 检查最多 4 轮
        critic_img_key = f"target_{task_name}_critic_desc{round_idx}_base64_jpg"
        critic_desc_key = f"target_{task_name}_critic_desc{round_idx}"
        critic_sugg_key = f"target_{task_name}_critic_suggestions{round_idx}"

        if critic_img_key in result and result[critic_img_key]:
            stages.append({
                "name": f"🔍 评审第 {round_idx} 轮",
                "image_key": critic_img_key,
                "desc_key": critic_desc_key,
                "suggestions_key": critic_sugg_key,
                "description": f"根据评审反馈进行优化（第 {round_idx} 次迭代）"
            })

    return stages

def display_candidate_result(result, candidate_id, exp_mode):
    """展示单个候选方案的结果。"""
    task_name = "diagram"

    if isinstance(result, dict) and result.get("status") == "failed":
        st.error(f"候选方案 {candidate_id} 失败：{result.get('error', 'Unknown error')}")
        detail = result.get("error_detail")
        if detail:
            with st.expander("查看失败详情", expanded=False):
                st.code(clean_text(detail))
        return

    # 根据 exp_mode 决定展示哪张图像
    # 对于演示模式，始终尝试查找最后一轮评审结果
    final_image_key = None
    final_desc_key = None

    # 尝试查找最后一轮评审
    for round_idx in range(3, -1, -1):  # 检查第 3、2、1、0 轮
        image_key = f"target_{task_name}_critic_desc{round_idx}_base64_jpg"
        if image_key in result and result[image_key]:
            final_image_key = image_key
            final_desc_key = f"target_{task_name}_critic_desc{round_idx}"
            break

    # 如果没有完成评审轮次则使用备选方案
    if not final_image_key:
        if exp_mode == "demo_full":
            # demo_full 在可视化之前使用风格化器
            final_image_key = f"target_{task_name}_stylist_desc0_base64_jpg"
            final_desc_key = f"target_{task_name}_stylist_desc0"
        else:
            # demo_planner_critic 使用规划器输出
            final_image_key = f"target_{task_name}_desc0_base64_jpg"
            final_desc_key = f"target_{task_name}_desc0"

    # 展示最终图像
    if final_image_key and final_image_key in result:
        img = base64_to_image(result[final_image_key])
        if img:
            st.image(img, use_container_width=True, caption=f"候选方案 {candidate_id}（最终版）")

            # 添加下载按钮
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            st.download_button(
                label="[DOWN] 下载",
                data=buffered.getvalue(),
                file_name=f"candidate_{candidate_id}.png",
                mime="image/png",
                key=f"download_candidate_{candidate_id}",
                use_container_width=True
            )
        else:
            st.error(f"候选方案 {candidate_id} 的图像解码失败")
    else:
        st.warning(f"候选方案 {candidate_id} 未生成图像")

    # 在折叠面板中展示演化时间线
    stages = get_evolution_stages(result, exp_mode)
    if len(stages) > 1:
        with st.expander(f"🔄 查看演化时间线（{len(stages)} 个阶段）", expanded=False):
            st.caption("查看图表在不同流水线阶段的演化过程")

            for idx, stage in enumerate(stages):
                st.markdown(f"### {stage['name']}")
                st.caption(stage['description'])

                # 展示该阶段的图像
                stage_img = base64_to_image(result.get(stage['image_key']))
                if stage_img:
                    st.image(stage_img, use_container_width=True)

                # 展示描述
                if stage['desc_key'] in result:
                    with st.expander(f"📝 描述", expanded=False):
                        cleaned_desc = clean_text(result[stage['desc_key']])
                        st.write(cleaned_desc)

                # 展示评审建议（如有）
                if 'suggestions_key' in stage and stage['suggestions_key'] in result:
                    suggestions = result[stage['suggestions_key']]
                    with st.expander(f"💡 评审建议", expanded=False):
                        cleaned_sugg = clean_text(suggestions)
                        if cleaned_sugg.strip() == "No changes needed.":
                            st.success("✅ 无需修改——迭代已停止。")
                        else:
                            st.write(cleaned_sugg)

                # 在阶段之间添加分隔线（最后一个除外）
                if idx < len(stages) - 1:
                    st.divider()
    else:
        # 如果只有一个阶段，使用更简洁的折叠面板展示描述
        with st.expander(f"📝 查看描述", expanded=False):
            if final_desc_key and final_desc_key in result:
                # 清理文本，移除无效的 UTF-8 字符
                cleaned_desc = clean_text(result[final_desc_key])
                st.write(cleaned_desc)
            else:
                st.info("暂无描述")

def main():
    st.title("🍌 PaperBanana 演示")
    st.markdown("AI 驱动的科学图表生成与精修")

    # 创建选项卡
    tab1, tab2 = st.tabs(["📊 生成候选方案", "✨ 精修图像"])

    # ==================== 选项卡 1：生成候选方案 ====================
    with tab1:
        st.markdown("### 从您的方法章节和图注生成多个图表候选方案")

        # 侧边栏配置（选项卡 1）
        with st.sidebar:
            st.title("[SET] 生成设置")

            exp_mode = st.selectbox(
                "流水线模式",
                ["demo_planner_critic", "demo_full"],
                index=0,
                key="tab1_exp_mode",
                help="选择使用哪种代理流水线"
            )

            mode_info = {
                "demo_planner_critic": "规划器 → 可视化器 → 评审器 → 可视化器",
                "demo_full": "检索器 → 规划器 → 风格化器 → 可视化器 → 评审器 → 可视化器。（风格化器能让图表更具美感，但可能过度简化。建议两种模式都尝试并选择最佳结果）"
            }
            st.info(f"**流水线：** {mode_info[exp_mode]}")

            retrieval_setting = st.selectbox(
                "检索设置",
                ["auto", "auto-full", "random", "none"],
                index=0,
                key="tab1_retrieval_setting",
                help="如何检索参考图表",
                format_func=lambda x: {
                    "auto": "auto — LLM 智能选参考，仅 caption（~3万 tokens/候选）",
                    "auto-full": "auto-full — LLM 智能选参考，含完整论文（[WARN] ~80万 tokens/候选）",
                    "random": "random — 随机选 10 个参考（免费）",
                    "none": "none — 不检索参考（免费）",
                }[x],
            )

            _retrieval_cost_info = {
                "auto": "💡 轻量 auto：仅发送图注（caption）给 LLM 做匹配，每个候选约 **3 万 tokens**，性价比最高。",
                "auto-full": "[WARN] **注意**：完整 auto 将 200 篇参考论文的全文发给 LLM，每个候选消耗约 **80 万 tokens**。仅在需要高精度检索时使用。",
                "random": "✅ 随机从 298 篇参考中选 10 个，不调用 API，零费用。",
                "none": "✅ 跳过检索，不使用参考图表，零费用。",
            }
            st.info(_retrieval_cost_info[retrieval_setting])

            num_candidates = st.number_input(
                "候选方案数量",
                min_value=1,
                max_value=20,
                value=5,
                key="tab1_num_candidates",
                help="要并行生成多少个候选方案"
            )

            concurrency_mode = st.selectbox(
                "并发策略",
                ["auto", "manual"],
                index=0,
                key="tab1_concurrency_mode",
                help="auto：自动并发（默认）| manual：使用固定并发上限"
            )

            max_concurrent = st.number_input(
                "并发上限",
                min_value=1,
                max_value=100,
                value=20,
                step=1,
                key="tab1_max_concurrent",
                help="候选任务并发上限，默认 20"
            )

            effective_concurrency_preview = compute_effective_concurrency(
                concurrency_mode=concurrency_mode,
                max_concurrent=int(max_concurrent),
                total_candidates=int(num_candidates),
            )
            estimated_batches_preview = math.ceil(
                int(num_candidates) / max(1, effective_concurrency_preview)
            )

            with st.expander("📈 并发可视化调节", expanded=True):
                c1, c2 = st.columns(2)
                c1.metric("有效并发", effective_concurrency_preview)
                c2.metric("预计批次数", estimated_batches_preview)
                st.caption(
                    f"策略：{concurrency_mode} | 并发上限：{int(max_concurrent)} | 候选数：{int(num_candidates)}"
                )

            aspect_ratio = st.selectbox(
                "宽高比",
                COMMON_ASPECT_RATIOS,
                key="tab1_aspect_ratio",
                help="生成图表的宽高比"
            )

            provider_for_resolution = st.session_state.get("tab1_provider", "gemini")
            resolution_options = ["1K", "2K", "4K"] if provider_for_resolution == "gemini" else ["2K", "4K"]
            default_resolution = st.session_state.get("tab1_image_resolution", "2K")
            if default_resolution not in resolution_options:
                default_resolution = "2K" if "2K" in resolution_options else resolution_options[0]

            image_resolution = st.selectbox(
                "图像分辨率",
                resolution_options,
                index=resolution_options.index(default_resolution),
                key="tab1_image_resolution",
                help=f"生成图像的分辨率（当前 Provider 支持：{', '.join(resolution_options)}）"
            )

            max_critic_rounds = st.number_input(
                "最大评审轮次",
                min_value=1,
                max_value=5,
                value=3,
                key="tab1_max_critic_rounds",
                help="评审优化迭代的最大轮次"
            )

            # Provider 选择
            provider = st.selectbox(
                "API Provider",
                ["gemini", "evolink"],
                index=0,
                key="tab1_provider",
                help="gemini：Google 官方 API（需翻墙）| evolink：国内代理"
            )

            # Provider 对应的默认配置
            _provider_defaults = {
                "evolink": {
                    "api_key_label": "API Key",
                    "api_key_help": "Evolink API 密钥（Bearer Token）",
                    "api_key_default": get_config_val("evolink", "api_key", "EVOLINK_API_KEY", ""),
                    "model_name": "gemini-2.5-flash",
                    "image_model_name": "nano-banana-2-lite",
                },
                "gemini": {
                    "api_key_label": "Google API Key",
                    "api_key_help": "Google AI Studio API 密钥",
                    "api_key_default": get_config_val("api_keys", "google_api_key", "GOOGLE_API_KEY", ""),
                    "model_name": "gemini-3.1-pro-preview",
                    "image_model_name": "gemini-3-pro-image-preview",
                },
            }
            _pd = _provider_defaults[provider]

            # 首次加载时设置默认值
            if "tab1_api_key" not in st.session_state:
                st.session_state["tab1_api_key"] = _pd["api_key_default"]
            if "tab1_model_name" not in st.session_state:
                st.session_state["tab1_model_name"] = _pd["model_name"]
            if "tab1_image_model_name" not in st.session_state:
                st.session_state["tab1_image_model_name"] = _pd["image_model_name"]

            # 检测 provider 切换，重置模型名称
            if "prev_provider" not in st.session_state:
                st.session_state["prev_provider"] = provider
            if st.session_state["prev_provider"] != provider:
                st.session_state["prev_provider"] = provider
                st.session_state["tab1_model_name"] = _pd["model_name"]
                st.session_state["tab1_image_model_name"] = _pd["image_model_name"]
                st.session_state["tab1_api_key"] = _pd["api_key_default"]
                new_resolution_options = ["1K", "2K", "4K"] if provider == "gemini" else ["2K", "4K"]
                if st.session_state.get("tab1_image_resolution") not in new_resolution_options:
                    st.session_state["tab1_image_resolution"] = (
                        "2K" if "2K" in new_resolution_options else new_resolution_options[0]
                    )
                st.rerun()

            # API Key
            api_key = st.text_input(
                _pd["api_key_label"],
                type="password",
                key="tab1_api_key",
                help=_pd["api_key_help"]
            )

            # 文本模型
            if provider == "gemini":
                current_gemini_text_model = st.session_state.get(
                    "tab1_model_name",
                    GEMINI_TEXT_MODELS[0],
                )
                if current_gemini_text_model not in GEMINI_TEXT_MODELS:
                    current_gemini_text_model = GEMINI_TEXT_MODELS[0]

                model_name = st.selectbox(
                    "文本模型",
                    GEMINI_TEXT_MODELS,
                    index=GEMINI_TEXT_MODELS.index(current_gemini_text_model),
                    key="tab1_model_name",
                    help="用于推理/规划/评审的模型名称（可展开选择）"
                )
            else:
                model_name = st.text_input(
                    "文本模型",
                    key="tab1_model_name",
                    help="用于推理/规划/评审的模型名称"
                )

            # 图像模型
            if provider == "gemini":
                gemini_image_models = [
                    "gemini-3-pro-image-preview",
                    "gemini-3.1-flash-image-preview",
                ]
                current_gemini_image_model = st.session_state.get(
                    "tab1_image_model_name",
                    "gemini-3-pro-image-preview",
                )
                if current_gemini_image_model not in gemini_image_models:
                    current_gemini_image_model = "gemini-3-pro-image-preview"

                image_model_name = st.selectbox(
                    "图像模型",
                    gemini_image_models,
                    index=gemini_image_models.index(current_gemini_image_model),
                    key="tab1_image_model_name",
                    help="用于图像生成的模型名称（可展开选择）"
                )
            else:
                image_model_name = st.text_input(
                    "图像模型",
                    key="tab1_image_model_name",
                    help="用于图像生成的模型名称"
                )

        st.divider()

        # 输入区域
        st.markdown("## 📝 输入")

        # 示例内容
        example_method = r"""## Methodology: The PaperBanana Framework

        In this section, we present the architecture of PaperBanana, a reference-driven agentic framework for automated academic illustration. As illustrated in Figure \ref{fig:methodology_diagram}, PaperBanana orchestrates a collaborative team of five specialized agents—Retriever, Planner, Stylist, Visualizer, and Critic—to transform raw scientific content into publication-quality diagrams and plots. (See Appendix \ref{app_sec:agent_prompts} for prompts)

### Retriever Agent

Given the source context $S$ and the communicative intent $C$, the Retriever Agent identifies $N$ most relevant examples $\mathcal{E} = \{E_n\}_{n=1}^{N} \subset \mathcal{R}$ from the fixed reference set $\mathcal{R}$ to guide the downstream agents. As defined in Section \ref{sec:task_formulation}, each example $E_i \in \mathcal{R}$ is a triplet $(S_i, C_i, I_i)$.
To leverage the reasoning capabilities of VLMs, we adopt a generative retrieval approach where the VLM performs selection over candidate metadata:
$$
\mathcal{E} = \text{VLM}_{\text{Ret}} \left( S, C, \{ (S_i, C_i) \}_{E_i \in \mathcal{R}} \right)
$$
Specifically, the VLM is instructed to rank candidates by matching both research domain (e.g., Agent & Reasoning) and diagram type (e.g., pipeline, architecture), with visual structure being prioritized over topic similarity. By explicitly reasoned selection of reference illustrations $I_i$ whose corresponding contexts $(S_i, C_i)$ best match the current requirements, the Retriever provides a concrete foundation for both structural logic and visual style.

### Planner Agent

The Planner Agent serves as the cognitive core of the system. It takes the source context $S$, communicative intent $C$, and retrieved examples $\mathcal{E}$ as inputs. By performing in-context learning from the demonstrations in $\mathcal{E}$, the Planner translates the unstructured or structured data in $S$ into a comprehensive and detailed textual description $P$ of the target illustration:
$$
P = \text{VLM}_{\text{plan}}(S, C, \{ (S_i, C_i, I_i) \}_{E_i \in \mathcal{E}})
$$

### Stylist Agent

To ensure the output adheres to the aesthetic standards of modern academic manuscripts, the Stylist Agent acts as a design consultant.
A primary challenge lies in defining a comprehensive "academic style," as manual definitions are often incomplete.
To address this, the Stylist traverses the entire reference collection $\mathcal{R}$ to automatically synthesize an *Aesthetic Guideline* $\mathcal{G}$ covering key dimensions such as color palette, shapes and containers, lines and arrows, layout and composition, and typography and icons (see Appendix \ref{app_sec:auto_summarized_style_guide} for the summarized guideline and implementation details). Armed with this guideline, the Stylist refines each initial description $P$ into a stylistically optimized version $P^*$:
$$
P^* = \text{VLM}_{\text{style}}(P, \mathcal{G})
$$
This ensures that the final illustration is not only accurate but also visually professional.

### Visualizer Agent

After receiving the stylistically optimized description $P^*$, the Visualizer Agent collaborates with the Critic Agent to render academic illustrations and iteratively refine their quality. The Visualizer Agent leverages an image generation model to transform textual descriptions into visual output. In each iteration $t$, given a description $P_t$, the Visualizer generates:
$$
I_t = \text{Image-Gen}(P_t)
$$
where the initial description $P_0$ is set to $P^*$.

### Critic Agent

The Critic Agent forms a closed-loop refinement mechanism with the Visualizer by closely examining the generated image $I_t$ and providing refined description $P_{t+1}$ to the Visualizer. Upon receiving the generated image $I_t$ at iteration $t$, the Critic inspects it against the original source context $(S, C)$ to identify factual misalignments, visual glitches, or areas for improvement. It then provides targeted feedback and produces a refined description $P_{t+1}$ that addresses the identified issues:
$$
P_{t+1} = \text{VLM}_{\text{critic}}(I_t, S, C, P_t)
$$
This revised description is then fed back to the Visualizer for regeneration. The Visualizer-Critic loop iterates for $T=3$ rounds, with the final output being $I = I_T$. This iterative refinement process ensures that the final illustration meets the high standards required for academic dissemination.

### Extension to Statistical Plots

The framework extends to statistical plots by adjusting the Visualizer and Critic agents. For numerical precision, the Visualizer converts the description $P_t$ into executable Python Matplotlib code: $I_t = \text{VLM}_{\text{code}}(P_t)$. The Critic evaluates the rendered plot and generates a refined description $P_{t+1}$ addressing inaccuracies or imperfections: $P_{t+1} = \text{VLM}_{\text{critic}}(I_t, S, C, P_t)$. The same $T=3$ round iterative refinement process applies. While we prioritize this code-based approach for accuracy, we also explore direct image generation in Section \ref{sec:discussion}. See Appendix \ref{app_sec:plot_agent_prompt} for adjusted prompts."""

        example_caption = "Figure 1: Overview of our PaperBanana framework. Given the source context and communicative intent, we first apply a Linear Planning Phase to retrieve relevant reference examples and synthesize a stylistically optimized description. We then use an Iterative Refinement Loop (consisting of Visualizer and Critic agents) to transform the description into visual output and conduct multi-round refinements to produce the final academic illustration."

        col_input1, col_input2 = st.columns([3, 2])

        with col_input1:
            # 方法内容示例选择器
            method_example = st.selectbox(
                "加载示例（方法章节）",
                ["无", "PaperBanana 框架"],
                key="method_example_selector"
            )

            # 根据示例选择或会话状态设置值
            if method_example == "PaperBanana 框架":
                method_value = example_method
            else:
                method_value = st.session_state.get("method_content", "")

            method_content = st.text_area(
                "方法章节内容（建议使用 Markdown 格式）",
                value=method_value,
                height=250,
                placeholder="在此粘贴方法章节内容...",
                help="论文中描述方法的章节内容。建议使用 Markdown 格式。"
            )

        with col_input2:
            # 图注示例选择器
            caption_example = st.selectbox(
                "加载示例（图注）",
                ["无", "PaperBanana 框架"],
                key="caption_example_selector"
            )

            # 根据示例选择或会话状态设置值
            if caption_example == "PaperBanana 框架":
                caption_value = example_caption
            else:
                caption_value = st.session_state.get("caption", "")

            caption = st.text_area(
                "图注（建议使用 Markdown 格式）",
                value=caption_value,
                height=250,
                placeholder="输入图注...",
                help="要生成的图表的标题或描述。建议使用 Markdown 格式。"
            )

        # 处理按钮
        if st.button("🚀 生成候选方案", type="primary", use_container_width=True):
            if not method_content or not caption:
                st.error("请同时提供方法内容和图注！")
            else:
                # 保存到会话状态
                st.session_state["method_content"] = method_content
                st.session_state["caption"] = caption

                with st.spinner(
                    f"正在并行生成 {num_candidates} 个候选方案（策略={concurrency_mode}, 上限={int(max_concurrent)}）..."
                ):
                    # 创建输入数据列表
                    input_data_list = create_sample_inputs(
                        method_content=method_content,
                        caption=caption,
                        aspect_ratio=aspect_ratio,
                        num_copies=num_candidates,
                        max_critic_rounds=max_critic_rounds,
                        image_resolution=image_resolution
                    )

                    effective_concurrency_runtime = compute_effective_concurrency(
                        concurrency_mode=concurrency_mode,
                        max_concurrent=int(max_concurrent),
                        total_candidates=len(input_data_list),
                    )
                    estimated_batches_runtime = math.ceil(
                        len(input_data_list) / max(1, effective_concurrency_runtime)
                    )

                    st.caption(
                        f"并发配置：{concurrency_mode} | 上限 {int(max_concurrent)} | "
                        f"有效并发 {effective_concurrency_runtime} | 预计批次 {estimated_batches_runtime}"
                    )
                    progress_bar = st.progress(0.0, text="等待任务开始...")
                    progress_text = st.empty()
                    status_text = st.empty()
                    status_history = []
                    run_started_at = time.perf_counter()

                    def on_progress(done_count: int, total_count: int, effective_count: int):
                        ratio = 0.0 if total_count <= 0 else min(done_count / total_count, 1.0)
                        elapsed = time.perf_counter() - run_started_at
                        progress_bar.progress(
                            ratio,
                            text=f"并发 {effective_count} | 已完成 {done_count}/{total_count}",
                        )
                        progress_text.caption(
                            f"已耗时 {elapsed:.1f}s | 剩余 {max(total_count - done_count, 0)} 个候选"
                        )

                    def on_status(message: str):
                        if not message:
                            return
                        elapsed = time.perf_counter() - run_started_at
                        ts = datetime.now().strftime("%H:%M:%S")
                        line = f"[{ts}] {message}"
                        if status_history and status_history[-1] == line:
                            return
                        status_history.append(line)
                        if len(status_history) > 50:
                            status_history.pop(0)
                        status_text.markdown(
                            "**实时状态**\n"
                            + "\n".join([f"- {x}" for x in status_history])
                            + f"\n\n`已耗时 {elapsed:.1f}s`"
                        )

                    # 并行处理
                    try:
                        results, used_concurrency = asyncio.run(process_parallel_candidates(
                            input_data_list,
                            exp_mode=exp_mode,
                            retrieval_setting=retrieval_setting,
                            model_name=model_name,
                            image_model_name=image_model_name,
                            provider=provider,
                            api_key=api_key,
                            concurrency_mode=concurrency_mode,
                            max_concurrent=int(max_concurrent),
                            progress_callback=on_progress,
                            status_callback=on_status,
                        ))
                        total_elapsed = time.perf_counter() - run_started_at
                        progress_bar.progress(
                            1.0,
                            text=f"并发 {used_concurrency} | 已完成 {len(results)}/{len(input_data_list)}",
                        )
                        progress_text.caption(f"总耗时 {total_elapsed:.1f}s")

                        st.session_state["results"] = results
                        st.session_state["exp_mode"] = exp_mode
                        st.session_state["concurrency_mode"] = concurrency_mode
                        st.session_state["max_concurrent"] = int(max_concurrent)
                        st.session_state["effective_concurrent"] = int(used_concurrency)
                        st.session_state["estimated_batches"] = int(estimated_batches_runtime)
                        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        st.session_state["timestamp"] = timestamp_str

                        # 将结果保存为 JSON 文件
                        try:
                            # 如果结果目录不存在则创建
                            results_dir = Path(__file__).parent / "results" / "demo"
                            results_dir.mkdir(parents=True, exist_ok=True)

                            # 生成带时间戳的文件名
                            json_filename = results_dir / f"demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

                            # 保存为 JSON 并正确处理编码（与 main.py 一致）
                            with open(json_filename, "w", encoding="utf-8", errors="surrogateescape") as f:
                                json_string = json.dumps(results, ensure_ascii=False, indent=4)
                                # 清理无效的 UTF-8 字符
                                json_string = json_string.encode("utf-8", "ignore").decode("utf-8")
                                f.write(json_string)

                            st.session_state["json_file"] = str(json_filename)
                            success_count = sum(
                                1 for item in results
                                if not (isinstance(item, dict) and item.get("status") == "failed")
                            )
                            failed_count = len(results) - success_count
                            st.success(
                                f"✅ 任务完成：成功 {success_count} 个，失败 {failed_count} 个候选方案。"
                            )
                            st.info(f"💾 结果已保存至：`{json_filename.name}`")
                        except Exception as e:
                            st.warning(f"[WARN] 已生成 {len(results)} 个候选方案，但 JSON 保存失败：{e}")
                    except Exception as e:
                        progress_bar.empty()
                        progress_text.empty()
                        status_text.empty()
                        st.error(f"处理过程中出错：{e}")
                        import traceback
                        st.code(traceback.format_exc())

        # 展示结果
        if "results" in st.session_state and st.session_state["results"]:
            results = st.session_state["results"]
            current_mode = st.session_state.get("exp_mode", exp_mode)
            timestamp = st.session_state.get("timestamp", "N/A")
            mode_used = st.session_state.get("concurrency_mode", concurrency_mode)
            max_used = st.session_state.get("max_concurrent", int(max_concurrent))
            effective_used = st.session_state.get("effective_concurrent", compute_effective_concurrency(mode_used, int(max_used), len(results)))

            st.divider()
            st.markdown("## 🎨 已生成的候选方案")
            success_count = sum(
                1 for item in results
                if not (isinstance(item, dict) and item.get("status") == "failed")
            )
            failed_count = len(results) - success_count
            st.caption(
                f"生成时间：{timestamp} | 流水线：{mode_info.get(current_mode, current_mode)} | "
                f"并发：{mode_used} (max={max_used}, effective={effective_used}) | "
                f"成功/失败：{success_count}/{failed_count}"
            )

            # 如果有 JSON 文件则显示下载按钮
            if "json_file" in st.session_state:
                json_file_path = Path(st.session_state["json_file"])
                if json_file_path.exists():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.info(f"📄 结果已保存至：`{json_file_path.relative_to(Path.cwd())}`")
                    with col2:
                        with open(json_file_path, "r", encoding="utf-8") as f:
                            json_data = f.read()
                        st.download_button(
                            label="[DOWN] 下载 JSON",
                            data=json_data,
                            file_name=json_file_path.name,
                            mime="application/json",
                            use_container_width=True
                        )

            # 以网格形式展示结果（3 列）
            num_cols = 3
            num_results = len(results)

            for row_start in range(0, num_results, num_cols):
                cols = st.columns(num_cols)
                for col_idx in range(num_cols):
                    result_idx = row_start + col_idx
                    if result_idx < num_results:
                        with cols[col_idx]:
                            display_candidate_result(results[result_idx], result_idx, current_mode)

            # 添加 ZIP 下载按钮
            st.divider()
            st.markdown("### 💾 批量下载")

            try:
                import zipfile

                zip_buffer = BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                    task_name = "diagram"

                    for candidate_id, result in enumerate(results):

                        # 查找最终图像键（逻辑与展示一致）
                        final_image_key = None

                        # 尝试查找最后一轮评审
                        for round_idx in range(3, -1, -1):
                            image_key = f"target_{task_name}_critic_desc{round_idx}_base64_jpg"
                            if image_key in result and result[image_key]:
                                final_image_key = image_key
                                break

                        # 如果没有完成评审轮次则使用备选方案
                        if not final_image_key:
                            if current_mode == "demo_full":
                                final_image_key = f"target_{task_name}_stylist_desc0_base64_jpg"
                            else:
                                final_image_key = f"target_{task_name}_desc0_base64_jpg"

                        if final_image_key and final_image_key in result:
                            try:
                                raw_bytes = base64.b64decode(result[final_image_key])
                                img = Image.open(BytesIO(raw_bytes))
                                image_format = (img.format or "").upper()
                                ext_map = {
                                    "JPEG": "jpg",
                                    "JPG": "jpg",
                                    "PNG": "png",
                                    "WEBP": "webp",
                                    "GIF": "gif",
                                }
                                image_ext = ext_map.get(image_format, "bin")
                                zip_file.writestr(
                                    f"candidate_{candidate_id}.{image_ext}",
                                    raw_bytes
                                )
                            except Exception:
                                pass

                zip_buffer.seek(0)
                st.download_button(
                    label="[DOWN] 下载 ZIP 压缩包",
                    data=zip_buffer.getvalue(),
                    file_name=f"papervizagent_candidates_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                    mime="application/zip",
                    use_container_width=True
                )
                st.success("ZIP 压缩包已准备好，可以下载！")
            except Exception as e:
                st.error(f"创建 ZIP 压缩包失败：{e}")

    # ==================== 选项卡 2：精修图像 ====================
    with tab2:
        st.markdown("### 精修并放大您的图表至高分辨率（2K/4K）")
        st.caption("上传候选方案中的图像或任意图表，描述修改需求，生成高分辨率版本")

        # 精修设置侧边栏
        with st.sidebar:
            st.title("✨ 精修设置")

            refine_resolution = st.selectbox(
                "目标分辨率",
                ["2K", "4K"],
                index=0,
                key="refine_resolution",
                help="更高的分辨率需要更长时间但能产生更好的质量"
            )

            refine_aspect_ratio = st.selectbox(
                "宽高比",
                COMMON_ASPECT_RATIOS,
                index=0,
                key="refine_aspect_ratio",
                help="精修图像的宽高比"
            )

            refine_num_images = st.number_input(
                "精修张数",
                min_value=1,
                max_value=12,
                value=3,
                step=1,
                key="refine_num_images",
                help="并发生成多少张不同的精修结果"
            )

        st.divider()

        # 上传区域
        st.markdown("## 📤 上传图像")
        uploaded_file = st.file_uploader(
            "选择一个图像文件",
            type=["png", "jpg", "jpeg"],
            help="上传您想要精修的图表"
        )

        if uploaded_file is not None:
            # 展示上传的图像
            uploaded_image = Image.open(uploaded_file)
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 原始图像")
                st.image(uploaded_image, use_container_width=True)

            with col2:
                st.markdown("### 编辑指令")
                edit_prompt = st.text_area(
                    "描述您想要的修改",
                    height=200,
                    placeholder="例如：'将配色方案改为学术论文风格' 或 '将文字放大加粗' 或 '保持内容不变但输出更高分辨率'",
                    help="描述您想要的修改，或使用'保持内容不变'仅进行放大",
                    key="edit_prompt"
                )

                if st.button("✨ 精修图像", type="primary", use_container_width=True):
                    if not edit_prompt:
                        st.error("请提供编辑指令！")
                    else:
                        refine_progress_bar = st.progress(0.0, text="等待精修任务启动...")
                        refine_progress_text = st.empty()
                        refine_status_text = st.empty()
                        refine_status_history = []
                        refine_started_at = time.perf_counter()

                        def on_refine_progress(done_count: int, total_count: int):
                            ratio = 0.0 if total_count <= 0 else min(done_count / total_count, 1.0)
                            elapsed = time.perf_counter() - refine_started_at
                            refine_progress_bar.progress(
                                ratio,
                                text=f"精修进度：已完成 {done_count}/{total_count}",
                            )
                            refine_progress_text.caption(
                                f"已耗时 {elapsed:.1f}s | 剩余 {max(total_count - done_count, 0)} 张"
                            )

                        def on_refine_status(message: str):
                            if not message:
                                return
                            ts = datetime.now().strftime("%H:%M:%S")
                            line = f"[{ts}] {message}"
                            if refine_status_history and refine_status_history[-1] == line:
                                return
                            refine_status_history.append(line)
                            if len(refine_status_history) > 10:
                                refine_status_history.pop(0)
                            html_lines = "<br>".join(html.escape(x) for x in refine_status_history)
                            refine_status_text.markdown(
                                (
                                    "**精修实时状态（最近10条）**\n"
                                    f"<div style='max-height:220px; overflow-y:auto; "
                                    f"border:1px solid rgba(255,255,255,0.12); border-radius:8px; "
                                    f"padding:8px 10px; line-height:1.6;'>"
                                    f"{html_lines}"
                                    "</div>"
                                ),
                                unsafe_allow_html=True,
                            )

                        with st.spinner(
                            f"正在并发精修 {int(refine_num_images)} 张图像至 {refine_resolution} 分辨率... 这可能需要一分钟。"
                        ):
                            try:
                                # 保持上传原图字节与 MIME（不做强制 JPEG 转换）
                                image_bytes = uploaded_file.getvalue()
                                input_mime_type = normalize_image_mime_type(getattr(uploaded_file, "type", None))
                                on_refine_status(f"[精修] input mime={input_mime_type}, bytes={len(image_bytes)}")

                                # 并发调用精修 API
                                refined_results = asyncio.run(
                                    refine_images_with_count(
                                        image_bytes=image_bytes,
                                        edit_prompt=edit_prompt,
                                        num_images=int(refine_num_images),
                                        aspect_ratio=refine_aspect_ratio,
                                        image_size=refine_resolution,
                                        api_key=api_key,
                                        provider=provider,
                                        input_mime_type=input_mime_type,
                                        progress_callback=on_refine_progress,
                                        status_callback=on_refine_status,
                                    )
                                )

                                total_refine_elapsed = time.perf_counter() - refine_started_at
                                refine_progress_bar.progress(
                                    1.0,
                                    text=f"精修进度：已完成 {int(refine_num_images)}/{int(refine_num_images)}",
                                )
                                refine_progress_text.caption(f"总耗时 {total_refine_elapsed:.1f}s")

                                refined_images = []
                                for idx, (refined_bytes, message) in enumerate(refined_results):
                                    if refined_bytes:
                                        refined_images.append({
                                            "index": idx + 1,
                                            "bytes": refined_bytes,
                                            "message": message,
                                        })

                                if refined_images:
                                    st.session_state["refined_images"] = refined_images
                                    st.session_state["refine_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                    # 注意：不能写回与已实例化 widget 同名的 key（refine_resolution）。
                                    st.session_state["refine_result_resolution"] = refine_resolution
                                    st.session_state["refine_count"] = int(refine_num_images)
                                    if "refined_image" in st.session_state:
                                        st.session_state.pop("refined_image")
                                    st.success(f"✅ 精修完成：成功生成 {len(refined_images)} 张图像。")
                                    st.rerun()
                                else:
                                    st.error("[ERR] 未获得有效精修图像，请稍后重试。")
                            except Exception as e:
                                refine_progress_bar.empty()
                                refine_progress_text.empty()
                                st.error(f"精修过程中出错：{e}")
                                import traceback
                                st.code(traceback.format_exc())

            # 展示精修结果（如有）
            if "refined_images" in st.session_state and st.session_state["refined_images"]:
                st.divider()
                st.markdown("## 🎨 精修结果")
                final_resolution = st.session_state.get("refine_result_resolution", refine_resolution)
                final_count = st.session_state.get("refine_count", len(st.session_state["refined_images"]))
                st.caption(
                    f"生成时间：{st.session_state.get('refine_timestamp', 'N/A')} | "
                    f"分辨率：{final_resolution} | 张数：{final_count}"
                )

                st.markdown("### 精修前")
                st.image(uploaded_image, use_container_width=True)

                st.markdown(f"### 精修后（{final_resolution}）")
                refined_images = st.session_state["refined_images"]

                import zipfile
                zip_buffer = BytesIO()
                zip_name = f"refined_{final_resolution}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                    for item in refined_images:
                        idx = item.get("index", 0)
                        img_bytes = item.get("bytes", b"")
                        if not img_bytes:
                            continue
                        file_name = f"refined_{final_resolution}_{idx}.png"
                        zip_file.writestr(file_name, img_bytes)

                # 两列网格预览，缩小占位（仅影响预览，不影响下载原图）
                preview_cols = 2
                preview_width_px = 420
                for row_start in range(0, len(refined_images), preview_cols):
                    cols = st.columns(preview_cols, gap="large")
                    for col_offset in range(preview_cols):
                        item_pos = row_start + col_offset
                        if item_pos >= len(refined_images):
                            continue

                        item = refined_images[item_pos]
                        idx = item.get("index", item_pos + 1)
                        img_bytes = item.get("bytes", b"")
                        if not img_bytes:
                            continue

                        with cols[col_offset]:
                            st.markdown(f"#### 结果 {idx}")
                            refined_image = Image.open(BytesIO(img_bytes))
                            st.image(refined_image, width=preview_width_px)

                            file_name = f"refined_{final_resolution}_{idx}.png"
                            st.download_button(
                                label=f"[DOWN] 下载结果 {idx}",
                                data=img_bytes,
                                file_name=file_name,
                                mime="image/png",
                                key=f"download_refined_{idx}_{final_resolution}_{item_pos}",
                                use_container_width=True
                            )

                zip_buffer.seek(0)
                st.download_button(
                    label="[DOWN] 一键下载全部结果（ZIP）",
                    data=zip_buffer.getvalue(),
                    file_name=zip_name,
                    mime="application/zip",
                    use_container_width=True,
                    key="download_refined_zip"
                )

if __name__ == "__main__":
    main()

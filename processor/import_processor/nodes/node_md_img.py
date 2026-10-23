import base64
import logging
import os
import re
import time
from collections import deque
from pathlib import Path
from typing import Tuple, List, Dict

from langchain_core.messages import HumanMessage
from langchain_core.prompts import load_prompt
from langchain_openai import ChatOpenAI
from minio import Minio
from minio.deleteobjects import DeleteObject
from openai import OpenAI

from processor.import_processor.base import BaseNode, setup_logging
from processor.import_processor.state import ImportGraphState
from utils.minio_utils import get_minio_client


class NodeMDImg(BaseNode):
    """
    MarkDown图片处理节点：多模态图片理解
    """

    name = "node_md_img"


    def process(self, state: ImportGraphState):

        """
                MD文件图片处理核心节点
                核心流程：
                1. 获取MD内容、文件路径、图片文件夹路径
                2. 扫描图片文件夹，筛选MD中实际引用的支持格式图片
                3. 调用多模态大模型为图片生成内容摘要
                4. 将图片上传至MinIO，替换MD中本地图片路径为MinIO访问URL，并填充图片摘要
                5. 备份原MD文件，保存处理后的新MD文件并更新状态

                必要参数：task_id、md_path、md_content
                更新参数：md_path、md_content

                :param state: 工作流状态对象
                :return: 更新后的状态对象
                """

        # 步骤1：初始化数据，获取MD核心信息
        md_content, md_path_obj, images_dir = self._step_1_get_content(state)

        #无图片文件夹，直接跳过图片处理逻辑
        if not images_dir.exists():
            self.logger.info(f"图片文件夹不存在，跳过图片处理：{images_dir.absolute()}")
            return state

        # 步骤2：扫描并筛选MD中引用的图片
        target_images = self._step_2_scan_images(md_content, images_dir)
        if not target_images:
            self.logger.info("未检测到MD中引用的支持格式图片，跳过后续处理")
            return state

        # 步骤3：调用多模态大模型生成图片摘要
        summaries = self._step_3_generate_summaries(md_path_obj.stem, target_images)

        # 步骤4：上传图片至MinIO，替换MD图片路径并填充摘要
        new_md_content = self._step_4_upload_and_replace(md_path_obj.stem, target_images, summaries, md_content)

        # 步骤5：备份并保存新MD文件
        new_md_file_name = self._step_5_backup_new_md_file(state['md_path'], new_md_content)

        # 步骤6：更新state状态值
        state["md_content"] = new_md_content
        state["md_path"] = new_md_file_name

        return state

    def _apply_api_rate_limit(self,request_times: deque[float], max_requests: int, window_seconds: int = 60):
        # 记录当前时间
        current_time = time.time()

        # 清理滑动窗口的过期请求
        while request_times and current_time - request_times[0] >= window_seconds:
            request_times.popleft()
        # 窗口内请求数达到上限， 计算需要等待的时间并阻塞
        if len(request_times) >= max_requests:
            # 计算需要等待的时长=窗口总时长 - 窗口内第一个请求的时间
            sleep_duration = window_seconds - (current_time - request_times[0])
            if sleep_duration > 0:
                logging.getLogger().info(f"请求被限速，等待{sleep_duration:.2f}秒")
                time.sleep(sleep_duration)
                current_time = time.time()
                while request_times and current_time - request_times[0] >= window_seconds:
                    request_times.popleft()
        # 记录当前请求的时间戳, 新请求入队
        request_times.append(current_time)
        logging.getLogger().info(f"{self.name}请求成功，当前{window_seconds}s窗口内请求次数为{len(request_times)}")

    def _step_1_get_content(self, state: ImportGraphState) -> Tuple[str, Path, Path]:
        """
                从全局状态中提取并初始化MD处理所需核心数据
                :param state: 流程全局状态对象
                :return: 三元组(MD文件内容, MD文件路径, 图片文件夹路径)
                :raise FileNotFoundError: 当状态中无有效MD文件路径时抛出
                """
        md_path = state.get("md_path")
        if not md_path:
            raise ValueError("核心参数md_path缺失")

        md_path_obj = Path(md_path)
        if not md_path_obj.exists():
            raise ValueError(f"MD文件不存在，绝对路径: {md_path_obj.absolute()}")

        md_content = md_path_obj.read_text(encoding="utf-8")

        img_dir = md_path_obj.parent / "images"

        return md_content, md_path_obj, img_dir

    def _step_2_scan_images(self, md_content, images_dir) -> List[Tuple[str, str, Tuple[str, str]]]:
        """
            扫描图片文件夹，过滤出「支持格式+MD中实际引用」的图片，组装处理元数据
            :param md_content: MD文件完整内容
            :param images_dir: 图片文件夹路径对象
            :return: 待处理图片列表，每个元素为(图片文件名, 图片完整路径, 图片上下文)元组
        """
        target_images = []
        for image_file in os.listdir(images_dir):
            file_ext = os.path.splitext(image_file)[1].lower()
            if file_ext not in self.config.image_extensions:
                self.logger.warning(f"图片{image_file}格式不支持")
                continue

            img_path = str(images_dir / image_file)
            context = self._find_image_in_md(md_content, image_file)
            if not context:
                self.logger.warning(f"图片{image_file}未在MD中找到")
                continue

            target_images.append((image_file, img_path, context))
        return target_images

    def _find_image_in_md(self, md_content: str, image_file: str, context_len: int=100) -> Tuple[str, str]:
        #re.escape(image_file)对特殊符号进行转义
        pattern = re.compile(r"!\[.*?\]\(.*?" + re.escape(image_file) + r".*?\)")
        match = pattern.search(md_content)
        if not match:
            return None

        start, end = match.span()
        # print(f"start: {start}")
        # print(f"end: {end}")
        pre_text = md_content[max(0,start-context_len):start]
        post_text = md_content[end: min(len(md_content)-end,context_len)]

        return pre_text, post_text

    def _step_3_generate_summaries(self, stem, target_images):
        """
            步骤3：批量为待处理图片生成内容摘要，带API速率限制防止触发大模型限流
            :param doc_stem: 文档文件名（不含后缀），作为大模型prompt上下文
            :param targets: 待处理图片列表，元素为(图片文件名, 图片完整路径, 图片上下文)
            :param requests_per_minute: 每分钟最大API请求数，默认9次（按大模型限制调整）
            :return: 图片摘要字典，键：图片文件名，值：图片内容摘要
        """
        summaries = {}

        # 1、外部初始化双端队列，用于API速率限制，跨循环复用
        request_deque = deque()

        # 2、循环处理图片
        for img_file, image_path, context in target_images:
            # 2.1、速率限制
            self._apply_api_rate_limit(request_deque, max_requests=3, window_seconds=10)

            # 2.2、调用大模型生成图片摘要
            self.logger.info(f"开始生成图片摘要：{image_path}")
            summaries[img_file] = self._summarize_image(image_path, root_folder=stem, image_content=context)

        self.logger.info(f"图片摘要批量生成完成，共处理{len(summaries)}张图片")
        return summaries

    def _summarize_image(self, image_path: str, root_folder: str, image_content: Tuple[str, str]) -> str:
        """
        调用多模态大模型生成图片内容摘要（适配LangChain工具类，复用项目统一LLM客户端）
        生成的摘要用于Markdown图片标题，严格控制50字以内中文描述
        :param image_path: 图片本地完整路径
        :param root_folder: 文档所属文件夹/主名，为大模型提供上下文
        :param image_content: 图片在MD中的上下文元组，格式(上文文本, 下文文本)
        :return: 图片内容摘要（异常时返回默认值"图片描述"）
        """


        with open(image_path, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode("utf-8")

        try:
            chat_model = ChatOpenAI(
                model=self.config.vl_model,
                api_key=self.config.openai_api_key,
                base_url=self.config.openai_api_base,
                temperature=self.config.llm_default_temperature
            )
            messages = [
                HumanMessage(
                    content=[
                        # 文本提示词：携带上下文，限定摘要规则
                        {
                            "type": "text",
                            "text": f"""这是"{root_folder}"文件中的一张图片，图片上文部分为"{image_content[0]}"，
                            下文部分为"{image_content[1]}"，请用中文简要总结这张图片的内容，用于 Markdown 图片标题。"""
                        },
                        # 多模态核心：Base64编码图片数据
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                )
            ]

            response = chat_model.invoke(messages)
            return response.content.strip().replace("\n", "")
        except Exception as e:
            self.logger.error(f"获取图片摘要失败：{image_path}, 错误：{e}")
            return "图片描述"

    def _step_4_upload_and_replace(self, stem: str,
                                   target_images: List[Tuple[str, str, Tuple[str,str]]],
                                   summaries: Dict[str, str],
                                   md_content: str) -> str:
        """
                步骤 4: 上传图片并合并信息，然后替换 Markdown 中的内容。

                流程：
                1. 确定 MinIO 上的上传目录（按文档名隔离）。
                2. 清理该目录下的旧数据。
                3. 批量上传图片。
                4. 合并“图片摘要”和“图片URL”。
                5. 替换 Markdown 文本中的图片引用。
                :param doc_stem: 文档文件名（不含后缀），作为MinIO上传子目录名（按文档隔离）
                :param target_images: 待处理图片列表，元素为(图片文件名, 图片完整路径, 图片上下文)
                :param summaries: 图片摘要字典，键：图片文件名，值：内容摘要
                :param md_content: 原始MD文件内容
                :return: 图片引用替换后的新MD内容
                """
        # 获取minio客户端
        minio_client = get_minio_client()
        # 获取文件上传路径
        minio_dir = self.config.img_dir

        upload_dir = f"{minio_dir}/{stem}".replace(" ","")
        # 清理已有目录
        self._clean_minio_dir(minio_client, upload_dir)
        urls = self._upload_images_batch(minio_client, upload_dir, target_images)

        image_info = self._merge_summary_and_url(summaries, urls)

        # 步骤4：替换MD内容中的本地图片引用为MinIO远程引用
        md_content = self._process_md_file(md_content, image_info)

        return md_content



    def _clean_minio_dir(self, minio_client: Minio, upload_dir: str):
        try:
            # 获取将要被删除的图片列表
            objects_to_delete = minio_client.list_objects(self.config.minio_bucket, upload_dir, recursive=True)
            delete_list = [DeleteObject(obj.object_name) for obj in objects_to_delete]
            errors = minio_client.remove_objects(
                self.config.minio_bucket,
                delete_list
            )
            for error in errors:
                self.logger.error(f"删除图片失败：{error}")
        except Exception as e:
            self.logger.error(f"minio连接失败，错误原因{e}")

    def _upload_images_batch(self, minio_client: Minio, upload_dir: str, target_images: List[Tuple]) -> Dict[str, str]:
        """
        批量上传待处理图片至MinIO，返回图片文件名与访问URL的映射关系
        :param minio_client: 初始化完成的MinIO客户端对象
        :param upload_dir: MinIO上传根目录
        :param target_images: 待处理图片列表，元素为(图片文件名, 图片完整路径, 图片上下文)
        :return: 图片URL字典，键：图片文件名，值：MinIO访问URL
        """
        urls = {}
        for img_file, img_path, _ in target_images:
            object_name = f"{upload_dir}/{img_file}"
            urls[img_file] = self._upload_to_minio(minio_client, img_path, object_name)
        return urls

    def _upload_to_minio(self, minio_client: Minio, local_path: str, object_name: str) -> str | None:
        """
                批量上传待处理图片至MinIO，返回图片文件名与访问URL的映射关系
                :param minio_client: 初始化完成的MinIO客户端对象
                :param upload_dir: MinIO上传根目录
                :param target_images: 待处理图片列表，元素为(图片文件名, 图片完整路径, 图片上下文)
                :return: 图片URL字典，键：图片文件名，值：MinIO访问URL
                """
        try:
            minio_client.fput_object(
                bucket_name=self.config.minio_bucket,  # MinIO存储桶名（从配置读取）
                object_name=object_name,  # MinIO对象名称
                file_path=local_path,  # 本地文件路径
                # 关键规则：多后缀文件仅拆分最后一个，如test.tar.gz拆分为("test.tar", ".gz")。
                # 这个属性根据上传的文件名的后缀，文件是什么类型就是什么类型
                content_type=f"image/{os.path.splitext(local_path)[1][1:]}"
            )
            url = f"http://{self.config.minio_endpoint}/{self.config.minio_bucket}/{object_name}"

            return url
        except:
           self.logger.error(f"上传图片失败：{local_path}")

    def _merge_summary_and_url(self, summaries :Dict[str, str], urls: Dict[str, str]) -> Dict[str,Tuple[str, str]]:
        """
                合并图片摘要字典和URL字典，过滤掉上传失败无URL的图片
                :param summaries: 图片摘要字典，键：图片文件名，值：内容摘要
                :param urls: 图片URL字典，键：图片文件名，值：MinIO访问URL
                :return: 合并后的图片信息字典，键：图片文件名，值：(摘要, URL)元组
                """
        image_info = {}
        for image_file, summary in summaries.items():
            if url := urls.get(image_file):
                image_info[image_file] = (summary, url)
        return image_info

    def _process_md_file(self, md_content: str, image_info: Dict[str, Tuple[str, str]]) -> str:
        """
        核心功能：替换MD内容中的本地图片引用为MinIO远程引用
        替换规则：![原描述](本地路径) → ![图片摘要](MinIO访问URL)
        :param md_content: 原始MD文件内容
        :param image_info: 合并后的图片信息字典，键：图片文件名，值：(摘要, URL)
        :return: 替换后的新MD内容
        """

        for image_file, (summary, url) in image_info.items():
            pattern = re.compile(r"!\[.*?\]\(.*?" + re.escape(image_file) + r".*?\)")

            md_content = pattern.sub(lambda m: f"![{summary}]({url})", md_content)

        return md_content

    def _step_5_backup_new_md_file(self, origin_md_path: str, md_content: str) -> str:
        """
        步骤5：将处理后的MD内容保存为新文件（原文件不变，避免数据丢失）
        新文件命名规则：原文件名 + _new.md（如test.md → test_new.md）
        :param origin_md_path: 原始MD文件完整路径
        :param md_content: 处理后的新MD内容
        :return: 新MD文件的完整路径
        """
        new_md_file_name = os.path.splitext(origin_md_path)[0] + "_new.md"
        with open(new_md_file_name, "w", encoding="utf-8") as f:
            f.write(md_content)
        return new_md_file_name


if __name__ == '__main__':
    setup_logging()

    md_path = r"D:\output\hak180产品安全手册\hak180产品安全手册.md"
    with open(md_path, "r", encoding="utf-8") as f:
        md_content = f.read()

    # 构造测试状态对象，模拟流程入参
    init_state = {
        "md_path": md_path,
        "md_content": md_content
    }

    # 执行核心处理流程
    node_md_img = NodeMDImg()
    final_state = node_md_img(init_state)
import json
import logging
import shutil
import time
import zipfile
from ctypes import HRESULT
from pathlib import Path

import requests

from processor.import_processor.base import BaseNode, setup_logging
from processor.import_processor.exceptions import StateFieldError, ConfigurationError, PdfConversionError, \
    FileProcessingError
from processor.import_processor.state import ImportGraphState


class NodePDFToMD(BaseNode):
    """
    PDF 转 Markdown 节点：PDF结构化解析
    """

    name = "node_pdf_to_md"

    def process(self, state: ImportGraphState):
        # 步骤1：校验PDF路径和输出目录
        pdf_path_obj, output_dir_obj = self._step_1_validate_paths(state)

        # 步骤2：上传PDF至MinerU并轮询解析结果
        zip_url = self._step_2_upload_and_poll(pdf_path_obj)
        self.logger.info(zip_url)

        # 步骤3：下载ZIP包并提取MD文件
        md_path = self._step_3_download_and_extract(zip_url, output_dir_obj, pdf_path_obj.stem)

        with open(md_path, "r", encoding="utf-8") as f:
            md_content = f.read()

        state["md_content"] = md_content
        state["md_path"] = str(md_path)
        self.logger.info(self.config.mineru_base_url)

        return state

    def _step_1_validate_paths(self, state: ImportGraphState):
        """
                步骤1：校验PDF文件路径和输出目录
                核心职责：参数非空校验 | 路径转换 | PDF文件有效性校验 | 输出目录自动创建
                返回：合法的PDF文件Path对象、输出目录Path对象
                异常：ValueError(参数缺失)、FileNotFoundError(文件无效)
                """

        pdf_path = state.get("pdf_path")
        if not pdf_path:
            raise StateFieldError(
                field_name="pdf_path",
                message="PDF文件路径不能为空",
                expected_type=str
            )
        file_dir = state.get("file_dir")
        if not file_dir:
            raise StateFieldError(
                field_name="file_dir",
                message="输出路径不能为空",
                expected_type=str
            )
        pdf_path_obj = Path(pdf_path)
        file_dir_obj = Path(file_dir)

        if not pdf_path_obj.exists():
            raise FileNotFoundError(
                f"文件{pdf_path_obj}不存在"
            )

        if not file_dir_obj.exists():
            self.logger.info(f"输出目录{file_dir_obj}不存在，已自动创建")
            file_dir_obj.mkdir(parents=True, exist_ok=True)

        return pdf_path_obj, file_dir_obj

    def _step_2_upload_and_poll(self, pdf_path_obj):
        """
                步骤2：上传PDF至MinerU并轮询解析任务状态
                核心流程：配置校验 → 获取上传链接 → 文件上传 → 任务轮询（直至完成/失败/超时）
                参数：pdf_path_obj-已校验的PDF Path对象
                返回：解析结果ZIP包下载链接full_zip_url
                异常：ValueError(配置缺失)、RuntimeError(请求/上传失败)、TimeoutError(任务超时)
                """
        # 1、参数校验
        if not self.config.mineru_base_url or not self.config.mineru_api_token:
            raise ConfigurationError("MinerU配置缺失：请在 .env 文件中正确配置 MINERU_API_TOKEN 和 MINERU_BASE_URL 参数")
        self.logger.info(f"【配置校验】MinerU配置校验成功，开始处理文件：{pdf_path_obj.name}")

        token = self.config.mineru_api_token
        url = f"{self.config.mineru_base_url}/file-urls/batch"
        header = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}"
        }
        data = {
            "files": [
                {"name": pdf_path_obj.name}
            ],
            "model_version": "vlm"
        }

        response = requests.post(url, headers=header, json=data)
        if response.status_code != 200:
            raise PdfConversionError("获取上传链接失败，相应状态:{}，响应结果:{}".format(response.status_code, response.text))
        result = response.json()
        if result["code"] != 0:
            raise PdfConversionError("获取上传链接失败，错误信息:{}".format(result["msg"]))
        batch_id = result["data"]["batch_id"]
        urls = result["data"]["file_urls"]
        with open(pdf_path_obj, 'rb') as f:
            res_upload = requests.put(urls[0], data=f)
            if res_upload.status_code != 200:
                raise PdfConversionError(f"{urls[0]}上传文件失败")
            self.logger.info("上传文件成功")

        pull_url = f"{self.config.mineru_base_url}/extract-results/batch/{batch_id}"
        start_time = time.time()
        timeout_seconds = 600
        poll_interval = 5
        self.log_step(step_name="轮询开始",
        message=f"轮询间隔{poll_interval}s,超时时间{timeout_seconds}s, batch_id:{batch_id}")
        while True:
            elapsed_time = time.time() - start_time
            if elapsed_time > timeout_seconds:
                raise TimeoutError(f"任务轮询超时，已耗时间：{int(elapsed_time)}s")
            try:
                pull_res = requests.get(pull_url, headers=header)
            except Exception as e:
                self.logger.warning(f"网络请求异常，{poll_interval}s后重试，batch_id:{batch_id}")
                self.logger.warning(f"异常信息：{str(e)}")
                time.sleep(poll_interval)
                continue

            if pull_res.status_code != 200:
                raise PdfConversionError(f"任务轮询失败，相应状态:{pull_res.status_code}")

            pull_res_json = pull_res.json()
            if pull_res_json["code"] != 0:
                raise PdfConversionError(f"任务轮询失败，错误码:{pull_res_json['code']}")

            extract_results = pull_res_json["data"]["extract_result"]
            extract_result = extract_results[0]
            data_state = extract_result["state"]
            if data_state == "done":
                self.log_step(step_name="任务轮询", message=f"解析完成，总耗时：{int(elapsed_time)}s, batch_id:{batch_id}")

                full_zip_url = extract_result["full_zip_url"]
                self.log_step(step_name="任务轮询", message=f"解析结果下载链接：{full_zip_url}")
                return full_zip_url

            elif data_state == "failed":
                self.log_step(step_name="任务轮询", message=f"解析失败，总耗时：{int(elapsed_time)}s, batch_id:{batch_id}")
                raise PdfConversionError(f"任务失败，错误信息:{extract_result['err_msg']}")
            else:
                self.log_step(step_name="任务轮询", message=f"处理中，已耗时：{int(elapsed_time)}s, batch_id:{batch_id}")

                time.sleep(poll_interval)

    def _step_3_download_and_extract(self, zip_url: str, output_dir_obj: Path, stem: str):
        """
               步骤3：下载MinerU解析结果ZIP包并解压，提取目标MD文件
               核心流程：下载ZIP → 清理旧目录并解压 → 查找MD文件 → 重命名统一为PDF同名
               参数：zip_url-ZIP包下载链接；output_dir_obj-输出目录Path；pdf_stem-PDF无后缀纯名称
               返回：最终MD文件的字符串格式绝对路径
               异常：RuntimeError(下载失败)
               """
        response = requests.get(zip_url)
        if response.status_code != 200:
            raise FileProcessingError("下载ZIP包失败")

        zip_save_path = output_dir_obj / f"{stem}_result.zip"
        with open(zip_save_path, 'wb') as f:
            f.write(response.content)
        self.log_step(step_name="下载成功", message=f"下载ZIP包成功，保存路径：{zip_save_path}")
        extract_target_dir = output_dir_obj / stem
        # 删除已有的目录，之前可能已经解压过
        if extract_target_dir.exists():
            shutil.rmtree(extract_target_dir)
        extract_target_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(zip_save_path, 'r') as zip_file_obj:
            zip_file_obj.extractall(extract_target_dir)
        self.log_step("zip解压","解压完成")

        target_md_file = extract_target_dir / "full.md"
        new_md_path = target_md_file.with_name(f"{stem}.md")
        target_md_file.rename(new_md_path)

        return new_md_path.absolute()


if __name__ == '__main__':

    setup_logging()
    init_state = {
        "pdf_path": r"d:\doc\hak180产品安全手册.pdf",
        "file_dir": r"d:\output"
    }

    node_pdf_to_md = NodePDFToMD()
    result = node_pdf_to_md(init_state)

    json_state = json.dumps(result, ensure_ascii=False, indent=4)

    logging.getLogger().info(json_state)
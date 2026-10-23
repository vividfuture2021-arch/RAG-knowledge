from modelscope.hub.snapshot_download import snapshot_download

model_dir = snapshot_download(
    'BAAI/bge-m3',
    cache_dir=r'D:/ai_models/modelscope_cache'
)
print(f"模型路径：{model_dir}")
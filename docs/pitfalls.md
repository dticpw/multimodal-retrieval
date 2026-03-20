# 踩坑记录

开发过程中遇到的问题及解决方案，可作为面试话题展示工程调试能力。

## P1 踩坑

### 1. CLIP model `get_text_features()` 返回类型不一致

**现象**: `model.get_text_features()` 返回 `BaseModelOutputWithPooling` 而非 tensor，导致后续 L2 归一化报错。
**原因**: transformers 版本差异，部分版本的 CLIPModel 高层 API 行为不一致。
**解决**: 绕过高层 API，手动调用底层模块：
```python
# 文本编码
text_out = model.text_model(input_ids=..., attention_mask=...)
embeddings = model.text_projection(text_out.pooler_output)

# 图像编码
vision_out = model.vision_model(pixel_values=...)
embeddings = model.visual_projection(vision_out.pooler_output)
```
**面试要点**: 展示对模型内部结构的理解，不只是调 API，能深入到模型子模块层面解决问题。

### 2. Windows 路径中文乱码

**现象**: 项目路径含中文（`论文精读/601_开始造项目`），日志输出乱码。
**解决**: 将项目移到纯英文路径 `E:/PG/multimodal-retrieval/`。
**面试要点**: 工程实践中路径编码问题是常见坑，尤其在跨平台部署时。

### 3. Windows 下没有 nohup

**现象**: 尝试用 `nohup python ... &` 后台运行评测脚本，报 command not found。
**原因**: nohup 是 Unix 工具，Windows 不自带。
**解决**: Windows 下直接在新终端运行，或用 `start /b python ...` 替代。

## P2 踩坑

### 4. OpenAI SDK base_url 拼接规则

**现象**: 中转 API 返回 403 Forbidden 或 404 Not Found。
**原因**: OpenAI Python SDK 会在 base_url 后自动拼接 `/chat/completions`。如果 base_url 写成 `https://proxy.example.com/claude/aws`，实际请求变成 `https://proxy.example.com/claude/aws/chat/completions`，而中转期望的是 `/v1/chat/completions`。
**解决**: base_url 必须以 `/v1` 结尾，如 `https://proxy.example.com/v1`。
**面试要点**: 调试 HTTP API 时要关注实际发出的请求 URL，日志中 httpx 会打印完整请求地址，是定位问题的关键线索。

### 5. .env 文件隐藏字符 / 编码问题

**现象**: API key 在 .env 中看起来正常，但运行时报 `UnicodeEncodeError: 'ascii' codec can't encode characters`。
**排查**: 用 Python 读取文件原始字节，检查 BOM 头和非 ASCII 字符：
```python
with open('.env', 'rb') as f:
    data = f.read()
print('BOM:', data[:3] == b'\xef\xbb\xbf')
for line in data.split(b'\n'):
    if b'API_KEY' in line:
        print([hex(b) for b in line if b > 127])
```
**解决**: 重新生成干净的 .env 文件，确保 UTF-8 无 BOM 编码。
**面试要点**: 配置文件编码问题在部署时很常见，掌握字节级排查手段。

### 6. th123 环境缺少 requests 包

**现象**: `ModuleNotFoundError: No module named 'requests'`。
**原因**: conda 环境是按需安装的，requests 不在初始依赖中。
**解决**: `pip install requests` 并加入 requirements.txt。
**经验**: 每次引入新的运行时依赖都要同步更新 requirements.txt。

## P3 踩坑 (Gradio Demo)

### 7. Gradio 6 构造参数迁移

**现象**: `gr.Blocks(theme=gr.themes.Soft())` 启动时报 UserWarning，且后续流程异常。
**原因**: Gradio 6.0 将 `theme`、`css` 等参数从 `Blocks()` 构造函数迁移到了 `launch()` 方法。
**解决**:
```python
# 错误 (Gradio 6+)
with gr.Blocks(theme=gr.themes.Soft()) as demo: ...

# 正确
with gr.Blocks() as demo: ...
demo.launch(theme=gr.themes.Soft())
```
**面试要点**: 第三方库大版本升级常有 API 迁移，要关注 deprecation warning，查文档确认参数签名变化。

### 8. Gradio 6 文件访问安全限制 (allowed_paths)

**现象**: 文本搜图功能返回"错误"，但独立调用服务层函数正常返回结果。
**原因**: Gradio 6 出于安全考虑，默认禁止访问任意本地文件路径。Gallery 组件尝试读取 `E:/PG/dataset/flickr30k-images/` 下的图片时被拒绝，导致 500 错误。前端只显示泛泛的"错误"二字，无具体信息。
**排查**: 关键线索是"独立调用正常，Gradio 内报错"——说明不是业务逻辑问题，而是 Gradio 框架层面的限制。
**解决**: 在 `launch()` 中显式声明允许访问的路径：
```python
demo.launch(
    allowed_paths=[settings.flickr30k_image_dir],
)
```
**面试要点**: 当"相同代码在不同环境下表现不同"时，优先排查环境/框架层面的差异（权限、沙箱、安全策略），而非反复检查业务逻辑。

### 9. Gradio 6 Analytics 网络请求导致启动崩溃

**现象**: Demo 启动后打印 `Running on local URL: http://0.0.0.0:7860`，但随即 crash，报 `httpx.RemoteProtocolError: Server disconnected without sending a response`。
**原因**: Gradio 启动时默认会请求 `https://api.gradio.app` 做 analytics 上报。在代理环境下该请求失败，且 Gradio 没有优雅降级，直接抛异常退出。
**解决**: 启动前设置环境变量禁用 analytics：
```python
import os
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
import gradio as gr  # 必须在设置环境变量之后再 import
```
**面试要点**: 第三方库的遥测/analytics 功能在公司内网或代理环境下经常出问题，这是部署时的常见坑。通常通过环境变量或配置项关闭。

# GLDAS-Gaze 网页单图推理

浏览器界面直接调用本机 TP / TA 模型，返回搜索轨迹、逐步热图与终止概率。

## 启动

在仓库根目录运行：

```bash
./web_demo/start.sh
```

打开 <http://127.0.0.1:8765>，选择示例或上传图片，再点击“开始推理”。
支持 JPEG、PNG、WebP，单张不超过 10 MB、2400 万像素。
图片仅在请求内处理，不写入服务器磁盘。

推理完成后，“逐步热图”会展示每次实际解码的原图叠加预览，包含对应的
终止概率和下一注视点。点击卡片可联动上方大图与轨迹回放；“热图强度”
可调整透明度，每张卡片还可单独下载 PNG。模型决定停止的那一步同样展示
热图，但不绘制新的注视点。达到最大步数后不会补画未经模型计算的热图。

启动器优先使用 `GLDAS_PYTHON` 指定的环境，其次尝试项目本地运行器和
已有 Conda / Python 环境。自有环境可显式指定：

```bash
GLDAS_PYTHON=/path/to/env/bin/python ./web_demo/start.sh \
  --device auto --port 8765 --dataset-root /path/to/dataset
```

也可以在已经安装项目依赖的 Python 环境里直接启动：

```bash
python -m web_demo.server --device cpu
```

依赖沿用项目的 PyTorch、torchvision、Detectron2、NumPy、Pillow 等；
HTTP 服务使用 Python 标准库，不需要 Node.js、前端构建或额外 Web 框架。
PyTorch 1.x 环境应使用 NumPy 1.x。完整环境安装步骤见仓库主 README。

项目本地的 `.web-runtime/` 是可选的辅助运行目录，已加入 Git 忽略。
复制仓库到其他机器后，应先安装依赖并激活自己的 Python 环境；
也可用 `GLDAS_PYTHON` 显式指定解释器。
启动默认仅监听 `127.0.0.1`；远程服务器可通过 SSH 转发访问：

```bash
ssh -L 8765:127.0.0.1:8765 user@server
```

## 模型与坐标

预训练权重可从仓库主 README 中的下载目录获取，放入项目根目录的
`pretrained_models/`。自备图片无需 COCO-Search18 数据集。
无需启动网页时，可用 `python tools/infer_image.py --help` 查看命令行单图
推理脚本；它会保存轨迹叠加图和不含热图二进制的 JSON 结果。

- 权重读取自 `pretrained_models/TP.pt` 和 `pretrained_models/TA.pt`，
  骨干与像素解码器使用同目录下的两个 `M2F_R50*.pkl` 文件。
- 首次选择某个模型时加载，后续请求复用缓存；不会启动训练。
- 用户选择的是 18 类搜索任务和模型训练条件，模型不自动判断目标存在性。
- 预处理与原评估一致：图片缩放至 512 × 320，再按 ImageNet 均值方差归一化。
- 返回坐标基于 512 × 320，画面按原图比例映射；EXIF 方向在服务端统一校正。
- 轨迹包含固定中心起点；TP 最多新增 6 点，TA 最多新增 10 点。
  终止概率严格大于所设阈值时停止；不使用真实框进行命中截断。
- 每个注视点的 `stop_probability` 表示在该点历史下的停止概率。
  达到解码上限后新增的最后一点没有再次求值，因此该概率为 `null`。
- 每张热图表示对应注视历史下的下一点预测分数，按单步最小值和最大值
  映射为透明 PNG；它不是跨步骤可直接比较的概率色标。
- 贪心解码取当前最大分数位置；采样解码使用请求专属的固定种子随机生成器。
  两种方式都沿用训练配置中的返回抑制半径。

## HTTP 接口

- `GET /api/status`：依赖、权重文件、缓存状态、支持的任务和解码步数。
  `ready` 表示前置检查通过；实际模型结构在首次加载时严格校验。
- `GET /api/examples?condition=TP&task=bottle`：当前条件与类别的前 6 个示例。
- `GET /api/example?id=TP%2Fbottle%2Ffilename.jpg`：读取受限的示例图片。
- `POST /api/predict`：传入 JSON，示例请求如下。

```json
{
  "image": "data:image/png;base64,...",
  "task": "bottle",
  "condition": "TP",
  "decoding": "greedy",
  "max_steps": 6,
  "stop_threshold": 0.5,
  "seed": 42
}
```

`image` 和 `example_id` 二选一。响应包含 `image`、`fixations`、`steps`、
`heatmaps`、`model`、`timing` 和 `meta`。页面导出 JSON 时会去掉图片及
热图二进制，只保留尺寸、坐标、逐步终止概率和推理参数。

模型同一时刻处理一个请求，忙碌时返回 HTTP 429；输入错误返回 400，
超出大小限制返回 413，运行环境或权重未就绪返回 503。

## 验证

```bash
.web-runtime/python -m unittest web_demo.test_server -v
node --check web_demo/static/app.js
```

接口测试不加载模型权重；真实 TP / TA 单图轨迹另与原评估函数
`scanpath_decode` 做过逐值核对。网页需在可连接的浏览器中进一步做视觉验收。

# TemplatePattern_final

运行命令与产物目录见 [两阶段部署流程](第一步和第二步部署流程总结.md)。测试脚本、验收报告及生成结果仅保留在本地，不纳入当前仓库版本。

独立的两阶段衬衫纸样与动态 SMPL 流水线。Python 环境现在可以使用 `uv` 管理；Docker 镜像也已改为基于 `uv`，不再依赖 Conda。CUDA 编译器、系统图形库和 NVIDIA 驱动仍属于系统级依赖，不能由 `uv` 替代。

## 1. 使用 uv 本地运行

```bash
cd /home/user/buff-tomma/SewingPattern
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.9.21
uv venv --python 3.9.21 .venv
uv pip install --python .venv/bin/python -r TemplatePattern_final/requirements-runtime.txt
uv pip install --python .venv/bin/python -e TemplatePattern_final
```

Stage 2 还需要本机准备 GarmentCode、NvidiaWarp-GarmentCode 和 SMPL 男体模型，并设置：

```bash
export TEMPLATEPATTERN_GARMENTCODE_ROOT=/path/to/GarmentCode
export TEMPLATEPATTERN_WARP_ROOT=/path/to/NvidiaWarp-GarmentCode
export TEMPLATEPATTERN_SMPL_MODEL_DIR=/path/to/smpl-models
```

SMPL 模型目录中必须包含：

```text
basicmodel_m_lbs_10_207_0_v1.1.0.pkl
```

本地首次使用 Warp 时，还要在 NvidiaWarp-GarmentCode 根目录执行：

```bash
.venv/bin/python /path/to/NvidiaWarp-GarmentCode/build_lib.py --cuda_path=/usr/local/cuda
```

Stage 1 不需要 SMPL 模型或 GPU；完整 Stage 2 仿真需要 NVIDIA GPU、驱动、CUDA Toolkit 和已编译的 Warp。
Stage 2 正反面渲染使用 EGL；依赖保留与 pyrender 0.1.45 声明一致的 `PyOpenGL==3.1.0`，并在加载时注册 Python 3.12 ctypes 参数处理器。可先运行 `python -m TemplatePattern_final.sim.render_support` 测试带纹理的无头渲染，完整 Stage 2 也会在耗时计算前自动检查。

## 2. Docker 运行（可选）

从 `SewingPattern` 目录构建：

```bash
docker build --network=host --progress=plain \
  -f TemplatePattern_final/Dockerfile \
  -t templatepattern:latest \
  TemplatePattern_final
```

Dockerfile 内的 `RUN` 指令默认以 root 执行，因此不使用 `sudo`。如果宿主机执行
`docker` 时提示无法访问 `/var/run/docker.sock`，请将当前用户加入 `docker` 用户组后重新登录，
或仅在宿主机命令前使用 `sudo`；这与 Dockerfile 内容无关。

如果出现 `sudo: The "no new privileges" flag is set`，说明当前终端本身运行在受限容器或沙箱中，
禁止提权；此时 `sudo docker` 和直接 `docker` 都不能连接宿主机 daemon。请在宿主机的原生终端执行构建，
或配置一个可访问 Docker daemon 的远程 Docker context。不要尝试在 Dockerfile 中安装或调用 `sudo`。

镜像内使用 Python 3.9、`uv`、GarmentCode 和 NvidiaWarp-GarmentCode；Warp 的 CUTLASS 子模块在构建时递归下载。SMPL 模型不放入镜像，运行时只读挂载。

## 3. Stage 1

```bash
.venv/bin/python -m TemplatePattern_final stage1 \
  --style long_sleeve \
  --body-config body-configs/body_001.json \
  --output-dir TemplatePattern_final/outputs/stage1/body_001
```

Docker 等价命令：

```bash
docker run --rm \
  -v "$PWD/body-configs:/data/inputs:ro" \
  -v "$PWD/TemplatePattern_final/outputs/stage1:/data/stage1" \
  templatepattern:latest stage1 \
  --style long_sleeve \
  --body-config /data/inputs/body_001.json \
  --output-dir /data/stage1/body_001
```

Stage 1 产物固定写入 `TemplatePattern_final/outputs/stage1/body_001/`，包含 `stage1_manifest.json`、`pattern.json`、`pattern.svg`、`body_target.json` 和 `stage2_request.json`。

## 4. Stage 2

```bash
.venv/bin/python -m TemplatePattern_final stage2 \
  --stage1-dir TemplatePattern_final/outputs/stage1/body_001 \
  --output-dir TemplatePattern_final/outputs/stage2/body_001/a30 \
  --pose a30
```

Docker GPU 命令：

```bash
docker run --rm --gpus all \
  -v "$PWD/TemplatePattern_final/outputs/stage1:/data/stage1:ro" \
  -v "$PWD/TemplatePattern_final/outputs/stage2:/data/stage2" \
  -v "$SMPL_MODEL_DIR:/models/smpl:ro" \
  templatepattern:latest stage2 \
  --stage1-dir /data/stage1/body_001 \
  --output-dir /data/stage2/body_001/a30 \
  --pose a30
```

Stage 2 产物固定写入 `TemplatePattern_final/outputs/stage2/body_001/a30/`，包含 `stage2_manifest.json`、`input/`、`body/` 和 `simulation/`。可用姿态为 `a30`、`a45`、`a60` 和 `tpose`；无 GPU 时可使用 `--convert-only`。

## 5. 批量运行

```bash
.venv/bin/python -m TemplatePattern_final batch-stage1 \
  --style long_sleeve \
  --input-dir body-configs \
  --output-root TemplatePattern_final/outputs/stage1
```

```bash
.venv/bin/python -m TemplatePattern_final batch-stage2 \
  --stage1-root TemplatePattern_final/outputs/stage1 \
  --output-root TemplatePattern_final/outputs/stage2 \
  --pose a30 --resume
```

批量 Stage 1 生成 `outputs/stage1/batch_manifest.json`；批量 Stage 2 只读取该 manifest 和任务目录。旧的 `python -m ...run_stage*` 入口仍保留兼容。

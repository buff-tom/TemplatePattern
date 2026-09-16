# TemplatePattern_final

运行命令与产物目录见 [两阶段部署流程](第一步和第二步部署流程总结.md)。测试脚本、验收报告及生成结果仅保留在本地，不纳入当前仓库版本。

独立的两阶段衬衫纸样与动态人体流水线。默认人体由 MHR LOD1 生成，再通过官方预计算的重心映射转换为 6890 顶点 SMPL 拓扑，因此现有分割、摆放和 GarmentCode 碰撞接口保持不变。旧 SMPL 后端仍可显式启用。

## 1. 使用 uv 本地运行

```bash
cd /home/user/buff-tomma/SewingPattern
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.12.3
uv venv --python 3.12.3 .venv
uv pip install --python .venv/bin/python \
  --index https://download.pytorch.org/whl/cpu torch==2.8.0
uv pip install --python .venv/bin/python -r TemplatePattern_final/requirements-runtime.txt
uv pip install --python .venv/bin/python -e TemplatePattern_final
```

下载 MHR v1.0.1 官方 TorchScript 模型（只提取 LOD1 模型，不需要完整 FBX/corrective 资产）：

```bash
wget -O /tmp/mhr-assets.zip \
  https://github.com/facebookresearch/MHR/releases/download/v1.0.1/assets.zip
mkdir -p TemplatePattern_final/models/mhr
unzip -j /tmp/mhr-assets.zip assets/mhr_model.pt \
  -d TemplatePattern_final/models/mhr
```

Stage 2 还需要本机准备 GarmentCode 和 NvidiaWarp-GarmentCode，并设置：

```bash
export TEMPLATEPATTERN_GARMENTCODE_ROOT=/path/to/GarmentCode
export TEMPLATEPATTERN_WARP_ROOT=/path/to/NvidiaWarp-GarmentCode
export TEMPLATEPATTERN_MHR_MODEL_DIR=/path/to/mhr-model-directory
```

MHR 模型目录中必须包含 `mhr_model.pt`。仅在使用兼容后端 `--body-model smpl` 时，才需再设置 `TEMPLATEPATTERN_SMPL_MODEL_DIR`，其目录包含：

```text
basicmodel_m_lbs_10_207_0_v1.1.0.pkl
```

本地首次使用 Warp 时，还要在 NvidiaWarp-GarmentCode 根目录执行：

```bash
.venv/bin/python /path/to/NvidiaWarp-GarmentCode/build_lib.py --cuda_path=/usr/local/cuda
```

Stage 1 不需要人体模型或 GPU；MHR 拟合可在 CPU 运行；完整 Stage 2 仿真需要 NVIDIA GPU、驱动、CUDA Toolkit 和已编译的 Warp。
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

镜像内使用 Python 3.12、`uv`、GarmentCode、NvidiaWarp-GarmentCode 和 MHR v1.0.1 TorchScript 模型；Warp 的 CUTLASS 子模块在构建时递归下载。默认 MHR 流程不再要求挂载 SMPL 模型。

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

当前 reference 的直接覆盖范围（毫米）为：身高 1500–1900、胸围 780–1160、腰围
640–1040、臀围 760–1150、肩宽 380–500、臂长 480–630。范围内优先选择六项
容量均能覆盖输入的最小合适模板；单项越界或六项组合没有模板能同时覆盖时，使用木桶回退选择
最大可满足/最接近的 reference anchor。此时：

- `requested_body_input` 保留用户原始输入；
- `effective_body_input` 和兼容字段 `body_input` 保存选中 anchor 的六项有效值；
- 纸样缩放、Stage 2 manifest 和 MHR 拟合全部使用同一组有效值。

因此 150–200cm 的身高输入都能生成 Stage 1 结果；190–200cm 中超出当前 190cm 上限的
输入会得到选定的 190cm anchor 结果，而不会要求 MHR 拟合一个与纸样模板不一致的 200cm 人体。

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
  templatepattern:latest stage2 \
  --stage1-dir /data/stage1/body_001 \
  --output-dir /data/stage2/body_001/a30 \
  --pose a30
```

Stage 2 产物固定写入 `TemplatePattern_final/outputs/stage2/body_001/a30/`，包含 `stage2_manifest.json`、`input/`、`body/` 和 `simulation/`。`body/` 同时保存 `mhr_params.json` 和兼容旧协议的 `smpl_params.json`。可用姿态为 `a30`、`a45`、`a60` 和 `tpose`；无 GPU 时可使用 `--convert-only`。

若已有同一 Stage 1 目标对应的 MHR 参数，可以跳过重新拟合：

```bash
.venv/bin/python -m TemplatePattern_final stage2 \
  --stage1-dir TemplatePattern_final/outputs/stage1/body_001 \
  --output-dir TemplatePattern_final/outputs/stage2/body_001/a30 \
  --pose a30 --mhr-params /path/to/mhr_params.json
```

程序会重新测量已有参数；若它与 Stage 1 六项目标不符则直接失败，不会把错误体型送入仿真。
已有文件可使用 MHR 原生的 `identity_coeffs`/`lbs_model_params`，也可使用 SAM3D 常见的 `shape_params`/`mhr_model_params` 命名。Stage 2 保留其中的 identity、柔性比例和骨骼 scale，但会清除原始姿态，再按命令中的 `--pose` 生成目标姿态。

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

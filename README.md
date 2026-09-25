# 桌面机械臂 VLM 实验台

Panda 机械臂 + 桌面三色积木 + 正面/腕部相机。用中文看图问答，或者让 Qwen3-VL 调用抓取、摆放、堆叠技能，并在浏览器查看结果。

**能力边界**：VLM 只接收正面 RGB 图像、用户指令和已执行技能记录。技能控制器使用仿真物体位置，通过 OSC 控制、夹爪接触和物理仿真执行动作；没有瞬移、焊接物体或把答案坐标交给模型。这不是端到端 VLA，也不代表真实机器人能力。第一版不包含模型训练。

## 环境与安装

在远程 Linux / NVIDIA GPU 主机安装和运行；本地只编辑代码和打开浏览器。测试环境使用 Python 3.11、robosuite 1.5.2、MuJoCo 3.3.7、PyTorch 2.11（CUDA 13.0）和 Transformers 4.57.6。需要兼容的 NVIDIA 驱动、EGL 和 `uv`。

```bash
git clone https://github.com/zweiyao/tabletop-vla-playground.git
cd tabletop-vla-playground
bash scripts/setup.sh
.venv/bin/python scripts/download_model.py
TABLETOP_GPU=0 bash scripts/run.sh
```

模型固定为 `Qwen/Qwen3-VL-8B-Instruct`，revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`，BF16 / SDPA。默认从 `hf-mirror.com` 下载，可通过 `HF_ENDPOINT=https://huggingface.co` 使用官方端点。下载器读取清单，按未下载文件核算空间，保留 8 GiB 余量；权重约 17.5 GB，依赖另外占用空间。

服务仅监听远程 `127.0.0.1:7860`。在本地建立转发后打开 <http://localhost:7860>：

```bash
ssh -N -L 7860:127.0.0.1:7860 -o IdentitiesOnly=yes \
  -i ~/.ssh/zhangweiyao_ssh zhangweiyao@aigc
```

启动器要求所选 GPU 已用显存不超过 512 MiB，模型与 EGL 渲染绑定同一张物理卡。若卡被占用，通过 `TABLETOP_GPU` 指定其他空闲卡。不要同时启动多个使用同一张卡的实例。运行测试前先停止网页服务。

## 使用

- 问答：“桌上有哪些颜色的积木？”、“红色在蓝色左边还是右边？”
- 抓取：“抓起绿色积木”。抓起后保持夹持，下一条可要求放下。
- 放置：“把红色积木放到左侧区域”。`left` / `center` / `right` 均按正面相机视角。
- 堆叠：“把红色积木叠在蓝色积木上”。
- 每轮最多五个技能，每次执行后重新看图；失败或点击停止后终止本轮。用“重置场景”恢复初始状态。

每个交互保存到 `runs/<UTC时间>/`，包含输入、模型原始输出、技能结果、耗时、观察图片和操作视频。`runs/` 不提交到 Git。网页是经 SSH 访问的单用户共享场景，不提供多用户场景隔离。

## 可替换接口

`tabletop.vlm.VLM.infer(image, instruction, history, stop)` 返回经过严格校验的 JSON：

```json
{"type":"answer","text":"中文回答"}
```

或者单次技能调用：

```json
{"type":"skill","skill":"stack","object":"red","target":"blue"}
```

`pick` 不带 `target`；`place` 的目标为 `left/center/right`；`stack` 的目标为另一块积木。非法字段、未知物体、自堆叠和非法动作拒绝执行。

VLA 接口位于 `tabletop.policy`：`Policy.predict(Observation) -> (T,7)`。观察包含正面/腕部 uint8 RGB 图像、机械臂本体状态和语言指令，不包含物体真实位置。动作在世界坐标系表达：XYZ 位移（米）、旋转向量（弧度）、夹爪开合（-1 开、+1 闭），20 Hz，单步限幅分别为 ±0.025 m 和 ±0.25 rad，动作块长度 1–20。`to_controller_actions` 完成校验和 OSC 归一化，`HoldPolicy` 提供最小接口示例。接入具体预训练 VLA 时仍需匹配其机器人、相机及动作归一化约定。

## 远程测试

按顺序运行，避免多个测试进程同时占卡：

```bash
.venv/bin/python -m pytest -q tests
TABLETOP_GPU=0 .venv/bin/python scripts/evaluate_physics.py
TABLETOP_GPU=0 .venv/bin/python scripts/evaluate_vlm.py
```

- 环境：20 个随机种子，检查渲染、初始化不重叠、重置可复现。
- 控制器：每类技能 20 次，至少 18 次成功；成功判定检查实际抬升/夹持或释放后的位置和稳定性。
- 问答：30 个颜色、数量、图像左右关系问题，报告正确率和失败截图。
- 指令：20 条中文指令，分别报告首次技能解析、最终物理结果和闭环完整成功率，目标至少 18 次完整成功。
- 异常：非法输出、不存在的物体、动作越界、超时和停止，检查拒绝执行。

这些是固定简单场景的工程验收，不是通用机器人 benchmark。调试种子 0–19；模型问答种子 100–109；指令种子 200–219。原始记录保存在远程 `runs/`，汇总在 `reports/`。

## 项目结构

- `src/tabletop/`：场景、技能、VLM、VLA 接口、交互循环与网页。
- `scripts/`：安装、模型下载和验收入口。
- `tests/`：结构化输出和动作契约测试。
- `reports/`：可公开的测试结果与小型示例截图。

项目使用 MIT 许可证。依赖和模型保留原许可证，详见 [THIRD_PARTY.md](THIRD_PARTY.md)。

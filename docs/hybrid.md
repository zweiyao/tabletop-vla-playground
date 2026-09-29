# π0.5 + OpenRouter 审查模式

保留原来的 Franka Panda、三个积木、相机和接触动力学。π0.5 直接预测动作；审查模型不调用抓取技能、不读取物体真值、不重置场景。

## 使用

1. 在远程项目根目录创建 `openrouter_key`，只写入一行原始密钥，权限设为 `600`。也可在 `.env` 中配置 `OPENROUTER_API_KEY`，环境变量优先。两种文件均被 Git 忽略；不要将密钥发送到聊天或填写在网页指令里。
2. 本项目通过持久账本限制本次联调累计费用 **$1**，不会因密钥账户额度更高而扩大预算，也不会修改密钥设置。可选用专用 $1 key、无周期重置并计入 BYOK 用量，作为额外的服务端限制。
3. 重启远程服务 `TABLETOP_GPU=0 bash scripts/run.sh`。启动前关闭原服务及其 π0.5 子进程，确认选定 GPU 空闲。通过现有 SSH 转发访问 `http://localhost:7860/`。
4. 选择“混合模式”，选择 π0.5 权重和审查 VLM，点击“启动 π0.5”，再输入指令、发送。已加载匹配权重时无需重新启动。
5. “停止”结束当前任务并保留模型；“关闭 π0.5”释放策略显存。更换权重需要先关闭再启动。审查模型可在下一次发送前切换。

未配置密钥、密钥无效、预算不足或模型不可用时，发送会停止，不执行提案动作。页面保留原有独立 Qwen、π0.5 原始权重和 LoRA 模式。

## 每轮发生什么

π0.5 根据当前观测和固定任务预测 **10×7** 动作。当前模型输出长度是 10；本版按确认后的 10 步审查，未补齐或外推到 15 步。

审查器接收原始任务、翻译后的策略任务、当前正面与腕部 RGB、基座系末端位姿、两指关节位置、初始机械臂姿态、相机外参，以及全部 10 步动作。第二轮起还提供上一轮执行前的两张图片、上一轮决策与实际执行的动作。图片大小默认 384×384。等待 API 时物理时间不前进。

模型只能返回两类结果：

- `accept`：原样执行前 5 步。
- `correct`：给出前 5 步逐步 XYZ、旋转增量修正以及夹爪 keep/open/close。校验全部通过后执行。

执行后重新观察并重新预测；剩余 5 步丢弃。默认最多 250 个物理控制步，即 50 次审查。VLM 不能返回“成功/结束”指令；仅用户停止、步数上限或错误结束运行。到达步数上限不代表完成任务。

动作使用实际 OSC 控制器的基座坐标系、20 Hz。归一化 XYZ 每单位 0.05 m，旋转向量每单位 0.5 rad，夹爪 -1 张开、+1 闭合。修正是加在原动作上的偏移，不是绝对位姿。每步平移修正向量范数最多 0.01 m，旋转修正最多 0.05 rad；最终七维动作必须处于 [-1,1]，超界直接拒绝，不截断后偷偷执行。模型看到的是动作提案，不能把动作累计量当成真实未来轨迹。

## 配置和模块

| 文件 | 负责内容 |
| --- | --- |
| `configs/hybrid.toml` | 执行块、图片大小、修正幅度、超时、输出长度、费用阈值、默认模型/权重 |
| `configs/models.toml` | 模型 ID、显示名、价格上限、上下文上限、JSON 格式和推理设置 |
| `prompts/hybrid_review.md` | 审查提示词，区分上轮结果和下轮意图 |
| `src/tabletop/review.py` | 输出契约与纯动作校验，无网络或仿真依赖 |
| `src/tabletop/openrouter.py` | HTTP、取消、费用记录；不导入仿真 |
| `src/tabletop/hybrid.py` | 观测整理、审查会话与审计记录 |
| `src/tabletop/engine.py` | 仿真线程和动作执行 |
| `src/tabletop/app.py` | 页面和模型生命周期按钮 |

默认从项目 `configs/` 读取；可以通过 `--config-dir` 或 `TABLETOP_CONFIG_DIR` 指定其他配置目录。提示词和账本路径相对项目根目录。配置在服务启动时读取，修改后重启；本版仍固定审查 10、执行 5，以匹配当前策略接口。

## 候选模型

2026-09-28 核对的 OpenRouter 公开价格，单位为美元/百万 token；这只是选型依据，不是该场景实测排名。图片按提供方规则计费，实际费用以 API 返回为准。

| 模型 | 输入 / 输出 | 选择理由 |
| --- | --- | --- |
| [Qwen3.8 Flash](https://openrouter.ai/qwen/qwen3.8-flash) | 0.15 / 0.47 | 图像审查候选；实测原生 schema 不稳定，改用 JSON object + 本地校验；曾遇上游限流 |
| [Qwen3.7 Flash](https://openrouter.ai/qwen/qwen3.7-flash) | 0.03 / 0.13 | 默认；通过夹爪合成错误纠正测试；使用 JSON object + 本地严格校验 |
| [Seed 2.0 Mini](https://openrouter.ai/bytedance-seed/seed-2.0-mini) | 0.10 / 0.40 | 对照；真实闭环格式稳定，但在夹爪合成错误测试中误放行 |
| [Gemini 3.1 Flash-Lite](https://openrouter.ai/google/gemini-3.1-flash-lite) | 0.25 / 1.50 | 当前远程地区实测 HTTP 403，已在配置禁用并从页面隐藏 |
| [GPT-4.1 mini](https://openrouter.ai/openai/gpt-4.1-mini) | 0.40 / 1.60 | 当前远程地区实测 HTTP 403，已在配置禁用并从页面隐藏 |

服务不会因模型下线或接口失败而自动换模型。调用前核对模型仍支持图像、JSON 和配置中的上下文范围，并向 OpenRouter 发送价格上限。配置里的 `budget_input_price` / `budget_output_price` 包含阶梯和缓存写入最高单价，用于保守预留；`image_price` 同时作为图片路由价格上限并计入最多四张图的预留。它们与表中基础 token 单价分别记录。

## 费用、停止与审计

本次联调累计限额 $1，本地停止阈值 $0.90。账本为 `runs/openrouter-budget.json`，跨任务、跨服务重启保留，所有模型共用。每次调用前按模型完整上下文、最高阶梯价格、图片费用及 5% BYOK 服务费余量计算保守预留，所以接近限额时可能提前停止；预留不是实际扣费。API 返回真实 `usage.cost` 后结算，若返回 BYOK 上游费用也累加记录。独立密钥的服务端限额是可选的额外约束。

HTTP 错误、超时、非法 JSON、错误 proposal ID、动作超界、停止后的迟到结果均不放行动作，也不自动重试。若请求取消或费用缺失，提供方可能已计费，账本保留 pending 并阻止后续付费调用。必须先停止服务，在 OpenRouter Activity 核对对应时间段请求及费用，再人工核销预留；不能通过删除账本、清零费用或更换 key 绕过累计预算。核销不确定请求时可按预留上限保守计费，保留原请求及核销依据。

每轮 `runs/<时间>/interaction.json` 保存配置、提示词哈希、模型、提案、修改、实际执行动作、耗时和费用。`review-XXXX/` 保存审查输入截图与 JSON；`review-prompt.md` 保存本轮提示词；`video.mp4` 记录执行画面。费用缺失记录为未确认，不将其宣称为免费。权重、密钥和运行记录不进 Git。

## 复现验证

所有命令在远程项目目录运行：

```bash
.venv/bin/python -m pytest -q tests
# 先关闭网页服务及其策略子进程，确认 GPU 空闲：
TABLETOP_GPU=0 .venv/bin/python scripts/probe_hybrid.py
```

单元测试使用模拟 HTTP/策略；冒烟测试使用真实 Panda 和真实 π0.5 LoRA，但审查是明确标注的确定性 mock，不产生 API 费用。它验证动作管道，不证明 VLM 纠错有效或提升任务成功率。真实 OpenRouter 联调和与纯 π0.5 的同种子对照需要配置密钥后进行，结果另行报告。

已配置密钥后，可用下列入口进行收费验证，所有入口共用累计费用账本：

```bash
# 对存档观测进行接口测试，无实际仿真动作：
.venv/bin/python scripts/probe_openrouter.py runs/<时间>/review-0000 --model bytedance-seed/seed-2.0-mini
# 使用同一观测，注入“要求张开但提案闭合”的已标注合成错误：
.venv/bin/python scripts/probe_openrouter.py runs/<时间>/review-0000 --model bytedance-seed/seed-2.0-mini --wrong-gripper
# 真实策略/混合模式同种子短程对照，需要先关闭网页服务释放 GPU：
TABLETOP_GPU=0 .venv/bin/python scripts/evaluate_hybrid.py --steps 50 --model bytedance-seed/seed-2.0-mini
```

对照脚本只统计末帧条件，不进行稳定夹持成功率验收；50 步是接口联调长度。环境种子相同，策略采样随机状态未配对，不能把个别轨迹差异归因于 VLM。每次结果放在新的时间戳目录，不覆盖既有记录。实际结果见 [接入验证报告](../reports/hybrid/README.md)。

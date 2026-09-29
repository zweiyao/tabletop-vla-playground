"""SSH-only, single-user Gradio interface; MuJoCo runs on one owner thread."""
import argparse
import os
import queue
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from .runtime import check_gpu_idle, configure_gpu


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--config-dir", help="Directory containing hybrid.toml and models.toml")
    args = parser.parse_args()
    from .hybrid_config import load_config
    settings = load_config(args.config_dir)
    hybrid_config, reviewer_models = settings
    check_gpu_idle()
    configure_gpu()
    import gradio as gr
    from .engine import Engine
    worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="simulation")
    engine = worker.submit(Engine, hybrid_settings=settings).result()
    initial = worker.submit(engine.env.images).result()
    model_busy = False

    def model_controls():
        alive = engine.pi05 is not None and engine.pi05.alive
        return gr.update(value="关闭 π0.5" if alive else "启动 π0.5", interactive=not model_busy), engine.pi05_status()

    def toggle_pi05(mode, hybrid_weight):
        nonlocal model_busy
        if model_busy:
            return
        model_busy = True
        alive = engine.pi05 is not None and engine.pi05.alive
        if alive:
            engine.stop.set()
        yield gr.update(interactive=False), "正在关闭 π0.5…" if alive else "正在启动 π0.5…"
        error = None
        try:
            if alive:
                worker.submit(engine.close_pi05).result()
            else:
                worker.submit(engine.start_pi05, hybrid_weight if mode == "hybrid" else mode).result()
        except (ValueError, RuntimeError) as exc:
            error = str(exc)
        finally:
            model_busy = False
        button, state = model_controls()
        yield button, f"{state}；{error}" if error else state

    def interact(prompt, use_wrist, mode, max_steps, hybrid_weight, reviewer_model):
        def update(front, wrist, message, details, editable):
            return front, wrist, message, *model_controls(), details, *(gr.update(interactive=editable) for _ in range(3))
        if model_busy:
            yield update(gr.skip(), gr.skip(), "π0.5 正在启动或关闭，请稍后发送。", gr.skip(), True)
            return
        if not prompt.strip():
            yield update(gr.skip(), gr.skip(), "请输入问题或指令。", gr.skip(), True)
            return
        updates = queue.Queue(maxsize=2)
        def emit(images, status):
            try:
                updates.put_nowait((images["front"], images["wrist"], status))
            except queue.Full:
                updates.get_nowait()
                updates.put_nowait((images["front"], images["wrist"], status))
        yield update(gr.skip(), gr.skip(), "正在准备执行…", {}, False)
        future = worker.submit(engine.run, prompt.strip(), emit, use_wrist=use_wrist, mode=mode,
                               max_steps=max_steps, hybrid_weight=hybrid_weight, reviewer_model=reviewer_model)
        try:
            while not future.done() or not updates.empty():
                try:
                    yield update(*updates.get(timeout=0.25), gr.skip(), False)
                except queue.Empty:
                    continue
            log = future.result()
            details = {k: log[k] for k in ("hybrid", "hybrid_metrics", "reviews", "error", "termination") if k in log}
            yield update(gr.skip(), gr.skip(), log["answer"], details, True)
        except Exception:
            yield update(gr.skip(), gr.skip(), "执行异常，请检查远程日志。", gr.skip(), True)
            raise

    def reset(seed):
        images = worker.submit(engine.reset, int(seed)).result()
        return images["front"], images["wrist"], f"已切换到场景 {int(seed)}"

    def stop():
        engine.stop.set()
        return "已请求停止，正在结束当前控制步或模型生成。"

    def change_mode(mode, seed):
        front_image, wrist_image, message = reset(seed)
        steps_update = gr.update(visible=mode != "vlm", **({"value": hybrid_config.default_max_steps} if mode == "hybrid" else {}))
        return (front_image, wrist_image, message, gr.update(interactive=mode == "vlm"), steps_update,
                *model_controls(), gr.update(visible=mode == "hybrid"), gr.update(visible=mode == "hybrid"))

    with gr.Blocks(title="桌面机械臂实验台") as demo:
        gr.Markdown("# 桌面机械臂实验台\n看图提问，或让 Panda 抓取、摆放、堆叠积木。")
        model_choices = [("Qwen VLM · 问答与抓放技能", "vlm"), ("π0.5 原始权重 · 直接动作控制", "pi05")]
        if (Path(os.environ.get("TABLETOP_PI05_LORA", "adapters/tabletop-four-v1/best")) / "adapter.safetensors").is_file():
            model_choices.append(("π0.5 LoRA · 红绿蓝抓取与回位", "pi05_lora"))
        model_choices.append(("混合模式 · π0.5 + VLM 审查", "hybrid"))
        mode = gr.Radio(choices=model_choices,
                        value="vlm", label="使用模型", info="切换模型会按当前场景编号重置机械臂。π0.5 用于动作任务，中文自动翻译为英文。")
        with gr.Column(visible=False) as hybrid_options:
            with gr.Row():
                hybrid_weight = gr.Dropdown(choices=[("四任务 LoRA", "pi05_lora"), ("原始权重", "pi05")],
                                           value=hybrid_config.default_weight, label="π0.5 权重", interactive=True)
                reviewer_model = gr.Dropdown(choices=[(m.label, m.id) for m in reviewer_models.values() if not m.disabled_reason],
                                            value=hybrid_config.default_model, label="审查 VLM · OpenRouter", interactive=True)
            key_state = "已配置密钥，发送时检查额度。" if os.getenv("OPENROUTER_API_KEY") else "尚未配置密钥：请在远程 .env 配置 OPENROUTER_API_KEY 后重启服务。"
            gr.Markdown("审查未来 10 步，每次执行前 5 步；失败即停止。本轮联调总预算上限 $1。\n\n" + key_state)
        with gr.Row():
            pi05_toggle = gr.Button("启动 π0.5")
            pi05_state = gr.Textbox(value="π0.5 未启动", label="π0.5 运行状态", interactive=False)
        gr.Markdown("选择 π0.5 权重后，先启动再发送。更换权重需先关闭再启动；停止仅结束当前动作，关闭会释放 π0.5 显存。")
        with gr.Row():
            front = gr.Image(value=initial["front"], label="正面相机 · 左右以此视角为准", interactive=False)
            wrist = gr.Image(value=initial["wrist"], label="腕部相机", interactive=False)
        prompt = gr.Textbox(label="问题或指令", placeholder="例如：把红色积木叠在蓝色积木上")
        use_wrist = gr.Checkbox(value=False, label="同时使用腕部相机提问模型",
                               info="仅影响 Qwen 问答。π0.5 固定使用斜视和腕部；混合审查固定使用正面和腕部。")
        max_steps = gr.Slider(50, 1000, value=300, step=50, label="π0.5 最多执行步数", visible=False)
        with gr.Row():
            submit = gr.Button("发送", variant="primary")
            stop_btn = gr.Button("停止")
            seed = gr.Number(value=0, precision=0, minimum=0, label="场景编号")
            reset_btn = gr.Button("重置场景")
        status = gr.Textbox(label="回答与执行状态", lines=8, interactive=False)
        with gr.Accordion("审查详情 · 动作、耗时与费用", open=False, visible=False) as hybrid_details:
            details = gr.JSON(label="本轮审查记录")
        gr.Examples(["桌上有哪些颜色的积木？", "红色积木在蓝色积木的左边还是右边？",
                     "抓起绿色积木", "空夹爪回到初始位置和朝向，并保持张开",
                     "把红色积木放到左侧", "把红色积木叠在蓝色积木上"], prompt)
        gr.Markdown("VLM 使用抓放技能；π0.5 根据图像和指令直接预测动作。LoRA 针对三色积木抓取和空夹爪回位训练，实际成功率请参阅实验报告。")
        pi05_toggle.click(toggle_pi05, [mode, hybrid_weight], [pi05_toggle, pi05_state], concurrency_id="model", concurrency_limit=1)
        inputs = [prompt, use_wrist, mode, max_steps, hybrid_weight, reviewer_model]
        outputs = [front, wrist, status, pi05_toggle, pi05_state, details, mode, hybrid_weight, reviewer_model]
        submit.click(interact, inputs, outputs, concurrency_id="scene", concurrency_limit=1)
        prompt.submit(interact, inputs, outputs, concurrency_id="scene", concurrency_limit=1)
        mode.input(change_mode, [mode, seed], [front, wrist, status, use_wrist, max_steps, pi05_toggle, pi05_state,
                                               hybrid_options, hybrid_details], concurrency_id="scene", concurrency_limit=1)
        reset_btn.click(reset, seed, [front, wrist, status], concurrency_id="scene", concurrency_limit=1)
        stop_btn.click(stop, outputs=status, queue=False)
        demo.load(model_controls, outputs=[pi05_toggle, pi05_state])
    demo.queue(max_size=4).launch(server_name="127.0.0.1", server_port=args.port, share=False)


if __name__ == "__main__":
    main()

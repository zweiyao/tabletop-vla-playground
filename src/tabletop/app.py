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
    args = parser.parse_args()
    check_gpu_idle()
    configure_gpu()
    import gradio as gr
    from .engine import Engine
    worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="simulation")
    engine = worker.submit(Engine).result()
    initial = worker.submit(engine.env.images).result()
    model_busy = False

    def model_controls():
        alive = engine.pi05 is not None and engine.pi05.alive
        return gr.update(value="关闭 π0.5" if alive else "启动 π0.5", interactive=not model_busy), engine.pi05_status()

    def toggle_pi05(mode):
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
                worker.submit(engine.start_pi05, mode).result()
        except (ValueError, RuntimeError) as exc:
            error = str(exc)
        finally:
            model_busy = False
        button, state = model_controls()
        yield button, f"{state}；{error}" if error else state

    def interact(prompt, use_wrist, mode, max_steps):
        if model_busy:
            yield gr.skip(), gr.skip(), "π0.5 正在启动或关闭，请稍后发送。", *model_controls()
            return
        if not prompt.strip():
            yield initial["front"], initial["wrist"], "请输入问题或指令。", *model_controls()
            return
        updates = queue.Queue(maxsize=2)
        def emit(images, status):
            try:
                updates.put_nowait((images["front"], images["wrist"], status))
            except queue.Full:
                updates.get_nowait()
                updates.put_nowait((images["front"], images["wrist"], status))
        future = worker.submit(engine.run, prompt.strip(), emit, use_wrist=use_wrist, mode=mode, max_steps=max_steps)
        while not future.done() or not updates.empty():
            try:
                yield *updates.get(timeout=0.25), *model_controls()
            except queue.Empty:
                continue
        future.result()

    def reset(seed):
        images = worker.submit(engine.reset, int(seed)).result()
        return images["front"], images["wrist"], f"已切换到场景 {int(seed)}"

    def stop():
        engine.stop.set()
        return "已请求停止，正在结束当前控制步或模型生成。"

    def change_mode(mode, seed):
        front_image, wrist_image, message = reset(seed)
        return front_image, wrist_image, message, gr.update(interactive=mode == "vlm"), gr.update(visible=mode != "vlm"), *model_controls()

    with gr.Blocks(title="桌面机械臂实验台") as demo:
        gr.Markdown("# 桌面机械臂实验台\n看图提问，或让 Panda 抓取、摆放、堆叠积木。")
        model_choices = [("Qwen VLM · 问答与抓放技能", "vlm"), ("π0.5 原始权重 · 直接动作控制", "pi05")]
        if (Path(os.environ.get("TABLETOP_PI05_LORA", "adapters/tabletop-four-v1/best")) / "adapter.safetensors").is_file():
            model_choices.append(("π0.5 LoRA · 红绿蓝抓取与回位", "pi05_lora"))
        mode = gr.Radio(choices=model_choices,
                        value="vlm", label="使用模型", info="切换模型会按当前场景编号重置机械臂。π0.5 用于动作任务，中文自动翻译为英文。")
        with gr.Row():
            pi05_toggle = gr.Button("启动 π0.5")
            pi05_state = gr.Textbox(value="π0.5 未启动", label="π0.5 运行状态", interactive=False)
        gr.Markdown("选择 π0.5 权重后，先启动再发送。更换权重需先关闭再启动；停止仅结束当前动作，关闭会释放 π0.5 显存。")
        with gr.Row():
            front = gr.Image(value=initial["front"], label="正面相机 · 左右以此视角为准", interactive=False)
            wrist = gr.Image(value=initial["wrist"], label="腕部相机", interactive=False)
        prompt = gr.Textbox(label="问题或指令", placeholder="例如：把红色积木叠在蓝色积木上")
        use_wrist = gr.Checkbox(value=False, label="同时使用腕部相机提问模型",
                               info="仅用于 VLM。π0.5 固定使用斜视相机与腕部相机，显示画面仍为正面和腕部。")
        max_steps = gr.Slider(50, 1000, value=300, step=50, label="π0.5 最多执行步数", visible=False)
        with gr.Row():
            submit = gr.Button("发送", variant="primary")
            stop_btn = gr.Button("停止")
            seed = gr.Number(value=0, precision=0, minimum=0, label="场景编号")
            reset_btn = gr.Button("重置场景")
        status = gr.Textbox(label="回答与执行状态", lines=8, interactive=False)
        gr.Examples(["桌上有哪些颜色的积木？", "红色积木在蓝色积木的左边还是右边？",
                     "抓起绿色积木", "空夹爪回到初始位置和朝向，并保持张开",
                     "把红色积木放到左侧", "把红色积木叠在蓝色积木上"], prompt)
        gr.Markdown("VLM 使用抓放技能；π0.5 根据图像和指令直接预测动作。LoRA 针对三色积木抓取和空夹爪回位训练，实际成功率请参阅实验报告。")
        pi05_toggle.click(toggle_pi05, mode, [pi05_toggle, pi05_state], concurrency_id="model", concurrency_limit=1)
        submit.click(interact, [prompt, use_wrist, mode, max_steps], [front, wrist, status, pi05_toggle, pi05_state], concurrency_id="scene", concurrency_limit=1)
        prompt.submit(interact, [prompt, use_wrist, mode, max_steps], [front, wrist, status, pi05_toggle, pi05_state], concurrency_id="scene", concurrency_limit=1)
        mode.change(change_mode, [mode, seed], [front, wrist, status, use_wrist, max_steps, pi05_toggle, pi05_state], concurrency_id="scene", concurrency_limit=1)
        reset_btn.click(reset, seed, [front, wrist, status], concurrency_id="scene", concurrency_limit=1)
        stop_btn.click(stop, outputs=status, queue=False)
        demo.load(model_controls, outputs=[pi05_toggle, pi05_state])
    demo.queue(max_size=4).launch(server_name="127.0.0.1", server_port=args.port, share=False)


if __name__ == "__main__":
    main()

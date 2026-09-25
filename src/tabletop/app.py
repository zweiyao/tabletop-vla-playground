"""SSH-only, single-user Gradio interface; MuJoCo runs on one owner thread."""
import argparse
import queue
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

    def interact(prompt):
        if not prompt.strip():
            yield initial["front"], initial["wrist"], "请输入问题或指令。"
            return
        updates = queue.Queue(maxsize=2)
        def emit(images, status):
            try:
                updates.put_nowait((images["front"], images["wrist"], status))
            except queue.Full:
                updates.get_nowait()
                updates.put_nowait((images["front"], images["wrist"], status))
        future = worker.submit(engine.run, prompt.strip(), emit)
        while not future.done() or not updates.empty():
            try:
                yield updates.get(timeout=0.25)
            except queue.Empty:
                continue
        future.result()

    def reset(seed):
        images = worker.submit(engine.reset, int(seed)).result()
        return images["front"], images["wrist"], f"已重置，随机种子 {int(seed)}"

    def stop():
        engine.stop.set()
        return "已请求停止，正在结束当前控制步或模型生成。"

    with gr.Blocks(title="桌面机械臂实验台") as demo:
        gr.Markdown("# 桌面机械臂实验台\n看图提问，或让 Panda 抓取、摆放、堆叠积木。")
        with gr.Row():
            front = gr.Image(value=initial["front"], label="正面相机 · 左右以此视角为准", interactive=False)
            wrist = gr.Image(value=initial["wrist"], label="腕部相机", interactive=False)
        prompt = gr.Textbox(label="问题或指令", placeholder="例如：把红色积木叠在蓝色积木上")
        with gr.Row():
            submit = gr.Button("发送", variant="primary")
            stop_btn = gr.Button("停止")
            seed = gr.Number(value=0, precision=0, label="场景种子")
            reset_btn = gr.Button("重置场景")
        status = gr.Textbox(label="回答与执行状态", lines=8, interactive=False)
        gr.Examples(["桌上有哪些颜色的积木？", "红色积木在蓝色积木的左边还是右边？",
                     "抓起绿色积木", "把红色积木放到左侧", "把红色积木叠在蓝色积木上"], prompt)
        gr.Markdown("VLM 看图理解；抓放技能使用仿真位置完成控制。第一版不是端到端 VLA。")
        submit.click(interact, prompt, [front, wrist, status], concurrency_id="scene", concurrency_limit=1)
        prompt.submit(interact, prompt, [front, wrist, status], concurrency_id="scene", concurrency_limit=1)
        reset_btn.click(reset, seed, [front, wrist, status], concurrency_id="scene", concurrency_limit=1)
        stop_btn.click(stop, outputs=status, queue=False)
    demo.queue(max_size=4).launch(server_name="127.0.0.1", server_port=args.port, share=False)


if __name__ == "__main__":
    main()

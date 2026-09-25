# Third-party components

This repository's original code uses the MIT license. Dependencies and separately downloaded weights retain their own licenses:

- [robosuite](https://github.com/ARISE-Initiative/robosuite): MIT; installed as a dependency, including Panda/table assets shipped by robosuite. See its bundled LICENSE and asset notices.
- [MuJoCo](https://github.com/google-deepmind/mujoco): Apache-2.0.
- [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct): Apache-2.0; weights are downloaded separately, not redistributed here.
- [Transformers](https://github.com/huggingface/transformers): Apache-2.0.
- [Gradio](https://github.com/gradio-app/gradio): Apache-2.0.
- [PyTorch](https://github.com/pytorch/pytorch): BSD-style license and third-party notices distributed with the package.

The environment is composed through robosuite's public APIs. No LIBERO/RLinf code or checkpoint is required by this project.

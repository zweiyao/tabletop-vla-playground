# Third-party components

This repository's original code uses the MIT license. Dependencies and separately downloaded weights retain their own licenses:

- [robosuite](https://github.com/ARISE-Initiative/robosuite): MIT; installed as a dependency, including Panda/table assets shipped by robosuite. See its bundled LICENSE and asset notices.
- [MuJoCo](https://github.com/google-deepmind/mujoco): Apache-2.0.
- [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct): Apache-2.0; weights are downloaded separately, not redistributed here.
- [Transformers](https://github.com/huggingface/transformers): Apache-2.0.
- [Gradio](https://github.com/gradio-app/gradio): Apache-2.0.
- [PyTorch](https://github.com/pytorch/pytorch): BSD-style license and third-party notices distributed with the package.
- [openpi](https://github.com/Physical-Intelligence/openpi) and [RLinf](https://github.com/RLinf/RLinf): Apache-2.0; optional π0.5 inference runtime, not vendored here.
- [RLinf-Pi05-LIBERO-SFT](https://huggingface.co/RLinf/RLinf-Pi05-LIBERO-SFT): separately hosted model weights; retain the model card and upstream model/dependency terms. Weights are not redistributed here.

The environment is composed through robosuite's public APIs. Qwen mode does not require LIBERO/RLinf; the optional π0.5 mode requires the separate RLinf/openpi runtime and checkpoint described in README.

# Execution rules
- Edit/read code locally; install, run and test only on zhangweiyao@aigc in ~/holmes/tabletop-vla-playground.
- Use ~/.ssh/zhangweiyao_ssh with IdentitiesOnly=yes.
- Default to one physical GPU for inference and EGL rendering. Check actual GPU use before launch; maximum four GPUs across our workloads.
- Do not resume or modify the paused simulation/LIBERO experiments.
- Preserve remote weights, caches and results during synchronization; never use rsync --delete.
- Do not commit model weights, credentials, virtual environments or large experiment artifacts.

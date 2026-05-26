# Debug Session: openmp-runtime-conflict

Status: [OPEN]

## Problem
- 运行 `python rag_ragas_eval.py` 测试集后报错：
- `OMP: Error #15: Initializing libiomp5md.dll, but found libomp140.x86_64.dll already initialized.`

## Scope
- 先收集运行时证据，不修改业务逻辑。
- 目标是定位哪个依赖链同时加载了多个 OpenMP runtime。

## Hypotheses
- H1: `faiss` / `numpy` / `scikit-learn` 一侧加载了 Intel OpenMP (`libiomp5md.dll`)，而 `torch` / `sentence_transformers` / `onnxruntime` 一侧加载了 LLVM OpenMP (`libomp140.x86_64.dll`)。
- H2: `rag_ragas_eval.py` 在评估过程中同时触发了检索链和 `ragas` 依赖链，导致两个运行库在同一进程内被重复初始化。
- H3: 当前 Python 环境 PATH 中存在额外的 OpenMP DLL 搜索路径，优先级导致重复加载了不同发行版的 runtime。
- H4: 某个可选依赖在 import 阶段就触发 OpenMP 初始化，即使实际没有走到对应功能，也会造成冲突。
- H5: 问题不是业务代码导致，而是虚拟环境内二进制包版本组合不兼容；脚本只能规避或延迟导入，根因在环境。

## Evidence Log
- `D:\workplace\Crawl\.venv\Lib\site-packages\faiss_cpu.libs\` 下存在 `libomp140.x86_64-*.dll`
- `D:\workplace\Crawl\.venv\Lib\site-packages\torch\lib\` 下存在 `libiomp5md.dll`
- 最小复现 1：
  - 先执行 `faiss` 检索，再 `import torch`
  - 结果：`Initializing libiomp5md.dll, but found libomp140.x86_64.dll already initialized`
- 最小复现 2：
  - 先 `import torch`，再执行 `faiss` 检索
  - 结果：`Initializing libomp140.x86_64.dll, but found libiomp5md.dll already initialized`
- 结论：
  - `faiss` 与 `torch` 同进程共存时会发生双 OpenMP runtime 冲突
  - 当前 `rag_ragas_eval.py` 的兼容层中，`try_load_ragas_runtime()` 会尝试真实导入 `transformers`
  - 在该虚拟环境里，`transformers` 会继续导入 `torch`，从而把 `libiomp5md.dll` 拉进来
  - 因此冲突很可能发生在“先执行 FAISS 检索，再加载 ragas/transformers”这一阶段

## Next Steps
- 修改 `rag_ragas_eval.py` 的 ragas shim，避免为兼容目的去真实导入 `transformers`
- 验证 `faiss` 运行后再调用 `try_load_ragas_runtime()` 是否仍触发 OMP 冲突

## Fix Applied
- 已修改 `rag_ragas_eval.py`：
  - 在 `_install_ragas_import_shims()` 中不再真实 `import transformers`
  - 直接注入最小 `transformers` stub，仅提供 `GPT2TokenizerFast`
  - 目的：避免 `transformers -> torch -> libiomp5md.dll` 在 FAISS 已加载 `libomp140` 后再次进入同一进程

## Post-Fix Evidence
- 复现链路：
  - 先执行 FAISS 检索
  - 再 `import rag_ragas_eval as m; m.try_load_ragas_runtime()`
- 修复前：
  - 直接报 `OMP: Error #15`
- 修复后：
  - 不再出现 OMP 冲突
  - 当前新的首个错误变为：`nltk is required for bleu score. Please install it using pip install nltk`

## Current Conclusion
- H1 Confirmed: `faiss` 与 `torch` 的 OpenMP runtime 冲突存在
- H2 Confirmed: `rag_ragas_eval.py` 的 ragas 兼容层曾通过 `transformers` 间接拉起 `torch`
- H3 Rejected: 目前证据不足以表明是系统 PATH 额外 DLL 污染主导
- H4 Confirmed: 冲突确实发生在 import/初始化阶段
- H5 Confirmed: 根因是环境中二进制依赖组合，代码侧已通过“避免不必要导入 torch 链”完成规避

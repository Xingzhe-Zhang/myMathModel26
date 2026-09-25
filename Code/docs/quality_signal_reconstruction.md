# A1/A18 质量信号重建：9 项统计 + 5 项公开模型通过总体审计

> 运行状态（2026-09-25）：正式脚本已生成 A1/A18 信号矩阵与 17 项 A1 误差表，14 项通过、3 项未通过。该入口**没有**训练 A4–A15 Loss 模型；十四项质量迁移与 Loss 实验的完整协议见[任务 3 实验设计](../../Q1/报告/任务3十四项质量桥接与配比实验设计.md)。

从仓库根目录在 PowerShell 执行：

```powershell
& 'Code/.venv/Scripts/python.exe' Code/scripts/run_quality_signal_reconstruction.py
```

当前工作区的 `Code/.venv` 已安装依赖。若在另一台机器复现，先用 Python ≥3.10 创建 `Code/.venv`，再运行 `& 'Code/.venv/Scripts/python.exe' -m pip install -e './Code[quality-models]'`。本机普通 `python` 命令不在 PATH 中，不应直接照抄未指定解释器的命令。

脚本对 A1 七域、A18 十七域分别做固定哈希抽样，每域默认最多 64 条；A1 必须达到目标数，A18 不足 64 条的领域保留全部可用文本，仍要求每域至少有一条。A1 官方 Q 使用任务 1/2 的 `sample_scores.csv`，因此先确保该文件和 `domain_summary.csv` 已生成。A18 只读取 `valid/` 文本。`sample_coverage.csv` 会在模型推断前写出各域实际可用数、入样数和短缺量；例如 `ubuntu_irc` 只有 16 条时，全部 16 条入样，后续映射须标记样本量不足和较高不确定性。默认每个模型截取文本前 8192 字符、最多 512 tokens、批量 4；这些参数写在 `configs/q1_quality_mixture_experiment.json`。有可用 CUDA 版 PyTorch 时可加 `--device cuda`；CPU 可明确加 `--device cpu`。六个固定版本的公开权重已在本机小样本试跑时下载到 `Code/.cache-hf/`；正式 CPU 推断可能需要数小时。运行时按模型与文本指纹复用缓存，单模型推断中断后可从 `.partial.npy` 继续。

该入口只输出质量信号，不读取 A4–A15，也不训练配比→Loss 模型。主要结果在 `Code/outputs/q1_quality_reconstruction_experiment/`：

- `rps_a1_audit.csv`：11 个统计量与 A1 官方字段的误差；
- `signal_a1_audit.csv`：11+6 个字段的官方均值、重算均值、MAE、相关系数与是否通过；
- `signal_a1_audit_by_domain.csv`：逐质量域误差，检查 pooled 相关是否掩盖失配；
- `accepted_fields.json`：通过 A1 校验且在 A18 至少有有效值的共同字段；A18 少量非有限文本统计量按 A1 中位数用于域画像，并在 `a18_feature_missingness.csv` 记录，任务 1/2 评分器仍按原缺失规则处理；
- `reconstructed_signals.npz`：后续 Model-2/3 使用的矩阵；
- `model_registry.json`：模型来源、固定提交版本、输入截断与样本量。
- `sample_coverage.csv`：A1/A18 每域可用数、实际入样数和不足目标数的标记。

五个字段使用 FineWeb-Edu 与 Meta-rater 同族公开权重。`fluency_en` 的万卷原始权重暂未找到公开下载地址，脚本用 CoLA 语法可接受度分类器作为**候选代理**；若它与 A1 官方流畅度字段不一致，自动剔除，不会把它写成已复现的官方模型。该代理的固定版本在本机缓存为 `pytorch_model.bin`：加载时优先离线读取，避免网络故障下对不存在的 `model.safetensors` 发起探测；若本机没有完整缓存，仍需联网下载。11 个统计量全部重算，但只有通过 A1 误差门槛的字段才可用于 Q 迁移。

若只是检查程序是否可运行，可加 `--per-domain 4 --out Code/outputs/q1_quality_reconstruction_pilot`。这个小样本只能排查运行问题，不能作为论文验证。正式运行生成的 `signal_a1_audit.csv`、`signal_a1_audit_by_domain.csv`、`accepted_fields.json`、`model_registry.json` 已位于 `Code/outputs/q1_quality_reconstruction_experiment/`。后续应先核验十四项 A16 软映射和任务 1/2 冻结评分器的跨域表现；**总体字段通过不等于 Q 跨域通过**。当前九项阶段已有的 Model-2/3 Loss 结果不自动升级为十四项结果。

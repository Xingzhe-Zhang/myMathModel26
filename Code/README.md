# 问题一代码与复现

`Code/` 实现问题一的质量评分与冲突诊断、A22/B23/B20 维度对照，以及 17 域配比对 13 项验证 Loss 的建模。从仓库根目录运行以下命令；所有脚本都能在未安装本地包的情况下找到 `src/`。

| 目录 | 内容 |
|---|---|
| `scripts/` | 质量评分、A/B 维度实验、配比建模和质量迁移入口 |
| `configs/` | 与运行入口对应的版本化配置 |
| `src/llm_resource/q1/` | 数据读取、预处理、评分、诊断、聚合与模型实现 |
| `analysis/` | 复算评审和维度统计结论的辅助脚本及评审证据 |
| `docs/` | 实验协议与标注格式说明 |
| `tests/` | 核心逻辑测试 |
| `outputs/` | 生成的结果，已由 `.gitignore` 排除 |

## 任务 1、2：质量评分与冲突诊断

```powershell
python Code/scripts/run_quality.py `
  --data-root real_attachments `
  --config Code/configs/q1_quality.json `
  --out Code/outputs/q1
```

脚本在 A1 拟合并冻结效用尺度、缺失值处理和阈值，再处理 A1/A2/A3。该入口保留原 22 字段等权基线，供既有任务 3 重建程序复用；新的非等权 A22 由下一节的 A/B 实验入口生成。有限补偿的 `Q_rule` 和惩罚强度情景也会输出，但在缺少独立标签时不充当主结果。主要结果包括 `sample_scores.csv`、`domain_summary.csv`、`conflict_pairs_*.csv`、`score_sensitivity_by_domain.csv` 和 `run_metadata.json`。快速检查可追加 `--bootstrap 10 --permutations 20`。

## A22、B23、B20 维度实验

```powershell
& 'Code/.venv/Scripts/python.exe' Code/scripts/run_ab_experiment.py `
  --data-root real_attachments `
  --quality-config Code/configs/q1_quality.json `
  --config Code/configs/q1_ab_experiment.json `
  --out Code/outputs/q1_ab_experiment
```

这个入口在同一 A1 冻结尺度下输出完整 CRITIC 加权 A22、QuRating 拆分且 DSIR 合并的 B23、QuRating 与 DSIR 双家族合并的 B20，并在 A1/A2/A3 上做排序、权重、归一化及收缩系数敏感性比较。各方案同时输出有限补偿候选分与共同原子信号冲突诊断。具体协议见 [维度实验说明](docs/ab_experiment.md)，结果解读见 [A/B 对照报告](../Q1/报告/任务1-2_质量评分对照.md)。无独立标签或受控训练反馈时，不能裁决哪套更接近真实训练价值。

保留的 22 维等权逐条记录可与三方案直接对照：`python Code/analysis/equal_baseline_comparison.py`。脚本核对 UID 和原评分一致性后生成 `Code/outputs/q1_ab_experiment/equal_baseline_comparison.csv`，不改写原记录。

## 任务 3：无质量配比基线与领域效应

```powershell
python Code/scripts/run_mixture.py `
  --data-root real_attachments `
  --config Code/configs/q1_mixture.json `
  --out Code/outputs/q1_task3
```

A4+A5 用于分组交叉验证与模型选择；A6–A11 用于外部检验；A12–A15 只用于外推一致性诊断。此入口的 Loss 模型仅使用配比 \(p\)，另以 A16–A18 审计质量映射的覆盖和文本来源；这些审计结果不进入本入口的基线模型。`quality_bridge_domain.csv`、`quality_bridge_recipe.csv` 和 `regmix_text_evidence.csv` 给出映射覆盖与缺失域的质量情景界。其他结果包括 `model_cv_summary.csv`、`external_metrics.csv`、`replacement_effects.csv`、`combination_effect_summary.csv` 和 `mixture_model.json`；解读见 [配比基线报告](../Q1/报告/任务3_配比基线.md)。

## 任务 3：质量信号审计与配比预测

公开模型信号重建入口已运行完成，A1 总体审计通过 9 项统计字段与 5 项公开模型字段。此入口只生成信号证据，不训练 Loss 模型：

```powershell
& 'Code/.venv/Scripts/python.exe' Code/scripts/run_quality_signal_reconstruction.py
```

运行前先按任务 1、2 命令生成 `sample_scores.csv`、`domain_summary.csv` 和 `fitted_scoring_model.json`。程序从 A1/A18 文本重算全部 11 个 RedPajama 统计量，调用固定版本的五个同族公开分类器，并以 CoLA 语法可接受度模型作为第六项 `fluency_en` 的候选代理。原万卷流畅度权重未公开；该代理只有在 A1 对照通过后才会用于质量迁移。A1 逐信号核验只保留通过预设门槛的字段。

公开模型入口**只重建并核验信号，不运行 Loss**；详细步骤与审计产物见 [质量信号重建指南](docs/quality_signal_reconstruction.md)。`Code/outputs/q1_quality_reconstruction_experiment/accepted_fields.json` 列出 14 项；`fluency_en` 代理与两项行级统计未过审计。配比—Loss 对照由下一入口独立执行。

### 质量增强配比实验

```powershell
& 'Code/.venv/Scripts/python.exe' Code/scripts/run_quality_mixture_14_stage.py --phase all
```

此入口先按冻结的 B20 原尺度基础分构建十四项画像桥接和直接评分，输出 A16 锚点、A1 留一域质量门禁；仅过关路线进入 A4+A5 五折选模及 A6–A11 实际 Loss 检验。A12–A15 只作估算外推诊断。结果在 `Code/outputs/q1_quality_mixture_14_stage/`，详见 [十四项完整实验报告](../Q1/报告/任务3_实验结果.md) 和 [实验前设计](../Q1/报告/任务3_实验方案.md)。`--phase quality`、`--phase loss` 可分阶段复算；程序不修改信号重建入口。

题外配方筛选诊断脚本 `Code/analysis/q1_task3_quality_diagnostic.py` 不作为问题一任务 3 的模型判定依据。正式结果以 13 个验证域 Loss 的预测检验与领域/组合分析为准。

## 辅助复算与测试

```powershell
python Code/analysis/review_audit.py
python -m unittest discover -s Code/tests -p "test_*.py" -v
```

`review_audit.py` 会重写同目录的 `review_evidence.json`。A/B 实验的权重、效用审计和敏感性表由 `run_ab_experiment.py` 一次生成。

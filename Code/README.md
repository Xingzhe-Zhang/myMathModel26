# 问题一代码与复现

`Code/` 实现问题一的质量评分与冲突诊断、两套维度方案对照，以及 17 域配比对 13 项验证 Loss 的建模。从仓库根目录运行以下命令；所有脚本都能在未安装本地包的情况下找到 `src/`。

| 目录 | 内容 |
|---|---|
| `scripts/` | 三个正式运行入口：质量评分、维度对照、配比建模 |
| `configs/` | 与三个入口对应的版本化配置 |
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

脚本在 A1 拟合并冻结效用尺度、缺失值处理和阈值，再处理 A1/A2/A3。默认的 `Q_primary` 等于 22 字段等权的 `Q_base`；有限补偿的 `Q_rule` 和惩罚强度情景也会输出，但在缺少独立标签时不充当主结果。主要结果包括 `sample_scores.csv`、`domain_summary.csv`、`conflict_pairs_*.csv`、`score_sensitivity_by_domain.csv` 和 `run_metadata.json`。快速检查可追加 `--bootstrap 10 --permutations 20`。

## 两套质量维度方案

```powershell
python Code/scripts/compare_dimensions.py `
  --data-root real_attachments `
  --quality-config Code/configs/q1_quality.json `
  --dimension-config Code/configs/q1_dimension_schemes.json `
  --out Code/outputs/q1_dimension_schemes
```

这个入口并行输出 22 字段基线和语义分组候选方案，以及权重展开、结构诊断和盲审模板。方案含义、独立标注的输入格式与选择门槛见 [维度方案运行说明](docs/dimension_schemes.md)。在获得独立标签或受控训练反馈前，结构统计不能裁决哪套评分更接近真实质量。

## 任务 3：配比与验证 Loss

```powershell
python Code/scripts/run_mixture.py `
  --data-root real_attachments `
  --config Code/configs/q1_mixture.json `
  --out Code/outputs/q1_task3
```

A4+A5 用于分组交叉验证与模型选择；A6–A11 用于外部检验；A12–A15 只用于外推一致性诊断。质量桥接使用 A16 的参考映射、A17 的域摘要和全量流式读取的 A18 原文；A17/A18 只产生覆盖与文本完整性证据，不会被误当成 22 信号质量评分。`quality_bridge_domain.csv`、`quality_bridge_recipe.csv` 和 `regmix_text_evidence.csv` 给出映射覆盖、来源核对与缺失域的质量情景界；质量桥接不进入主 Loss 模型。其他主要结果包括 `model_cv_summary.csv`、`external_metrics.csv`、`replacement_effects.csv`、`combination_effect_summary.csv` 和 `mixture_model.json`；解读见 [任务 3 结果报告](../Q1/报告/任务3配比与验证Loss.md)。

## 辅助复算与测试

```powershell
python Code/analysis/dimension_audit.py
python Code/analysis/review_audit.py
python -m unittest discover -s Code/tests -p "test_*.py" -v
```

`review_audit.py` 会重写同目录的 `review_evidence.json`。两份审计脚本用于复核研究结论，正式结果由 `scripts/` 的三个入口生成。

对 B 方案零权重的复评在正式维度对照完成后运行 `python Code/analysis/reassess_semantic.py`，结果写入 `Code/outputs/q1_dimension_reassessment/`；方法与结论见 [B 方案复评](../Q1/报告/B方案零权重合理性复评.md)。

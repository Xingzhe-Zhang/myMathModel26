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

这个入口在同一 A1 冻结尺度下输出完整 CRITIC 加权 A22、QuRating 拆分且 DSIR 合并的 B23、QuRating 与 DSIR 双家族合并的 B20，并在 A1/A2/A3 上做排序、权重、归一化及收缩系数敏感性比较。各方案同时输出有限补偿候选分与共同原子信号冲突诊断。具体协议见 [维度实验说明](docs/ab_experiment.md)，结果解读见 [A/B 对照报告](../Q1/报告/任务12_A_B23_B20完整对照.md)。无独立标签或受控训练反馈时，不能裁决哪套更接近真实训练价值。

保留的 22 维等权逐条记录可与三方案直接对照：`python Code/analysis/equal_baseline_comparison.py`。脚本核对 UID 和原评分一致性后生成 `Code/outputs/q1_ab_experiment/equal_baseline_comparison.csv`，不改写原记录。

## 任务 3：配比与验证 Loss

```powershell
python Code/scripts/run_mixture.py `
  --data-root real_attachments `
  --config Code/configs/q1_mixture.json `
  --out Code/outputs/q1_task3
```

A4+A5 用于分组交叉验证与模型选择；A6–A11 用于外部检验；A12–A15 只用于外推一致性诊断。质量桥接使用 A16 的参考映射、A17 的域摘要和全量流式读取的 A18 原文；A17/A18 只产生覆盖与文本完整性证据，不会被误当成 22 信号质量评分。`quality_bridge_domain.csv`、`quality_bridge_recipe.csv` 和 `regmix_text_evidence.csv` 给出映射覆盖、来源核对与缺失域的质量情景界；质量桥接不进入主 Loss 模型。其他主要结果包括 `model_cv_summary.csv`、`external_metrics.csv`、`replacement_effects.csv`、`combination_effect_summary.csv` 和 `mixture_model.json`；解读见 [任务 3 结果报告](../Q1/报告/任务3配比与验证Loss.md)。

## 任务 3：质量信息是否改善配比预测

以下入口保留为**已完成的九项历史阶段**；它只使用通过 A1 复现校验的九个统计字段，不调用公开分类模型，其 Loss 数字不得当作十四项结果：

```powershell
& 'Code/.venv/Scripts/python.exe' Code/scripts/run_quality_mixture_9_stage.py
```

程序锁定 A16 六条老师给定的 direct/near_direct 映射，以 A1/A18 的九项共同信号域画像推断其余十一条，并输出 `completed_a16_mapping.csv`、第一/第二候选、相似度权重和删去单项信号的稳定性。A16 硬映射 Q 与 A18 相似度加权 Q 分别进入配方质量汇总和逐域质量耦合模型。另用任务 1/2 的冻结评分器对九项重建信号评分并做留一域检验；若该直接评分路线未通过门槛，不让它进入 Loss 对照。所有 Loss 模型仅在 A4+A5 内五折选参数，在 A6–A11 上检验；A12–A15 只作外推诊断。结果见 `Code/outputs/q1_quality_mixture_9_stage_complete_a16/` 和 [九项阶段报告](../Q1/报告/任务3九项质量引入三层实验.md)。

可用 `--phase quality` 或 `--phase loss` 分阶段运行。固定抽样量敏感性实验加 `--sample-per-domain 128` 并指定独立 `--out`。当前固定域 (Q) 只是配比模型的质量先验，(pQ) 仍是 (p) 的确定函数。

公开模型信号重建入口如下；**本机运行已完成**，A1 总体审计通过 9 项统计字段与 5 项公开模型字段。它只生成信号证据，不会自动更新九项 Loss 实验：

```powershell
& 'Code/.venv/Scripts/python.exe' Code/scripts/run_quality_signal_reconstruction.py
```

运行前先按任务 1、2 命令生成 `sample_scores.csv`、`domain_summary.csv` 和 `fitted_scoring_model.json`。程序从 A1/A18 文本重算全部 11 个 RedPajama 统计量，调用固定版本的五个同族公开分类器，并以 CoLA 语法可接受度模型作为第六项 `fluency_en` 的候选代理。原万卷流畅度权重未公开；该代理只有在 A1 对照通过后才会用于质量迁移。A1 逐信号核验只保留通过预设门槛的字段。

公开模型入口**只重建并核验信号，不运行 Loss**；详细步骤与审计产物见 [质量信号重建指南](docs/quality_signal_reconstruction.md)。`Code/outputs/q1_quality_reconstruction_experiment/accepted_fields.json` 列出 14 项；`fluency_en` 代理与两项行级统计未过审计。十四项完整质量桥接与 Loss 对照现由下一入口独立执行，不覆盖九项产物。既有 `run_quality_mixture_experiment.py` 是早期两条 Q 路线对照入口，不对应新版三层模型。

### 十四项完整实验

```powershell
& 'Code/.venv/Scripts/python.exe' Code/scripts/run_quality_mixture_14_stage.py --phase all
```

此入口先按冻结的 B20 原尺度基础分构建十四项画像桥接和直接评分，输出 A16 锚点、A1 留一域质量门禁；仅过关路线进入 A4+A5 五折选模及 A6–A11 实际 Loss 检验。A12–A15 只作估算外推诊断。结果在 `Code/outputs/q1_quality_mixture_14_stage/`，详见 [十四项完整实验报告](../Q1/报告/任务3十四项质量桥接与配比完整实验报告.md) 和 [实验前设计](../Q1/报告/任务3十四项质量桥接与配比实验设计.md)。`--phase quality`、`--phase loss` 可分阶段复算；程序不修改信号重建入口。

旧的配方筛选诊断脚本 `Code/analysis/q1_task3_quality_diagnostic.py` 及对应报告仍保留作题外附录，不是问题一任务 3 的模型判定依据。十四项正式结果以 13 个验证域 Loss 的预测检验与领域/组合分析为准。

旧五特征结果见 [历史对照实验报告](../Q1/报告/任务3质量引入对照实验.md)，不代表上述新实验。

## 辅助复算与测试

```powershell
python Code/analysis/review_audit.py
python -m unittest discover -s Code/tests -p "test_*.py" -v
```

`review_audit.py` 会重写同目录的 `review_evidence.json`。A/B 实验的权重、效用审计和敏感性表由 `run_ab_experiment.py` 一次生成。

# 任务 1、2：A22、B23、B20 完整维度实验

从仓库根目录运行：

```powershell
& 'Code/.venv/Scripts/python.exe' Code/scripts/run_ab_experiment.py
```

输出在 `Code/outputs/q1_ab_experiment/`。复算时可用 `--bootstrap 0` 快速检查流程，正式配置使用 100 次分域 bootstrap。随机种子、分组成员、CRITIC 混合系数、选择比例及归一化情景集中在 `Code/configs/q1_ab_experiment.json`。原始 22 字段评分与冲突审计仍可用 `Code/scripts/run_quality.py` 复算；任务 3 的重建程序依赖其冻结模型与结果，不应删除 `Code/outputs/q1/`。

## 数据和尺度

三方案共用 `schema.py` 与 `preprocess.py`。A1 51,230 条用于拟合缺失值、参考分位数、QuRating 子项效用、DSIR 参考映射和 CRITIC 权重。A2/A3 使用冻结变换；相同文本跨来源去重，输出保留来源归属。`sample_scores.csv` 对每条唯一文本给出全部方案、两套归一化下的基础分 `Q_*`、有限补偿候选 `Q_rule_*`、惩罚、共同的 25 原子信号冲突诊断。

二元模型 logits 经 softmax 取正类概率；ModernBERT 的 6 个 logits 经 softmax 求 0–5 等级期望并除以 5；QuRating 原列表的 4 项单独保留；百分比字段变成 0–1。`fineweb_edu`、流畅、洁净、可读、推理、专业、无广告、QuRating、DSIR 与唯一词比例按“越大越偏好”处理；top 2/3gram 占比按“越小越偏好”；词数、句数、熵和平均词长按 A1 的典型区间处理；非字母、全大写、数值字符比例按上限容忍处理。后两种是暂定偏好，不能当成质量真值。每个原子信号的具体变换、边界和 A1 饱和率见 `utility_dictionary.csv`、`atomic_utility_audit_A1.csv`。

`frozen_original` 保持既有 22 字段评分器的 A1 冻结效用：参考正/负指标按 A1 分域等质量权的 1%–99% 分位数作截断线性变换；区间和上限容忍按原配置。`dsir_reference_rank` 只替换 3 个 DSIR 效用：用 A1 分域等质量权的经验分位点构造单调参考百分位，A2/A3 用同一映射，超界截断到 0 或 1。其余 22 个原子信号完全不变。两套结果是尺度敏感性对照；百分位不是更正确的质量标定。

## 方案和权重

25 个原子信号对应源数据的 22 字段；QuRating 列表展开增加 3 项。

| 方案 | 概念数 | 合并方式 |
|---|---:|---|
| A22 | 22 | QuRating 四项按原配置等权形成一项；DSIR 三项分别保留 |
| B23_flat | 23 | QuRating 四项拆开；DSIR 三项等权合成一个目标相关性概念 |
| B20_hierarchical | 20 | QuRating 四项以 A1 内层 CRITIC 合为一个家族；DSIR 三项等权合为一个家族 |

三方案各自在 A1 的概念矩阵上估计分域均衡的 CRITIC 权重。对概念 \(j\)，信息量为 \(C_j=\sigma_j\sum_k(1-r_{jk})\)，归一化成 \(w_j\)。正式值使用 `critic_shrinkage=1`，即完整 CRITIC；`shrinkage_sensitivity.csv` 还报告 \(\xi\in\{0,.25,.5,.75,1\}\) 下 \((1-\xi)/d+\xi w_j\) 的排序变化。`weights.csv` 同时提供概念权重、展开后的 25 原子实际系数和分域 bootstrap 区间；`fitted_experiment.json` 保存精确映射与权重。B20 的 QuRating 内层均等权对照见 `qurater_inner_sensitivity.csv`。

CRITIC 只根据变异度和相关性分配**统计信息量**，不能推断训练收益或某维度的规范重要性。DSIR 三项的合并是以相关性核查支持的冗余处理候选，而三目标等权只是无偏目标先验；其“相似于目标域”含义也不同于内在文本质量。B23 保留 QuRating 四项的独立权重，因此可检验拆分后的家族预算是否明显膨胀；B20 把家族预算限制为一次外层 CRITIC 分配。没有独立标签时，不以任一方案为已验证最优。

## 任务 2 的可比诊断

三方案都在同一 25 原子效用上给出高低并存标签与强度；这避免维度数改变时冲突定义本身漂移。另把原任务 2 的四组候选语义冲突对映射到各方案概念上，用非负交互系数计算单调的有限补偿 `Q_rule_*`。B23 把 QuRating–无广告关系展开为四对。该惩罚依旧属于**候选偏好情景**，并非由无标签数据证明应扣分。`rule_scheme_comparison.csv` 对照 `Q_rule` 的排序；主方案比较使用不带惩罚的 `Q_*`。

`scheme_comparison.csv` 报同域 A22 与 B23/B20 的 Spearman 等级相关、前 20% 重合率和平均分差；`extension_stability.csv` 比较 A1 与 A2/A3 新文本的分布；`normalization_sensitivity.csv` 比较两套 DSIR 尺度；`qurater_inner_sensitivity.csv` 比较 B20 内层权重；`domain_summary.csv` 给分域摘要。所有“重合率”均在同一来源域内计算。A2/A3 的分布稳定性和 bootstrap 是稳健性证据，不能代替独立人工质量标签或受控训练的验证 Loss。

`group_redundancy.csv` 直接报告 A1、A2 新增和 A3 新增文本中的 DSIR/QuRating 组内两两原始 Spearman 相关。bootstrap 只重抽 A1 行，预处理参考边界在每次重抽中固定；归一化不确定性另由两尺度情景检验。

既有 22 维等权逐条实验 `Code/outputs/q1/sample_scores.csv` 保留在原位。运行 `python Code/analysis/equal_baseline_comparison.py`，会先核验两份输出的 UID、域、来源归属及旧 `Q_equal=Q_primary`、旧 `Q_critic=新 A22`，再生成 `Code/outputs/q1_ab_experiment/equal_baseline_comparison.csv` 与同名 JSON 元数据。表中以旧等权分为统一基线，分别比较三方案的基础分及有限补偿候选分；它不修改旧实验记录。

依据：[CRITIC 原始论文](https://doi.org/10.1016/0305-0548(94)00059-H)、[QuRating 论文与四子项](https://proceedings.mlr.press/v235/wettig24a.html)、[DSIR 论文](https://proceedings.neurips.cc/paper_files/paper/2023/hash/6b9aa8f418bde2840d5f4ab7a02f663b-Abstract-Conference.html)、[数据发布方字段说明](https://huggingface.co/datasets/opendatalab/SlimPajama-Meta-rater/blob/main/README.md)。

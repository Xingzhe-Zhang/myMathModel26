# Q1 两套质量维度方案与选择协议

## 两套可运行方案

`Q22_equal` 精确复现问题一任务 1、2 的 22 字段等权分数。四项 QuRating 先各占父字段的四分之一，整个 QuRating 家族占 `1/22`。

`Q_semantic` 先展开 25 个原子分数，按发布方给出的语义分组。表达组包括流畅度、可读性和 QuRating 写作风格；知识组包括 FineWeb 教育价值、推理、QuRating 事实知识和教育价值；洁净组包括格式洁净、无广告、非字母比例、大小写比例、句末标点与重复程度。2-gram、3-gram 先取平均，构成一个“重复”概念。其余信号仍进入报告：长度、句数、词熵、独特词比例、数字比例、平均词长形成结构诊断；专业性和所需专业知识形成复杂度诊断；三个 DSIR 保留为目标相关性向量。

默认候选分的十个非 QuRating 核心概念等权，并把 QuRating 父字段预算固定为 `1/22`：

```
Q_semantic = (21/22) × (十个非 QuRating 核心概念的均值)
             + (1/22) × (QuRating 写作风格、事实知识、教育价值的均值)
```

这使四项列表拆分不暗中提高 QuRating 的总权重。`quality_group_weights` 的 0.2、0.2、0.6 分别对应表达组 2 个、知识组 2 个、洁净组 6 个非 QuRating 概念；这些是可复现的中性起点，不是训练收益最优权重。`Q_semantic` 是面向一般自然语言的候选代理；GitHub 代码质量缺少可执行性、正确性等专门信号，跨领域均值不可被解释为代码普遍更差。

所有指标使用同一套在 A1 拟合并冻结的效用变换。分组、权重和选择门槛集中在 `Code/configs/q1_dimension_schemes.json`。`dimension_weights.csv` 展开两个分数对 25 个原子信号的实际系数。

## 运行

从项目根目录执行：

```powershell
python Code/scripts/compare_dimensions.py `
  --data-root real_attachments `
  --quality-config Code/configs/q1_quality.json `
  --dimension-config Code/configs/q1_dimension_schemes.json `
  --out Code/outputs/q1_dimension_schemes
```

输出包括全部去重记录的两套评分、六个概念组分数、QuRating 四子项、域级汇总、A1 内样本选择重合率、非重叠 A2/A3 扩展集分布漂移、原子信号相关矩阵与 7 域均衡抽取的盲审模板。模板每域随机抽取 60 篇 A1 文本，隐藏两个模型的评分。`blind_review_template.csv` 保留前 8,000 字的预览和截断标记；标记为真的记录须凭 `uid` 阅读 `blind_review_fulltext.jsonl` 中的完整原文后再标注。

## 如何选择

第一层只作结构核验：A1 的领域和长度调整后相关、A2/A3 非重叠部分的相关复现、满分饱和率、分组权重守恒、域内前 20% 入选差异、域间分布漂移。它们能发现冗余、规则失真和迁移问题，**不能证明哪个 Q 更接近真实质量**。A2/A3 没有独立质量标签，且只有 arxiv/github 扩展；不能用“均值更稳定”选赢家。

第二层使用独立盲审。两位评阅者分别看原文，按预先写定的细则给 `quality∈[0,1]` 和 `defect∈{0,1}`；记录阅读不完整、代码等特殊类型。报告二人缺陷一致率、Cohen κ、质量分平均绝对差；分歧经第三方或讨论裁决后形成一行一个 `uid,quality,defect` 的终评 CSV。两套模型和分组参数在终评前冻结。盲审样本按领域随机抽样，比较每域各自排前 20% 的文章，最后对 7 域等权平均，避免 GitHub 的扩展集数量支配结果。可运行：

```powershell
python Code/scripts/compare_dimensions.py `
  --labels Code/labels/q1_adjudicated.csv `
  --review-per-domain 0
```

标签文件还可选带 `quality_r1,quality_r2,defect_r1,defect_r2`，程序报告评阅者一致性。程序使用按领域配对重抽样给两模型入选文本的人工质量均值差、缺陷率差提供 95% 区间。默认门槛：至少 200 条标签且每域至少 20 条；只有候选方案的质量改进区间下限大于 0、缺陷率增幅区间上限小于 2 个百分点、任何领域的质量均值下降不超过 0.05，才报告候选方案获得此盲审集支持。2 个百分点和 0.05 是预先写定的工程容忍度，可在查看标签前修改，并非论文给出的通用常数。无法满足门槛时输出“不确定”，继续使用 22 字段基线作为保守主结果。

第三层若算力允许，采用固定模型规模、相同来源池、相同训练 Token 预算、相同训练配置和随机种子，只改变按两套 Q 选出的训练样本；保持每域抽样配额一致，按相同验证集比较 Loss 和下游任务，至少重复多个种子。这才可选择面向 LLM 训练效用的方案。若独立盲审与训练反馈结论不同，应分别报告“人工文本质量”与“训练效用”，不强行合并为同一尺度。

## 为什么这样设计

- [官方数据卡](https://huggingface.co/datasets/opendatalab/SlimPajama-Meta-rater/blob/main/README.md)明确 QuRating 四项顺序、PRRC 四项语义及 DSIR 的目标域含义。
- [QuRating，ICML 2024](https://proceedings.mlr.press/v235/wettig24a.html)分别研究写作风格、所需专业知识、事实知识和教育价值，并强调质量与多样性平衡；四项不宜无说明地等同。
- [DSIR，NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/6b9aa8f418bde2840d5f4ab7a02f663b-Abstract-Conference.html)把相似性作为面向目标分布的重要性权重，支持将 DSIR 与普适质量区分。
- [Meta-rater，ACL 2025](https://aclanthology.org/2025.acl-long.533/)通过代理模型训练反馈学习多维质量权重；当前附件没有相同的反馈，故只能把语义分组作为待验证候选。
- [DataComp-LM，NeurIPS 2024](https://proceedings.nips.cc/paper_files/paper/2024/hash/19e4ea30dded58259665db375885e412-Abstract-Datasets_and_Benchmarks_Track.html)使用固定训练与评价条件比较数据整理策略，支持上述固定算力对照设计。

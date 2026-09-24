# Q_domain 映射模块设计（方案，尚未执行）

本模块只把 A1 的 7 个质量域 `arxiv, book, c4, commoncrawl, github, stackexchange, wikipedia` 的任务 1/2 质量评分 Q，借助 A1/A18 共同的、**已通过 A1 复现校验**的信号，迁移到 17 个 RegMix 域。A16 是领域语义对应关系及检验依据，不含 RegMix 域的真实 Q；模块不读取 A4–A15，也不训练或评估 Loss 模型。当前只固定接口与判断规则，等质量重建完成后再运行。

## 输入契约

| 来源 | 格式和必要字段 | 用途 |
| --- | --- | --- |
| A16 `real_attachments/A_data_value/domain_mapping_guide.csv` | 17 行，`mixture_domain,quality_domain,mapping_type,note`；`mapping_type` 只接受 `direct/near_direct/inferred` | 检查域全集、校验 6 个已知语义映射，保留 11 个待推断域 |
| 任务 1/2 `Code/outputs/q1/domain_summary.csv` | A1 七域的 `dataset,domain,Q_primary_mean,n_unique`，以及可用的不确定性列 | 7 个源域的 Q 及样本量；必须与冻结的任务 1/2 评分口径一致 |
| 待完成的质量重建 `accepted_fields.json`、`signal_a1_audit.csv`、`signal_a1_audit_by_domain.csv` | 通过的共同信号名、A1 总体和分域复现误差 | 白名单与审计证据；不能把“可计算”直接当成“可迁移” |
| 待完成的质量重建 `reconstructed_signals.npz` | `a1_x[n1,k]`, `a18_x[n18,k]`, `a1_domain[n1]`, `a18_domain[n18]`；列顺序与 `accepted_fields.json` 一致 | 同口径文本级特征；聚合为 7+17 个域画像 |
| 待完成的质量重建 `sample_coverage.csv` | `dataset,domain,target,available,selected,shortfall,sampling` | 检查代表性和样本量；A18 少于 64 条的域保留全部文本并标记低覆盖 |

读取前强制检查：A16 恰好覆盖 17 个唯一 RegMix 域；A1 Q 覆盖 7 个唯一质量域；矩阵行数、域标签、特征列数一致；所有输入特征有限；任一域零文本即停止。模型版本、截断长度与信号转换方式必须由重建元数据固定。**本设计没有读取这些尚未完成的重建产物。**

## 经校验的共同信号

候选池为 11 个 `rps_*` 统计字段，以及 6 个公开模型/代理字段：`fineweb_edu`、4 个 `modernbert_*`（cleanliness、readability、reasoning、professionalism）、`fluency_en`。实际入模集合只由 A1 官方值与重算值的误差审计及 A18 有效值覆盖决定；未通过的字段不得使用。`fluency_en` 的当前 CoLA 模型只是流畅度代理，不能因为可运行就视为复现成功。除总体阈值外，需检查每个 A1 域的误差，避免总体相关由域间均值差异造成。对高度相关的字段做分组/降权敏感性分析，避免某一类统计特征重复计票。文档不预设最终通过字段数。

## 映射逻辑

1. 用同一组已校验信号，分别计算 A1 七域和 A18 十七域的画像：每域样本量、各特征均值、标准差及 10/50/90 分位数。特征尺度只由 A1 样本拟合，再作用于 A18；不使用 Q 或 Loss 拟合尺度。
2. 计算每个 RegMix 域到 7 个质量域的画像距离，转为非负且和为 1 的相似度权重 `w_{r,s}`。主结果为 `Q_domain(r)=Σ_s w_{r,s} Q_A1(s)`，保留全部 7 个权重，而非中性填补或强行单域匹配。距离和温度应预先固定；画像中的均值、离散度可沿用现有 `quality_transfer.py` 的实现，分位数作为稳健性备选。
3. **先校验，后解释。** 暂不把 A16 强制写入相似度：检查 3 个 `direct` 与 3 个 `near_direct` 域的已知来源在相似度排序中的位置；报告 top-1/top-3、已知域 Q 与相似度 Q 的差异。A16 的近似对应仅是语义参考，并非 A18 真实质量标签，因此 Q 差异是代理检验，不是外部真值误差。若该校验不通过，不能把 11 个 `inferred` 域的 Q 当作可靠定量结果。通过后再输出 17 域估计，同时并列保留 A16 原始映射和相似度结果，供检查冲突。
4. 对每个域做特征逐一剔除、温度变化及域内文本 bootstrap，输出 Q 范围和权重稳定性。`ubuntu_irc` 这类短样本域使用全部实际文本，不复制凑足 64 条；短样本及相似度接近并列的域标为低置信。较宽区间应传递给后续分析，不能只留一个点值。

## 输出契约（设计目标，不表示已产生）

| 文件 | 最少字段 |
| --- | --- |
| `q_domain_mapping.csv` | 17 行：`mixture_domain,a16_mapping_type,a16_quality_domain,a18_n,q_domain,top1_quality_domain,top1_weight,top2_quality_domain,top2_weight,weight_entropy,q_sensitivity_min,q_sensitivity_max,confidence_flag`；另附 7 个 `weight_<quality_domain>` 列 |
| `q_domain_anchor_validation.csv` | 6 行：A16 已知映射、相似度 top-1/top-3、真实语义域的排序、Q 代理差异及是否达预设门槛 |
| `q_domain_sensitivity.csv` | 每域在温度、逐特征剔除和 bootstrap 场景下的 `q_domain`、top-1 来源与权重 |
| `q_domain_metadata.json` | 输入文件/校验版本、特征白名单、标准化规则、距离/温度、样本覆盖、门槛与是否准许下游使用 |

这些文件只属于质量层。即使映射检验通过，固定的域级 Q 与配比加权项 `p_i Q_i` 可能与 `p_i` 线性共线；后续 Layer 2/3 的 Loss 设计需另行解决可识别性问题，不能把映射通过直接解释为预测增益。本阶段不启动任何映射计算或 Loss 实验。

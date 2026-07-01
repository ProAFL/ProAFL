# box_features 特征重要性分析方案（优化版）

> 目标：科学、可复现地量化 `box_features.py` 中 8 个训练动态特征对 **error / correct** GT box 的区分重要性。

---

## 一、当前方案的问题诊断

现有 `ours/discussion/features/box_features.py` 的做法：对 8 个特征**逐个**做 Mann-Whitney U 检验 + Cohen's r，配合箱线图/KDE 可视化。

不足之处：

1. **单变量分析无法回答"重要性"**：8 个特征间高度共线（`conf_mean` 与 `early/lastly_conf_mean` 几乎一定相关），单看 r 大无法反映**边际信息贡献**。
2. **样本污染**：未匹配上的 GT box 用"最可疑端值"填充。VisDrone 上未匹配占 ~45%，会人为拉大两组分布差异，使所有 r 虚高。
3. **多重比较未校正**：8 次检验都报 p ≈ 0；样本量上万时 p 值几乎无信息。
4. **不确定性未量化**：r 没有置信区间，跨数据集差异无法判断是否在抽样波动内。
5. **无跨数据集稳定性度量**：三个数据集 r 排名有差异，未量化稳定性。
6. **超参敏感性缺失**：`K=0.2`、`D_*` 阈值（`0.5*lastly_mean`）写死，未做扫描。
7. **`min_e_*` 初值 bug**（前面已确认）：永不超阈值时 D=0 而不是 1，与"最可疑"语义颠倒，会污染 D_* 分布。

---

## 二、优化后的分析方案

### Step 0：先修 bug 再分析

- 修复 `min_e_conf` / `min_e_iou` 初值为 `epochs`（永不超阈值时 D=1）。
- 评估"未匹配 GT box 用极端值填充"的影响：建议**默认排除**这部分样本，作为主分析；将"含填充"作为敏感性对比。

### Step 1：数据准备与基础描述

对每个数据集：
- 报告 `correct / error` 样本量、未匹配比例。
- 8 个特征的描述统计：mean / median / IQR / 缺失率，按 error / correct 分组列出。
- 输出 `feature_describe.csv` 与 `feature_describe.png`（小提琴图矩阵）。

### Step 2：单变量重要性（多指标交叉验证）

对每个特征 × 每个数据集：

| 指标 | 含义 | 备注 |
|---|---|---|
| **AUC (ROC)** | error 当正类，特征值（按 sign 调向）当 score | **首选**：语义即"做 error 检测器的能力"，跨特征/数据集可比 |
| **Cliff's δ** | 非参数效应量，比 r 更直观 | δ ≈ 2·AUC − 1 |
| **Cohen's r** | 现有指标 | 保留以便对比 |
| **KS 统计量** | 分布差异最大值 | 反映分布形态差异 |
| **互信息 MI** | 非线性依赖 | 用于检验非单调关系 |

每个指标都附 **Bootstrap 95% CI（n=1000）**。

### Step 3：相关性结构

- 8×8 **Spearman ρ** 热图：识别冗余特征簇。
- 期望可见：`conf_mean ↔ early/lastly_conf_mean` 强相关、iou 系列内部强相关、conf↔iou 中等相关。

### Step 4：多变量 / 边际重要性

解决冗余问题，回答"加进来还有没有用"：

1. **L1 / Elastic-Net Logistic 回归**：8 个特征标准化后同时入模，看保留下来的特征及其系数。
2. **Random Forest / Gradient Boosting + Permutation Importance**：报告打乱该特征后 **CV-AUC 下降量** 作为边际重要性。比内置 split importance 更可靠。
3. **SHAP 值**：每个样本每个特征的边际贡献，可看正/负方向。
4. **Greedy forward selection on CV-AUC**：每轮加入使 AUC 提升最大的特征，绘制饱和曲线，回答"用几个就够"。

### Step 5：稳定性 / 鲁棒性

1. **跨数据集排名一致性**：三数据集间特征重要性排名的 **Kendall τ** / **Spearman**。
2. **超参敏感性**：
   - `K ∈ {0.1, 0.15, 0.2, 0.25, 0.3}`
   - `D_*` 阈值系数 ∈ {0.3, 0.5, 0.7}
   看排名是否稳定。
3. **Bootstrap-on-samples**：每个数据集 500 次有放回采样，输出每个特征 AUC / r 的分布宽度，量化估计稳定性。

### Step 6：可视化升级

- 保留：箱线图 + KDE。
- 新增：
  - **ROC 曲线总图**：8 条曲线一图，一眼分高下。
  - **特征相关性热图**。
  - **AUC 森林图（带 95% CI）**：横向对比 8 个特征在 3 个数据集上的 AUC。
  - **Permutation importance 条形图**。
  - **Forward selection 饱和曲线**。

### Step 7：综合排序与结论

最终输出一张主表：

| 特征 | AUC (95% CI) | Cliff's δ | Cohen's r | Perm. Importance | LR 标准化系数 | 排名稳定性 (Kendall τ) |
|---|---|---|---|---|---|---|

并据此给出：
- **首选特征**（兼顾单变量强 + 边际贡献大 + 稳定）
- **冗余特征**（与首选高相关且边际贡献低）
- **数据集差异**与可能的原因（如 VisDrone 小目标场景下区分能力下降）

---

## 三、最低限度的补丁（如果不想全做）

如果只能加两步，优先：

1. **AUC + bootstrap 95% CI** 替代/补充 Cohen's r —— 解决"单变量但不可比"和"无不确定性"两个问题。
2. **排除未匹配填充样本，单独跑一遍** —— 解决"样本污染"问题，对照看结论是否稳健。

---

## 四、推荐输出物结构

```
ours/discussion/features/results/
├── box_features_<dataset>_<model>/
│   ├── run.log
│   ├── summary.md
│   ├── descriptive_stats.csv
│   ├── univariate_metrics.csv         # 每特征 AUC/Cliff's δ/r/KS/MI + CI
│   ├── correlation_heatmap.png
│   ├── roc_curves.png
│   ├── auc_forest_plot.png
│   ├── permutation_importance.png
│   ├── forward_selection_curve.png
│   └── *.png                           # 单特征 box+KDE
└── cross_dataset_summary.md            # 三数据集横向对比 + Kendall τ
```

---

## 五、实施顺序建议

1. 修 `min_e_*` 初值 bug。
2. 增加排除未匹配样本的开关。
3. 用 AUC + bootstrap CI 替换/扩展现有指标。
4. 加相关性热图 + ROC 总图。
5. 加 LogReg / RF + permutation importance。
6. 加超参敏感性扫描和跨数据集稳定性表。

每一步都能产出独立可读的中间结果，便于增量推进与回滚。

# Cross-Condition Generalization Notes

## 核心结论

GLDAS-Gaze 在 COCO-Search18 上呈现明显的非对称跨条件迁移：

- `TA -> TP` 的迁移明显强于 `TP -> TA`。
- TP 训练更偏向快速目标获取和提前停止，难以直接迁移到 TA 的系统性排查。
- TA 训练学习到更广域的上下文搜索先验，因此能部分迁移到 TP。
- 跨条件迁移对 SS、SemSS、FED、SemFED 影响更明显，对 MM 的影响相对较小，说明几何轨迹形态比语义序列决策更容易迁移。

## 实验结果

| Train | Test | SS ↑ | SemSS ↑ | FED ↓ | SemFED ↓ | MM ↑ | Len. Err. ↓ | Avg. Len |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| TP | TP | 0.489 | 0.555 | 2.129 | 1.553 | 0.882 | 0.693 | 2.588 |
| TP | TA | 0.358 | 0.362 | 5.212 | 5.195 | 0.838 | 3.512 | 5.673 |
| TA | TA | 0.419 | 0.412 | 4.204 | 4.257 | 0.845 | 2.705 | 4.087 |
| TA | TP | 0.455 | 0.528 | 2.300 | 1.709 | 0.866 | 0.764 | 2.770 |

## 退化幅度

### TP -> TA

以 `TA -> TA` 作为 TA in-domain 参考：

- SS: `0.419 -> 0.358`，下降 `0.061`，约 `14.6%`。
- SemSS: `0.412 -> 0.362`，下降 `0.050`，约 `12.1%`。
- FED: `4.204 -> 5.212`，增加 `1.007`，约 `24.0%`。
- SemFED: `4.257 -> 5.195`，增加 `0.939`，约 `22.1%`。
- Avg. Len: `4.087 -> 5.673`，路径更长但序列质量更差。

解释：TP 模型遇到目标缺失时会继续搜索，但缺少 TA 中需要的系统性候选区域排查策略。

### TA -> TP

以 `TP -> TP` 作为 TP in-domain 参考：

- SS: `0.489 -> 0.455`，下降 `0.034`，约 `6.9%`。
- SemSS: `0.555 -> 0.528`，下降 `0.027`，约 `4.9%`。
- FED: `2.129 -> 2.300`，增加 `0.171`，约 `8.0%`。
- SemFED: `1.553 -> 1.709`，增加 `0.156`，约 `10.0%`。
- Avg. Len: `2.588 -> 2.770`，路径略长，说明 TA 模型在 TP 上更保守。

解释：TA 训练迫使模型学习更广域的场景上下文和候选区域搜索先验，这些先验能部分迁移到 TP。

## 论文表述建议

不要写成：

```text
GLDAS-Gaze shows strong cross-condition generalization.
```

建议写成：

```text
GLDAS-Gaze reveals asymmetric cross-condition transfer between TP and TA.
```

或者：

```text
The learned search priors are partially transferable, with stronger transfer from TA to TP than from TP to TA.
```

## 英文草稿

```text
To further examine the transferability of the learned search strategy,
we conduct a cross-condition generalization experiment on COCO-Search18.
Specifically, we evaluate the TP-trained model directly on TA images and
the TA-trained model directly on TP images without any fine-tuning.

The results reveal an asymmetric transfer pattern. When the TP-trained
model is evaluated on TA, performance drops substantially, with SS
decreasing from 0.419 to 0.358 compared with the TA-trained in-domain
model, and FED increasing from 4.204 to 5.212. This suggests that a
model trained for target-present search mainly learns rapid target
acquisition and stopping behavior, but lacks the systematic exploration
strategy required to verify target absence.

In contrast, the TA-trained model transfers more favorably to TP.
Compared with the TP-trained in-domain model, SS decreases from 0.489
to 0.455 and FED increases from 2.129 to 2.300. This smaller degradation
indicates that target-absent training encourages the model to learn
broader contextual search priors, which remain partially useful when
the target is present. However, the slightly longer average scanpath
length suggests that the TA-trained model is more conservative and less
efficient in terminating after target localization.

Overall, these results demonstrate that TP and TA correspond to
asymmetric search regimes. While geometric scanpath similarity remains
relatively stable across conditions, semantic and sequential metrics are
more sensitive to the mismatch between training and testing conditions.
This further supports the necessity of explicitly evaluating both
target-present and target-absent scenarios in task-driven scanpath
prediction.
```

## 下一步建议

优先做停止机制单独评估，因为已有预测文件可以直接复用，实验成本最低，且能补强论文中 termination head 的贡献：

1. 统计 Human 与 GLDAS 的 scanpath length distribution。
2. 报告 length MAE / length error mean / avg length。
3. 分 TP、TA 分别画长度分布柱状图。
4. 对 TP 进一步统计是否在 target hit 后及时停止。
5. 对 TA 分析过早停止与过度搜索趋势。

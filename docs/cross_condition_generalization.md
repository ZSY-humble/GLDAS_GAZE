# 跨条件评估操作说明

本说明用于复现 TP / TA 模型在同条件及跨条件数据上的评估。
执行前请按主 README 安装依赖，准备模型权重和处理后的 COCO-Search18 数据。
所有命令从仓库根目录运行。

## 参数对应关系

- `--model-hparams`：训练时使用的配置，决定待加载的模型结构。
- `--ckpt`：与模型配置一致的检查点。
- `--eval-hparams`：待评估条件的配置，决定数据协议和默认解码步数。
- `--tap`：与评估配置一致的 `TP` 或 `TA`。
- `--dataset-root`：处理后的数据集根目录。
- `--disable-saliency`：跳过 cIG、cNSS、cAUC 计算。

## 运行四种条件组合

以下脚本依次运行 `TP -> TP`、`TP -> TA`、`TA -> TP` 和 `TA -> TA`，
并分别保存结果。将数据路径替换为实际目录。

```bash
dataset_root=/path/to/dataset_root
for model_condition in TP TA; do
  for eval_condition in TP TA; do
    python tools/cross_condition_eval.py \
      --model-hparams \
        "configs/coco_search18_dense_SSL_${model_condition}.json" \
      --eval-hparams \
        "configs/coco_search18_dense_SSL_${eval_condition}.json" \
      --dataset-root "$dataset_root" \
      --ckpt "pretrained_models/${model_condition}.pt" \
      --tap "$eval_condition" \
      --gpu-id 0 \
      --eval-mode greedy \
      --disable-saliency \
      --log-dir \
        "assets/cross_condition_eval/${model_condition}_to_${eval_condition}"
  done
done
```

脚本直接加载检查点进行评估，不执行微调。
`--eval-mode sample` 可改用采样解码；需要逐步叠加图时增加
`--save-overlays`，并用 `--overlay-dir` 指定输出目录。

## 解码步数与时间步嵌入

TP 配置的默认最大解码步数为 6，TA 为 10。
当评估步数超过模型训练步数时，脚本默认重复最后一个已训练时间步的嵌入，
对应参数 `--extend-time-embedding repeat_last`。
也可使用 `--extend-time-embedding zero` 进行零填充。

如需将跨条件评估限制在 6 步，在命令后增加：

```bash
--max-eval-steps 6
```

比较结果时，应保持数据划分、解码方式、最大步数与嵌入扩展策略一致，
并记录实际使用的参数。

## 输出与停止行为统计

每种组合的指标与 `predictions_*.json` 保存到对应的 `--log-dir`。
未指定该参数时，默认写入 `assets/cross_condition_eval/` 下，
子目录由模型配置名、评估配置名和评估条件组成。

完成评估后运行：

```bash
python tools/analyze_stopping_behavior.py \
  --dataset-root /path/to/dataset_root \
  --pred-root assets/cross_condition_eval \
  --output-dir assets/stopping_analysis
```

输出文件：

- `stopping_summary.csv`：各组预测的长度及命中统计。
- `stopping_metrics.json`：结构化统计结果。
- `length_distribution_*.png`：人类与模型轨迹长度分布。

统计包括平均轨迹长度、长度绝对误差、长度偏差、过度搜索比例、
搜索不足比例和 TP 条件的目标命中情况。

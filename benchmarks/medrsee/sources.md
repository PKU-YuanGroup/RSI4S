# 测试评分：入口、指标与代码来源

当前接入 15 项 MedRSEE 任务。下表记录其中原 14 项的指标来源；MED_02 的实现另见 [brats_metrics.py](brats_metrics.py)。任务成员以 [tasks_medrsee.txt](../../configs/tasks_medrsee.txt) 为准。

## 从哪里看评分代码

组织方的终轮评分入口是 [scoring.py](scoring.py) 中的 `final_submission`，输出格式检查见 [submission.py](submission.py)。指标分派和聚合位于 [registry.py](registry.py)，官方／作者源码保存在 [evaluation/official/](evaluation/official/)。

下表说明指标来源、方向和聚合口径。科研方法由 Agent 在实验目录中实现，原生评分与样本成员由宿主维护。

## 6 项：数据集官方／作者代码

“复用官方代码”在这里指实际调用下列指标函数。样本选择、任务输出和总分汇总按本项目的固定划分协议执行。

| 任务 | 主指标 | 实际代码位置与函数 | 计算方式和适配 |
|---|---|---|---|
| RS_02 MADOS | Macro F1 ↑ | evaluation/official/mados/utils/metrics.py → Evaluation | 原标签1–15映射为0–14，原标签0作为忽略区域；汇总有效像素，沿用作者对真值／预测类别并集求 macro F1。 |
| RS_01 Landslide4Sense | 滑坡 F1 ↑ | evaluation/official/landslide/utils/tools.py → eval_image | 逐图计算统计量，累计全部测试像素的 TP／FP／FN，再算滑坡类 F1。 |
| MED_05 CoNIC | mPQ+ ↑ | evaluation/official/conic/metrics/stats_utils.py → get_multi_pq_info | 六类核分别累计全体图像的 TP／FP／FN／IoU，再按作者 compute_stats.py 求六类平均；保留原有1e-6项。 |
| MED_01 ACDC | 三结构 Dice ↑ | evaluation/official/acdc.py → metrics | 调用原函数，取 LV／RV／MYO 三个 Dice；ED／ES 在病例内等权，再病例等权。输出为结构 Dice 和病例明细。 |
| MED_07 VerSe | 椎体身份 Dice ↑ | evaluation/official/verse_compat.py → compute_dice | 复用作者 Dice，兼容副本将 np.bool 改为 np.bool_。本项目取每扫描真值／预测中出现的非零身份并集，身份等权后扫描等权；这是本项目的身份分割总分。 |
| MED_04 LUNA16 | CPM ↑ | evaluation/official/luna/noduleCADEvaluationLUNA16.py → evaluateCAD | 沿用结节半径匹配、排除注释处理及每扫描100候选上限；取0.125、0.25、0.5、1、2、4、8 FP/scan的敏感度平均。交付 CPM 点估计及7点敏感度。 |

上游来源与版本：

- MADOS：[作者仓库](https://github.com/gkakogeorgiou/mados)，commit c20a7e972bdf111126540d8e6caa45808352f138。
- Landslide4Sense：[官方仓库](https://github.com/iarai/Landslide4Sense-2022)，commit 2df4c378cfc4a0feedf197826e769da520100d81。
- CoNIC：[官方仓库](https://github.com/TissueImageAnalytics/CoNIC)，commit c1c4964d2aefcb9ec0ba08e88ed563fd80d41ac4。
- ACDC：[官方 metrics_acdc.py](https://www.creatis.insa-lyon.fr/Challenge/acdc/code/metrics_acdc.py)，2026-10-07取回，指标函数保持原样。
- VerSe：[作者仓库](https://github.com/anjany/verse)，commit 02b292b86021a8873043124982f5f58da9ba1cb8。
- LUNA16：官方 evaluationScript.zip；原始 Python 2 文件保留于发布 ZIP，当前可执行副本完成 Python 3、CSV文本模式和 matplotlib 参数适配，并将零检出敏感度设为0；bootstrap设置为False。

## 8 项：通用库指标＋项目聚合

以下评分由成熟库计算基础指标，本项目实现任务单位、有效目标选择及分组平均。它们是本项目固定实例的评估口径。

| 任务 | 主指标 | 实际调用 | 计算方式 |
|---|---|---|---|
| ENV_01 GLORIA | log10 Chla RMSE ↓ | sklearn.metrics.root_mean_squared_error | 真值与预测均在 log10 叶绿素浓度空间计算 RMSE。 |
| ENV_02 CAMELS-GB | 流域平均 NSE ↑ | sklearn.metrics.r2_score(force_finite=False) | 每流域按 NSE = 1−SSE/SST 计算，再流域等权；保留负值。常量真值触发明确错误。 |
| ENV_03 GEFCom2014-W | Mean pinball loss ↓ | sklearn.metrics.mean_pinball_loss | 24小时、99分位数（0.01…0.99）；每分位对有效小时求损失，再99分位等权。报告有效／缺测小时数。 |
| ENV_04 ECOPROPHET | 站点平均 MAE ↓ | sklearn.metrics.mean_absolute_error | 单位 gC m^-2 day^-1；站点内算 MAE，再站点等权。 |
| ENV_05 电池弛豫容量 | 电芯平均 MAE ↓ | sklearn.metrics.mean_absolute_error | 单位 Ah；电芯内算 MAE，再电芯等权；另给出按化学体系的 RMSE。 |
| MED_03 MSD Liver | 结构 Dice ↑ | monai.metrics.compute_dice | 肝脏区域合并标签1／2，肿瘤取标签2；两结构等权后病例等权，并报告各结构明细。 |
| MED_08 BUS-BRA | 患者平均 Dice ↑ | monai.metrics.compute_dice | 先计算每张图病灶 Dice，患者内图像平均，再患者等权。 |
| MED_06 CrossMoDA ceT1 | 结构 Dice ↑ | monai.metrics.compute_dice | 当前为增强T1监督实例；肿瘤／耳蜗分别计算，结构等权后病例等权。 |

MONAI 的二值 Dice 输入为逐结构掩膜，设置 ignore_empty=False：真值和预测双空得1，单空得0。来源：[MONAI 1.3.2](https://monai.readthedocs.io/en/1.3.2/metrics.html)。sklearn 基础函数见 [RMSE](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.root_mean_squared_error.html)、[pinball loss](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_pinball_loss.html)。实际依赖版本记录在 environment.json。

BUS-BRA 作者仓库也保存在 evaluation/official/busbra，来源为 [wgomezf/BUS-BRA](https://github.com/wgomezf/BUS-BRA)，commit 296f9ae570b33fbb608b10b0cdd133f8addfab10；本轮实际调用的是上表的 MONAI Dice。

## 当前适配与实验记录

样本成员由 [membership.py](membership.py) 与 `splits.csv` 管理，输入读取与任务接口见 [source.py](source.py)、[runtime.py](runtime.py)，提交校验与终轮评分分别位于 `submission.py`、`scoring.py`。

运行结果及论文表格统一保存在 RSI4S 的 `runs/<实验ID>/`，见[实验记录规范](../../docs/experiment_records.md)。原生指标的来源与许可证继续随评分源码保留。

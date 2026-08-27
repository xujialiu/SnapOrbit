# SnapOrbit

论文 *"Uncertainty-Aware Deep Learning Framework for Orbital Disease Triage
Using Smartphone-Captured Images: A Prospective Study"* 的代码。

[English](README.md) | [中文](README.zh.md)

一个两阶段 **DINOv3 + LoRA** 流程,从智能手机外眼照片中区分 **TED / SOLs / 正常**:

1. **关键点定位** —— 检测 4 个眦点,据此裁剪**双眼**(448×896)或**单眼**(448×448)ROI。
2. **三分类器** —— DINOv3-Huge + LoRA,并用 **snapshot-ensemble 不确定性**对最不确定的样本进行转诊(deferral)。

> 仓库不含任何患者数据或训练权重。`projects/` 被 git 忽略,用于存放运行时输出与你的本地数据。

## 环境

```bash
conda create -n ted python=3.11 -y && conda activate ted
# 按你的 CUDA 安装 PyTorch(https://pytorch.org),然后:
pip install -r requirements.txt
```

DINOv3 权重首次使用时由 `timm`(≥ 1.0.19)自动下载。

## 数据

准备下列 CSV(不随仓库分发),需包含患者级 `fold` 列(0–9)用于交叉验证,并在配置中
设置 `dataset.image_path` / `csv_path`。

- **分类** —— 长表,每图 4 行眦点:`relative_path, keypoint_x, keypoint_y, label,
  fold`(`label`:0=正常,1=TED,2=SOLs;关键点为百分比 0–100)。
- **关键点** —— 宽表,每图一行,坐标归一化到 0–1:`relative_path, x1,y1, x2,y2,
  x3,y3, x4,y4, fold`。用 `python data_ted/codes/make_keypoints_csv.py` 生成。

## 使用

在仓库根目录运行。配置位于 `configs/`;任意字段都可在命令行覆盖(如 `device=cuda:0`)。

```bash
# 1) 关键点模型
python train_keypoints.py -c configs/ted_keypoints/dinov3_huge_448x448_lora_r8a16_keypoints_0.yaml

# 2) 分类器 —— 双眼(主)/ 单眼(消融)
python train_classification.py -c configs/ted_classification/dinov3_huge_448x896_lora_r8a16_d01_dp01_f1_aug_v3_double_weighted-sampler_0.yaml
#    评估:追加  --eval paths.ckpt_path=<checkpoint.pth>

# 3) 不确定性(snapshot ensemble = 各折验证 F1 最高的 6 个 ckpt × 5 折;预测熵)
python uncertainty_classification.py -c configs/ted_classification/<run>/config_eval_snap.yaml

# 4) 可解释性(ViT 注意力图)
python explainability_classification.py -c configs/ted_classification/<run>/config_exp.yaml
```

5 折脚本:`configs/ted_keypoints/script_train_5fold.sh`、
`configs/ted_classification/script_train_{double,single}.sh`,以及
`script_{uncertainty,explainability,reader_study}.sh`。

## 目录

```
train_{classification,keypoints}.py   入口(及 engine_*、uncertainty_*、explainability_*)
models/ datasets/ augmentations/ loss/ util/   框架代码
configs/{ted_classification,ted_keypoints}/    配置 + 运行脚本
data_ted/codes/   数据准备与绘图      visualization/   关键点叠加可视化
```

## 关键设置

DINOv3-Huge+ 骨干,LoRA 注入注意力 QKV 投影(双眼 r8/α16,单眼 r32/α64,dropout
0.1);输入 448×896 / 448×448;交叉熵 + 类别加权采样器;base LR 1e-3、weight decay
0.05、100 epochs(cosine + 10 轮 warmup)、混合精度。各实验的具体超参见 `configs/`。

## 许可证

基于 [MIT License](LICENSE) 发布。

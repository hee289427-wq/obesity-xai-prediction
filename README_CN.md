<div align="center">

[English](README.md) | [简体中文](README_CN.md)

# 基于可解释人工智能的肥胖程度预测系统

### 从模型预测延伸到决策解释与 What-if 情景分析

一个融合 **随机森林、SHAP 可解释性分析和反事实情景分析** 的交互式机器学习应用，用于预测肥胖程度并解释单个样本的模型决策。

[![在线体验](https://img.shields.io/badge/在线体验-Render-46E3B7?style=for-the-badge&logo=render&logoColor=white)](https://obesity-xai-prediction.onrender.com)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Gradio](https://img.shields.io/badge/Gradio-交互式应用-F97316?style=for-the-badge)](https://www.gradio.app/)
[![SHAP](https://img.shields.io/badge/可解释人工智能-SHAP-7C3AED?style=for-the-badge)](https://shap.readthedocs.io/)
[![MIT License](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

[![GitHub stars](https://img.shields.io/github/stars/hee289427-wq/obesity-xai-prediction?style=social)](https://github.com/hee289427-wq/obesity-xai-prediction/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/hee289427-wq/obesity-xai-prediction?style=social)](https://github.com/hee289427-wq/obesity-xai-prediction/forks)

</div>

---

## 系统演示

<p align="center">
  <img
    src="assets/demo.gif"
    alt="可解释肥胖程度预测系统操作演示"
    width="950"
  >
</p>

<p align="center">
  <strong>
    输入个人信息 → 生成预测结果 → 查看 SHAP 特征贡献 →
    调整 What-if 参数 → 比较反事实结果
  </strong>
</p>

<div align="center">

### [打开在线应用](https://obesity-xai-prediction.onrender.com)

本项目部署在 Render 免费实例上。长时间无人访问后，首次打开可能需要等待服务器自动唤醒。

</div>

---

## 项目简介

许多机器学习系统只能输出预测结果，却无法说明模型为什么作出这一判断。

本项目不仅完成肥胖程度分类，还尝试回答三个实际问题：

1. **模型预测的肥胖等级是什么？**
2. **哪些特征影响了这个人的预测结果？**
3. **修改部分变量后，预测结果是否会发生变化？**

系统使用人口统计、身体状况、饮食习惯和生活方式等特征，预测七种肥胖程度类别。

系统同时提供：

- 肥胖等级预测；
- 模型置信度；
- SHAP Waterfall 局部解释；
- 可调整的 What-if 情景；
- 原始情景与反事实情景对比。

> 本系统仅用于机器学习与可解释人工智能的教育展示，不构成医学诊断或健康建议。

---

## 项目亮点

许多入门机器学习项目在输出准确率后就结束了。

本项目实现了一套较完整的端到端流程：

```text
结构化数据
    ↓
数据预处理
    ↓
随机森林多分类
    ↓
模型性能评估
    ↓
SHAP 局部解释
    ↓
What-if 反事实分析
    ↓
Gradio 交互式界面
    ↓
Render 云端部署
```

最终成果不仅是一个训练好的模型，也是一款可以通过电脑和手机在线访问的可解释人工智能应用。

---

## 核心功能

### 多类别肥胖程度预测

随机森林模型预测七种肥胖程度类别：

- 体重不足
- 正常体重
- 超重一级
- 超重二级
- 肥胖一级
- 肥胖二级
- 肥胖三级

### SHAP 局部可解释性分析

SHAP Waterfall 图用于解释单个用户的预测结果。

图中可以观察：

- 哪些特征推动模型趋向当前预测类别；
- 哪些特征降低模型对当前类别的判断；
- 每个特征贡献的相对大小。

### What-if 反事实情景分析

用户可以调整以下变量：

- 体重；
- 运动频率；
- 蔬菜摄入频率；
- 饮水量。

系统会比较原始情景和修改后情景的：

- 预测肥胖等级；
- 模型置信度；
- 特征取值；
- 预测类别是否发生变化。

### 交互式作品集界面

Gradio 网页包含：

- 分类输入区域；
- 响应式页面布局；
- 模型性能指标卡；
- 预测置信度展示；
- SHAP 图；
- 反事实对比表；
- 重置功能；
- 电脑端和移动端支持。

---

## 模型表现

随机森林模型使用 **200 棵决策树**，并采用分层 80/20 训练测试集划分。

| 评估指标 | 结果 |
|---|---:|
| 测试集准确率 | **95.7%** |
| 加权 F1-score | **95.8%** |
| 加权 AUC — One-vs-Rest | **99.7%** |

上述指标仅反映模型在当前数据集及测试集划分上的表现，不代表临床验证结果。

---

## 项目截图

<details>
<summary><strong>预测结果概览</strong></summary>

<br>

<img
  src="assets/prediction_summary.png"
  alt="肥胖程度预测结果"
  width="850"
>

</details>

<details>
<summary><strong>SHAP Waterfall 局部解释</strong></summary>

<br>

<img
  src="assets/shap_waterfall.png"
  alt="SHAP Waterfall 特征贡献图"
  width="850"
>

</details>

<details>
<summary><strong>What-if 反事实对比</strong></summary>

<br>

<img
  src="assets/counterfactual_comparison.png"
  alt="What-if 反事实对比结果"
  width="850"
>

</details>

<details>
<summary><strong>完整网页界面</strong></summary>

<br>

<img
  src="assets/web_interface.png"
  alt="可解释肥胖程度预测系统完整界面"
  width="950"
>

</details>

---

## 数据集

项目使用 **Obesity Levels Based on Eating Habits and Physical Condition** 数据集。

模型使用 16 个输入特征：

| 特征类别 | 示例 |
|---|---|
| 人口统计 | 性别、年龄 |
| 身体数据 | 身高、体重 |
| 饮食习惯 | 高热量食物、蔬菜摄入、正餐次数 |
| 生活方式 | 吸烟、饮酒、热量监控 |
| 活动情况 | 运动频率、电子设备使用时间 |
| 背景信息 | 家族超重史 |
| 出行方式 | 主要交通方式 |

目标变量为：

```text
NObeyesdad
```

其中包含七种肥胖程度类别。

完整数据字段说明：

**[查看数据集说明](data/README.md)**

---

## 可解释人工智能方法

### SHAP

SHAP 用于解释随机森林对单个样本作出的预测。

对于每一个用户输入，Waterfall 图会将模型输出拆解为不同特征的贡献值。

这样可以检查模型是否依赖以下因素：

- 体重；
- 家族超重史；
- 年龄；
- 身高；
- 运动频率；
- 饮食习惯；
- 饮水量。

### 基于特征的 What-if 分析

反事实模块在其他特征保持不变的情况下，修改选定变量。

它可以帮助观察：

> 当用户的体重或运动频率发生变化时，模型预测是否会改变？

该模块用于展示模型敏感性和决策边界，不用于提供医学或行为干预建议。

---

## 技术栈

| 模块 | 技术 |
|---|---|
| 编程语言 | Python |
| 数据处理 | Pandas、NumPy |
| 机器学习 | Scikit-learn、Random Forest |
| 可解释人工智能 | SHAP、TreeExplainer |
| 数据可视化 | Matplotlib |
| Web 界面 | Gradio |
| 版本管理 | Git、GitHub |
| 云端部署 | Render |
| 开发环境 | Google Colab、Jupyter Notebook |

---

## 项目结构

```text
obesity-xai-prediction/
│
├── app.py
├── requirements.txt
├── .python-version
├── .gitignore
├── README.md
├── README_CN.md
├── LICENSE
├── ObesityDataSet_raw_and_data_sinthetic.csv
│
├── notebooks/
│   └── obesity_xai_analysis.ipynb
│
├── assets/
│   ├── demo.gif
│   ├── web_interface.png
│   ├── prediction_summary.png
│   ├── shap_waterfall.png
│   └── counterfactual_comparison.png
│
├── docs/
│   └── Explainable Obesity Level Prediction System _ Portfolio.pdf
│
└── data/
    └── README.md
```

---

## 分析 Notebook

完整的机器学习和可解释性分析流程位于：

### [查看分析 Notebook](notebooks/obesity_xai_analysis.ipynb)

Notebook 包含：

1. 数据集加载；
2. 数据检查；
3. 分类变量编码；
4. 分层训练测试集划分；
5. 随机森林训练；
6. Accuracy、F1-score 和 AUC 评估；
7. 混淆矩阵；
8. ROC 曲线；
9. Precision-Recall 曲线；
10. 手动用户样本预测；
11. SHAP 局部解释；
12. What-if 反事实分析。

---

## 本地运行

### 1. 克隆项目

```bash
git clone https://github.com/hee289427-wq/obesity-xai-prediction.git
cd obesity-xai-prediction
```

### 2. 创建虚拟环境

Windows：

```bash
python -m venv venv
venv\Scripts\activate
```

macOS 或 Linux：

```bash
python -m venv venv
source venv/bin/activate
```

### 3. 安装依赖

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. 检查数据集

确认以下文件位于仓库根目录：

```text
ObesityDataSet_raw_and_data_sinthetic.csv
```

### 5. 启动应用

```bash
python app.py
```

在浏览器中打开终端显示的本地地址，通常为：

```text
http://127.0.0.1:10000
```

---

## 项目作品集

项目作品集包含系统设计、模型评估、可解释性方法、网页界面、云端部署和个人贡献说明：

### [查看项目作品集](<docs/Explainable Obesity Level Prediction System _ Portfolio.pdf>)

---

## 项目背景与个人贡献

项目最初的学术研究部分由课程小组合作完成。

当前 GitHub 仓库展示的是我在此基础上独立整理、开发和扩展的作品集版本，主要工作包括：

- 模型实验与性能评估；
- 随机森林参数配置；
- 可复现分析 Notebook；
- SHAP 局部解释功能；
- What-if 反事实分析；
- Gradio 网页开发；
- 用户界面重新设计；
- 输入数据验证；
- GitHub 项目结构整理；
- Render 云端部署；
- 项目作品集制作。

---

## 项目局限

- 模型仅使用一个结构化数据集进行训练，可能无法泛化到其他人群。
- 分类变量使用 Label Encoding。
- 当前应用会在服务启动时重新训练模型。
- 反事实变量相互独立地进行修改。
- 预测结果变化不代表医学上可行的改善建议。
- 该系统尚未经过临床验证。
- Render 免费服务长时间无人访问后会进入休眠状态。

---

## 后续计划

未来可以继续完善：

- 保存并加载预训练模型；
- 增加自动化测试；
- 增加全局 SHAP 分析；
- 比较更多分类模型；
- 评估不同人口群体之间的公平性；
- 检查反事实情景的现实可行性；
- 增加无障碍设计和多语言支持；
- 加入持续集成和自动部署检查。

---

## 作者

**何宸臻（Eric He）**

马来亚大学  
人工智能硕士

- GitHub：[hee289427-wq](https://github.com/hee289427-wq)
- 在线应用：[可解释肥胖程度预测系统](https://obesity-xai-prediction.onrender.com)
- 项目作品集：[查看作品集](<docs/Explainable Obesity Level Prediction System _ Portfolio.pdf>)

---

## 支持项目

如果这个项目对你有帮助，或者让你更好地理解了可解释人工智能：

- 欢迎为仓库点一个 ⭐；
- 体验在线应用；
- 提交问题或改进建议；
- Fork 项目并进行自己的实验。

<div align="center">

### 如果这个项目对你有帮助，欢迎点一个 Star ⭐

[![Star this repository](https://img.shields.io/github/stars/hee289427-wq/obesity-xai-prediction?style=for-the-badge&logo=github&label=为项目点个%20Star)](https://github.com/hee289427-wq/obesity-xai-prediction)

</div>

---

## 开源许可

本项目采用 [MIT License](LICENSE)。

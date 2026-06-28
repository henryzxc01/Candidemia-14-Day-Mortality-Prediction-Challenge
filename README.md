# Candidemia 14-Day Mortality Prediction Challenge

本專案以 **念珠菌血症（Candidemia）患者 14 天死亡率預測** 為主題，透過多種機器學習分類器進行模型訓練與比較，評估不同演算法在臨床死亡風險預測任務中的表現。

專案主要目標是比較各分類器的預測準確程度，並建立一個可重現的資料處理、模型訓練與模型評估流程，作為醫療資料分析與機器學習分類任務的實作範例。

## 專案簡介

念珠菌血症是一種嚴重的侵襲性真菌感染，常見於重症、免疫功能低下、長期住院或使用侵入性醫療裝置的患者。由於此疾病可能造成較高的短期死亡風險，因此若能透過臨床資料建立預測模型，將有助於早期辨識高風險族群。

本專案將 14 天死亡率預測視為一個二元分類問題，並使用不同分類器進行訓練與比較，以找出較適合此任務的模型。

## 專案目標

* 建立 Candidemia 患者 14 天死亡率預測模型
* 比較不同機器學習分類器的預測表現
* 評估模型在醫療分類任務中的準確度與穩定性
* 建立資料前處理、模型訓練與評估的完整流程
* 提供後續模型優化與臨床風險預測研究的基礎

## 預測任務

| 項目   | 說明                           |
| ---- | ---------------------------- |
| 任務類型 | 二元分類 Binary Classification   |
| 預測目標 | 患者是否於 Candidemia 確診後 14 天內死亡 |
| 輸入資料 | 臨床資料、檢驗數值、病患基本資訊等            |
| 輸出結果 | 死亡 / 存活預測結果，或死亡風險機率          |
| 主要目的 | 比較不同分類器的預測準確程度               |

## 使用技術

本專案主要使用 Python 進行資料處理與模型訓練，使用到的套件包括：

* pandas
* NumPy
* SciPy
* scikit-learn
* openpyxl


## 可能使用的分類器

本專案可比較以下常見分類模型：

* Logistic Regression
* K-Nearest Neighbors, KNN
* Support Vector Machine, SVM
* Decision Tree
* Random Forest
* Naive Bayes
* Gradient Boosting
* AdaBoost
* Extra Trees Classifier

實際使用的模型可依資料特性與實驗需求調整。

## 專案流程

```text
資料匯入
   ↓
資料清理與前處理
   ↓
特徵選擇 / 特徵工程
   ↓
切分訓練集與測試集
   ↓
建立多種分類器
   ↓
模型訓練
   ↓
模型預測
   ↓
比較各分類器表現
   ↓
輸出評估結果
```

## 模型評估指標

為了比較不同分類器的表現，本專案可使用以下評估指標：

* F1_score 
* MCC
* AUROC


## 專案結構

目前專案可依照以下架構整理：

```text
Candidemia-14-Day-Mortality-Prediction-Challenge/
├── README.md              # 專案說明文件
├── requirements.txt       # Python 套件需求
├── data/                  # 資料檔案
├── code/                  # 程式碼
└── results/               # 模型評估結果
```

若目前尚未建立上述資料夾，可依實際程式碼與資料擺放方式調整。

## 安裝方式

### 1. 下載專案

```bash
git clone https://github.com/henryzxc01/Candidemia-14-Day-Mortality-Prediction-Challenge.git
cd Candidemia-14-Day-Mortality-Prediction-Challenge
```

### 2. 建立虛擬環境

```bash
python -m venv venv
```

Windows：

```bash
venv\Scripts\activate
```

macOS / Linux：

```bash
source venv/bin/activate
```

### 3. 安裝套件

```bash
pip install -r requirements.txt
```

## 使用方式

若主程式為 `main.py`，可使用以下指令執行：

```bash
python main.py
```

若使用 Jupyter Notebook 進行分析，可執行：

```bash
jupyter notebook
```

並開啟對應的 `.ipynb` 檔案進行資料分析與模型訓練。

## 輸出結果

模型訓練完成後，可輸出各分類器的評估結果，例如：

| Model               | F1_mean | F1_std | mcc_mean | mcc_std | AUROC_mean |
| ------------------- | -------: | --------: | -----: | -------: | ------: |
| Logistic Regression |        - |         - |      - |        - |       - |
| Random Forest       |        - |         - |      - |        - |       - |
| SVM                 |        - |         - |      - |        - |       - |
| KNN                 |        - |         - |      - |        - |       - |

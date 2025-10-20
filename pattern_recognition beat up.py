import numpy as np
import pandas as pd
import os, sys
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import lightgbm as lgb

# ==========================
# 📂 参数设置
# ==========================
beat_up_dataset_path = "data/final_dataset/beat_dataset"
prev_window_size = 5   # log return 窗口长度（11个价格→10个log return）

# ==========================
# 📊 1. 加载数据
# ==========================
beat_up_dataset_list = []
for file in os.listdir(beat_up_dataset_path):
    beat_up_dataset = pd.read_parquet(os.path.join(beat_up_dataset_path, file), engine="pyarrow")
    beat_up_dataset_list.append(beat_up_dataset)

beat_up_dataset = pd.concat(beat_up_dataset_list).reset_index(drop=True)
print(f"✅ number of total beat up data is {len(beat_up_dataset)}")

# ==========================
# 🎯 2. 标签
# ==========================
y = beat_up_dataset["situation flag"].astype(int)
# y = y.replace({3: 1})

# ==========================
# 📈 3. total price seq → log return
# ==========================
def price_seq_to_log_returns(seq):
    prices = np.array(seq, dtype=np.float64)
    prices = prices[-prev_window_size:]
    returns = np.diff(np.log(prices))
    return returns

log_return_expanded = beat_up_dataset["price_prev"].apply(price_seq_to_log_returns)
log_return_df = pd.DataFrame(
    log_return_expanded.tolist(),
    columns=[f"log_return_day_{i}" for i in range(prev_window_size - 1)]
)

beat_up_dataset = pd.concat([beat_up_dataset.reset_index(drop=True), log_return_df], axis=1)

# ==========================
# 🔧 4. 特征列准备
# ==========================
# 转换 Surprise 为数值类型
for col in [f"Surprise {i}" for i in range(8, 0, -1)]:
    beat_up_dataset[col] = pd.to_numeric(beat_up_dataset[col], errors='coerce').fillna(0)

numeric_features = [
    "EPS", "Surprise", "PE", "PE Change", "market cap", "%Px Chg"
] + [f"Surprise {i}" for i in range(8, 0, -1)] + [f"log_return_day_{i}" for i in range(prev_window_size - 1)]

categorical_features = ["Sector", "Quarter"]
feature_cols = numeric_features + categorical_features

X = beat_up_dataset[feature_cols]

# ==========================
# ✂️ 5. 划分训练/测试集
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==========================
# ⚖️ 6. 计算 class_weight
# ==========================
classes = np.unique(y_train)
weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
class_weight_dict = {cls: w for cls, w in zip(classes, weights)}
print("✅ Computed class weights:", class_weight_dict)

# ==========================
# 🧰 7. 构建预处理器
# ==========================
numeric_transformer = Pipeline(steps=[
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# ==========================
# 🌲 8. LightGBM 模型
# ==========================
model = lgb.LGBMClassifier(
    objective='multiclass',
    num_class=3,
    n_estimators=1000,          # ✅ 增加迭代次数，让模型有更多学习机会
    learning_rate=0.02,         # ✅ 降低学习率，让每棵树学得更细
    num_leaves=20,             # ✅ 增加叶子数，提升模型表达能力（默认31）
    max_depth=-1,               # ✅ 不限制树深度，让模型自由分裂
    min_child_samples=5,        # ✅ 减少分裂所需最小样本数，避免早停
    min_split_gain=0.0,         # ✅ 放宽分裂增益阈值，确保不会太早停止
    subsample=0.8,              # ✅ 随机采样，提升泛化
    colsample_bytree=0.8,       # ✅ 特征子采样，减少过拟合
    reg_alpha=0.1,              # ✅ L1 正则化，控制复杂度
    reg_lambda=0.1,             # ✅ L2 正则化，控制复杂度
    class_weight=class_weight_dict,  # ✅ 保持类别权重
    random_state=42,
    force_col_wise=True         # ✅ 多分类小数据建议开启，提高效率
)

clf = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", model)
])

# ==========================
# 🚀 9. 模型训练
# ==========================
clf.fit(X_train, y_train)

# ==========================
# 📊 10. 预测与评估
# ==========================
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)

# 📊 打印标签占比（真实数据分布）
print("\n📊 Label distribution (train set):")
print(y_train.value_counts(normalize=True).sort_index().apply(lambda x: f"{x:.2%}"))

print("\n📊 Label distribution (test set):")
print(y_test.value_counts(normalize=True).sort_index().apply(lambda x: f"{x:.2%}"))


# 📊 打印权重占比（不用于模型，仅分析用）
total_weight = sum(class_weight_dict.values())
weight_ratio = {cls: w / total_weight for cls, w in class_weight_dict.items()}

print("\n📊 Class weight ratios (not normalized for training, only for reference):")
for cls, ratio in weight_ratio.items():
    print(f"  Class {cls}: {ratio:.2%}")


print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("✅ F1 Score (macro):", f1_score(y_test, y_pred, average='macro'))
print("✅ ROC-AUC (OvR):", roc_auc_score(y_test, y_proba, multi_class='ovr'))
# print("✅ ROC-AUC:", roc_auc_score(y_test, y_proba[:, 1]))



print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred))

# ==========================
# 📊 11. 特征重要性提取
# ==========================

# 1️⃣ 拿到训练好的 LightGBM 模型
lgb_model = clf.named_steps["classifier"]

# 2️⃣ 获取 OneHot 编码后的特征名
# 数值特征名（不用变）
num_feature_names = numeric_features

# 类别特征名（需要从 onehot 拿出来）
ohe_feature_names = clf.named_steps["preprocessor"].named_transformers_["cat"]["onehot"].get_feature_names_out(categorical_features)

# 合并成完整特征名
all_feature_names = np.concatenate([num_feature_names, ohe_feature_names])

# 3️⃣ 获取特征重要性
importances = lgb_model.feature_importances_

# 4️⃣ 构建 DataFrame 排序展示
importance_df = pd.DataFrame({
    "Feature": all_feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False).reset_index(drop=True)

print("\n🌟 Top 20 Most Important Features:")
print(importance_df.head(20))

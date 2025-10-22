import numpy as np
import pandas as pd
import os, sys
from datetime import datetime
import matplotlib.pyplot as plt
import shutup
from tqdm import tqdm
from utils.data_process_toolkit import month_to_quarter
from utils.data_preparation_toolkit import situatuion_judgement, situatuion_judgement2
from API.api import get_price_from_csv, get_market_cap_from_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import lightgbm as lgb

shutup.please()
# parameters
influence_period = 20 # trading days
lower_date_boundry = "2020-01-01" 
upper_date_boundry = "2025-05-01"
output_ols_report = True
num_total_quarter = 8
price_prev_window = 20
price_after_window = 20
random_seed = 42
beat_threshold = 0.1


PE_check_flag = False
Raw_data_process = False
beat_up_dataset_construction = False
situation_prediction = False
beat_analysis_excel_construction = True


path = "data/ern"
sector_df_path = "data/spx_sector.csv"
total_equity_df_path = "data/final_dataset/beat_dataset/total_equity_df_4sit.parquet"
final_dataset_4sit_path = "data/final_dataset/beat_dataset/final_dataset_4sit_path.parquet"
price_path = "data/price"
beat_analysis_path = "data/final_dataset/beat_analysis/"

########################################### Run PE Check ###########################################

if PE_check_flag:
    for file in os.listdir(path):
        df = pd.read_excel(os.path.join(path, file), engine="openpyxl")
        columns = df.columns
        if "P/E" not in columns:
            print(f"[Error] {file} does not have P/E column")
            sys.exit(1)
        else:
            print(f"{file} check passed")

########################################### Raw Data Process ###########################################
if Raw_data_process:
    total_equity_df_list = []
    sector_df = pd.read_csv(sector_df_path)
    sector_dic = dict(zip(sector_df["Ticker"], sector_df["Sector-Code"]))
    for equity in tqdm(os.listdir(path), desc="Raw Data Processing"):  
        columns = []
        equity_name = equity[:-5]
        data = pd.read_excel(os.path.join(path, equity), 
                            engine="openpyxl")
        data["Equity name"] = equity_name
        data["Ann Date"] = pd.to_datetime(data["Ann Date"], errors='coerce')
        data["Prev Ann Date"] = data["Ann Date"].shift(-1)
        data["Next Ann Date"] = data["Ann Date"].shift(1)

        data["EPS"] = data["Comp"]
        try:
            data["Surprise"] = data["%Surp"].replace("N.M.", "0").str.rstrip("%").astype(float) / 100
        except Exception:
            print(f"{equity_name} encountered a problem while transforming Surprise")
            sys.exit(1)

        for i in range(len(data)):
            pe = data.loc[i, "P/E"]
            if type(pe) is str:
                if 'k' in pe:
                    data.loc[i, "P/E"] = float(pe.strip("k")) * 1000
                elif pe == '':
                    pass
                else:
                    tqdm.write(f"{equity_name} has wrong value in PE")
                    sys.exit(1)
        data["PE"] = data["P/E"]
        data["PE Change"] = data["PE"].pct_change(periods=-1)

        data["%Px Chg"] = data["%Px Chg"].replace("N.M.", "0").str.rstrip("%").astype(float) / 100
        # Up: 1, Down: 0
        data["Up Down Flag"] = data["%Px Chg"].apply(lambda x: 1 if x > 0 else 0)
        # Beat: 1, Miss: 0
        data["Beat Miss Flag"] = data["Surprise"].apply(lambda x: 1 if x>0 else 0)

        try:
            data["Sector"] = sector_dic[equity_name]
        except KeyError as e:
            print(f"{equity_name} has no sector code")
            sys.exit(1)

        data["Quarter"] = data["Per End"].apply(month_to_quarter)

        columns.extend(["Equity name", "Ann Date", "Prev Ann Date", "Next Ann Date", "EPS", "Surprise", "PE", "PE Change", "Up Down Flag", "Beat Miss Flag", "Sector", "Quarter", "%Px Chg"])
        for i in range(num_total_quarter):
            col = f"Surprise {8-i}"
            data[col] = data["Surprise"].shift(8-i)
            columns.append(col)
        
        start_date_index = (data["Ann Date"] - pd.to_datetime(lower_date_boundry)).abs().idxmin()
        end_date_index = (data["Ann Date"] - pd.to_datetime(upper_date_boundry)).abs().idxmin()
        data = data.iloc[end_date_index:start_date_index,:]

        equity_df = data[columns]
        equity_df.dropna(how="any", inplace=True)
        if not equity_df.empty:
            total_equity_df_list.append(equity_df)

    total_equity_df = pd.concat(total_equity_df_list).reset_index(drop=True)
    total_equity_df = total_equity_df.sample(frac=1, random_state=42).reset_index(drop=True)
    total_equity_df = total_equity_df.reset_index(drop=True)
    print(f"number of total data is {len(total_equity_df)}")
    total_equity_df.to_parquet(total_equity_df_path, engine="pyarrow", index=False)

########################################### Beat Up dataset consturction ###########################################
if beat_up_dataset_construction:
    print("Commencing Beat Up Dataset Construction")
    total_equity_df= pd.read_parquet(total_equity_df_path, engine="pyarrow")
    total_equity_df["total price seq"] = None
    total_equity_df["market cap"] = None
    total_equity_df["situation flag"] = None
    total_equity_df["situation details"] = None
    total_equity_df["price_prev"] = None
    # total_equity_df = total_equity_df[(total_equity_df["Beat Miss Flag"] ==1)&(total_equity_df["Up Down Flag"] == 1)&(total_equity_df["Surprise"] >=beat_threshold)].reset_index(drop=True)
    total_equity_df = total_equity_df[(total_equity_df["Beat Miss Flag"] ==1)&(total_equity_df["Surprise"] >=beat_threshold)].reset_index(drop=True)
    print(f"number of beat data is {len(total_equity_df)}")
    for idx, row in tqdm(total_equity_df.iterrows(), 
                    total=total_equity_df.shape[0], 
                    desc=f"Beat Up Dataset Construction"):
        equity_name = row["Equity name"]
        ann_date = row["Ann Date"]
        prev_ann_date = row["Prev Ann Date"]
        next_ann_date = row["Next Ann Date"]
        px_change = row["%Px Chg"]
        full_price_seq = get_price_from_csv(equity_name,prev_ann_date, next_ann_date, price_path).reset_index(drop=True)
        market_cap = get_market_cap_from_csv(equity_name, ann_date, price_path)
        try:
            day0_idx = full_price_seq.index[full_price_seq["Date"] == ann_date][0]
            day0_price = full_price_seq.loc[day0_idx, "Price"]
            day1_price = full_price_seq.loc[day0_idx+1, "Price"]
            day1_price_change = (day1_price - day0_price) / day0_price
            if abs(day1_price_change - px_change) > 0.001:
                price_idx = day0_idx - 1
            else:
                price_idx = day0_idx
            price_seq_prev = full_price_seq["Price"][price_idx+1-price_prev_window:price_idx+1].tolist()
            price_seq_after = full_price_seq[["Price"]][price_idx:price_idx+price_after_window+1]
        except Exception as e:
            total_equity_df.loc[idx, "situation flag"] = None
            total_equity_df.at[idx, "total price seq"] = None
            total_equity_df.loc[idx, "market cap"] = None
            total_equity_df.loc[idx, "price_prev"] = None
            continue

        if len(price_seq_prev) < price_prev_window or len(price_seq_after) < price_after_window:
            total_equity_df.loc[idx, "situation flag"] = None
            total_equity_df.at[idx, "total price seq"] = None
            total_equity_df.loc[idx, "market cap"] = None
            total_equity_df.loc[idx, "price_prev"] = None
            continue
        else:
            situation_flag, situation_details = situatuion_judgement(price_seq_after, price_after_window)

        total_equity_df.loc[idx, "situation flag"] = situation_flag
        total_equity_df.at[idx, "total price seq"] = full_price_seq["Price"].tolist()
        total_equity_df.loc[idx, "market cap"] = market_cap
        total_equity_df.at[idx, "price_prev"] = price_seq_prev
    print(f"number of beat data befroe dropna: {len(total_equity_df)}")
    total_equity_df.dropna(subset="situation flag", inplace=True)
    print(f"number of validated beat data is {len(total_equity_df)}")
    total_equity_df.to_parquet(final_dataset_4sit_path, engine="pyarrow", index=False)
    
########################################### Situation Prediction ###########################################
if situation_prediction:
    prev_window_size = 5   # log return 窗口长度（11个价格→10个log return）

    beat_dataset = pd.read_parquet(final_dataset_4sit_path, engine="pyarrow")
    # 去掉beat_down情况
    beat_dataset = beat_dataset[beat_dataset["situation flag"] != 4].reset_index(drop=True)
    print(f"✅ number of total beat up data is {len(beat_dataset)}")
    y = beat_dataset["situation flag"].astype(int)
    # y = y.replace({3: 1})

    def price_seq_to_log_returns(seq):
        prices = np.array(seq, dtype=np.float64)
        prices = prices[-prev_window_size:]
        returns = np.diff(np.log(prices))
        return returns

    log_return_expanded = beat_dataset["price_prev"].apply(price_seq_to_log_returns)
    log_return_df = pd.DataFrame(
        log_return_expanded.tolist(),
        columns=[f"log_return_day_{i}" for i in range(prev_window_size - 1)]
    )

    beat_up_dataset = pd.concat([beat_dataset.reset_index(drop=True), log_return_df], axis=1)

    # 转换 Surprise 为数值类型
    for col in [f"Surprise {i}" for i in range(8, 0, -1)]:
        beat_up_dataset[col] = pd.to_numeric(beat_up_dataset[col], errors='coerce').fillna(0)

    numeric_features = [
        "EPS", "Surprise", "PE", "PE Change", "market cap", "%Px Chg"
    ] + [f"Surprise {i}" for i in range(8, 0, -1)] + [f"log_return_day_{i}" for i in range(prev_window_size - 1)]

    categorical_features = ["Sector", "Quarter"]
    feature_cols = numeric_features + categorical_features

    X = beat_up_dataset[feature_cols]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight_dict = {cls: w for cls, w in zip(classes, weights)}

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

    model = lgb.LGBMClassifier(
        objective='multiclass',
        num_class=3,
        n_estimators=1000,          # ✅ 增加迭代次数，让模型有更多学习机会
        learning_rate=0.01,         # ✅ 降低学习率，让每棵树学得更细
        num_leaves=31,             # ✅ 增加叶子数，提升模型表达能力（默认31）
        max_depth=-1,               # ✅ 不限制树深度，让模型自由分裂
        min_child_samples=5,        # ✅ 减少分裂所需最小样本数，避免早停
        min_split_gain=0.0,         # ✅ 放宽分裂增益阈值，确保不会太早停止
        subsample=0.9,              # ✅ 随机采样，提升泛化
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

    clf.fit(X_train, y_train)

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

if beat_analysis_excel_construction:
    total_equity_df= pd.read_parquet(final_dataset_4sit_path, engine="pyarrow")
    total_equity_df["market cap"] = None
    total_equity_df["situation flag"] = None
    total_equity_df["retrace_date"] = None
    total_equity_df["trough_date"] = None
    total_equity_df["trough_loss"] = None
    total_equity_df["peak_date"] = None
    total_equity_df["peak_gain"] = None
    total_equity_df["full_time_peak_date"] = None
    total_equity_df["full_time_peak_gain"] = None
    for i in range(price_after_window+1):
        total_equity_df[f"day_{i}"] = None
    # total_equity_df = total_equity_df[(total_equity_df["Beat Miss Flag"] ==1)&(total_equity_df["Up Down Flag"] == 1)].reset_index(drop=True)
    total_equity_df = total_equity_df[(total_equity_df["Beat Miss Flag"] ==1)&(total_equity_df["Surprise"] >=beat_threshold)].reset_index(drop=True)
    for idx, row in tqdm(total_equity_df.iterrows(), 
                    total=total_equity_df.shape[0], 
                    desc=f"Beat_analysis_excel_construction"):
        equity_name = row["Equity name"]
        ann_date = row["Ann Date"]
        prev_ann_date = row["Prev Ann Date"]
        next_ann_date = row["Next Ann Date"]
        px_change = row["%Px Chg"]
        full_price_seq = get_price_from_csv(equity_name,prev_ann_date, next_ann_date, price_path).reset_index(drop=True)
        market_cap = get_market_cap_from_csv(equity_name, ann_date, price_path)
        try:
            #### Day 0 calibration
            day0_idx = full_price_seq.index[full_price_seq["Date"] == ann_date][0]
            day0_price = full_price_seq.loc[day0_idx, "Price"]
            day1_price = full_price_seq.loc[day0_idx+1, "Price"]
            day1_price_change = (day1_price - day0_price) / day0_price
            if abs(day1_price_change - px_change) > 0.001:
                price_idx = day0_idx - 1
            else:
                price_idx = day0_idx
            price_seq_prev = full_price_seq["Price"][price_idx+1-price_prev_window:price_idx+1].tolist()
            price_seq_after = full_price_seq[["Price"]][price_idx:price_idx+price_after_window+1]
        except Exception as e:
            total_equity_df.loc[idx, "situation flag"] = None
            total_equity_df.loc[idx, "market cap"] = None
            continue
        if len(price_seq_prev) < price_prev_window or len(price_seq_after) < price_after_window:
            total_equity_df.loc[idx, "situation flag"] = None
            total_equity_df.loc[idx, "market cap"] = None
            continue
        else:
            situation_flag, situation_details = situatuion_judgement(price_seq_after, price_after_window)

        total_equity_df.loc[idx, "situation flag"] = situation_flag
        total_equity_df.loc[idx, "market cap"] = market_cap
        for i in range(price_after_window+1):
            total_equity_df.loc[idx, f"day_{i}"] = price_seq_after["Price"].reset_index(drop=True)[i]
        if situation_flag == 1:
            total_equity_df.loc[idx, "retrace_date"] = situation_details["retrace_date"]
            total_equity_df.loc[idx, "trough_date"] = situation_details["trough_date"]
            total_equity_df.loc[idx, "trough_loss"] = situation_details["trough_loss"]
            total_equity_df.loc[idx, "peak_date"] = situation_details["peak_date"]
            total_equity_df.loc[idx, "peak_gain"] = situation_details["peak_gain"]
            total_equity_df.loc[idx, "full_time_peak_date"] = situation_details["full_time_peak_date"]
            total_equity_df.loc[idx, "full_time_peak_gain"] = situation_details["full_time_peak_gain"]

    print(f"number of data befroe dropna: {len(total_equity_df)}")
    total_equity_df.dropna(subset="situation flag", inplace=True)
    print(f"number of validated is {len(total_equity_df)}")
    situation_1 = total_equity_df[total_equity_df["situation flag"]==1].reset_index(drop=True)
    situation_2 = total_equity_df[total_equity_df["situation flag"]==2].reset_index(drop=True)
    situation_3 = total_equity_df[total_equity_df["situation flag"]==3].reset_index(drop=True)
    situation_4 = total_equity_df[total_equity_df["situation flag"]==4].reset_index(drop=True)
    output_path = os.path.join(beat_analysis_path, f"beat_analysis_{beat_threshold*100}.xlsx")

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        situation_1.to_excel(writer, sheet_name="situation 1", index=False)
        situation_2.to_excel(writer, sheet_name="situation 2", index=False)
        situation_3.to_excel(writer, sheet_name="situation 3", index=False)
        situation_4.to_excel(writer, sheet_name="situation 4", index=False)

    print(f"number of situation 1 is {len(situation_1)}")
    print(f"number of situation 2 is {len(situation_2)}")
    print(f"number of situation 3 is {len(situation_3)}")
    print(f"number of situation 4 is {len(situation_4)}")
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from utils.data_process_toolkit import generate_ols_report
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from collections import Counter
import os, io

PLT_PATH = r"figure/"
REPORT_PATH = r"report/"
XLSX_PATH = r"xlsx/"

def path_check(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        os.makedirs(folder_path + PLT_PATH)
        os.makedirs(folder_path + REPORT_PATH)

def pdf_write(filename, title, content):
    pdf = canvas.Canvas(filename)
    pdf.setFont("Helvetica", 10)

    # 设置初始位置
    y_position = 800  
    x_position = 50

    # 标题
    pdf.drawString(x_position, y_position, title)
    y_position -= 20  # 换行

    # 逐行写入 DataFrame 内容
    for line in content.split("\n"):
        pdf.drawString(x_position, y_position, line)
        y_position -= 15  # 逐行向下移动
    pdf.save()

def pdf_plot_write(filename, title, content, img_buffer):
    return 0

def output_beat_miss_statistical_feature(security):
    # statistic results
    print(security["name"])

    print(f'number of beat ups: {len(security["beat_up"])}')
    print(f'number of beat downs: {len(security["beat_down"])}')
    print(f'number of miss ups: {len(security["miss_up"])}')
    print(f'number of miss downs: {len(security["miss_down"])}')

    print(f'beat_surprise_avg: {security["statistics"]["beat_surprise_avg"]:.3f}')
    print(f'miss_surprise_avg: {security["statistics"]["miss_surprise_avg"]:.3f}')
    print(f'beat_surprise_var: {security["statistics"]["beat_surprise_var"]:.3f}')
    print(f'miss_surprise_var: {security["statistics"]["miss_surprise_var"]:.3f}')

    print(f'beat_surprise_max: {security["statistics"]["beat_surprise_max"]:.3f}')
    print(f'beat_surprise_min: {security["statistics"]["beat_surprise_min"]:.3f}')
    print(f'beat_surprise_max: {security["statistics"]["beat_surprise_max"]:.3f}')
    print(f'miss_surprise_min: {security["statistics"]["miss_surprise_min"]:.3f}')

def output_beat_miss_beta(security_list, avg_beta_list, situation):
    print(f"\nOutputing AVG Beta when {situation}:")
    for i, (security, average_beta) in enumerate(zip(security_list, avg_beta_list)):
        print(f"AVG Beta for {security} is {average_beta:.3f}")

def plot_alpha_figure(security_list, avg_excess_return_list, influence_period, situation, region, sector):
    # Plot avergae alpha
    plt.figure(figsize=(15, 8))
    for i, (security, avg_excess_return) in enumerate(zip(security_list, avg_excess_return_list)):
        plt.plot(range(1, len(avg_excess_return)), avg_excess_return[1:], marker='o', linestyle='-', label=security)

    plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
    
    # title = "Avg Excess Return for Multiple Securities " + "when "+ situation+ " " + str(influence_period) + " trading days"
    title = f"Avg Alpha [{situation}, Region-{region}, Sector-{sector} trading days-{influence_period}]"
    plt.title(title)
    plt.xlabel("Time (Trading Days)")
    plt.ylabel("AVG Alpha")

    # **设置 X 轴刻度从 0 开始**
    plt.xticks(range(0, len(avg_excess_return) + 1))  # 让 X 轴从 0 开始，但数据仍然从 1 开始


    # **设置图例到最右侧**
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0) 

    # **调整布局，确保图例不会被裁剪**
    plt.tight_layout(rect=[0, 0, 0.85, 1]) 

    plt.savefig(PLT_PATH + title + f".png")

def plot_alpha_beatmiss_ratio_figure(total_beatmiss_size_alpha, avg_alpha_period, situation, region, sector, output_ols_report):
    total_beatmiss_size_alpha = np.array(total_beatmiss_size_alpha)
    alpha_list = total_beatmiss_size_alpha[:,0]
    beatmiss_list = total_beatmiss_size_alpha[:,1]

    plt.figure(figsize=(8, 6))
    plt.scatter(beatmiss_list, alpha_list, color='b', alpha=0.7) 
    plt.ylabel("Alpha")
    plt.xlabel(f"{situation} size")
    title = f"Alpha and {situation} size [Alpha Time Window-{avg_alpha_period}, Region-{region}, Sector-{sector}]"
    plt.title(title)
    plt.ylim(-0.025, 0.2)
    plt.xlim(-0.5, 0.5)
    plt.grid(True)
    plt.savefig(PLT_PATH + title + ".png")

    if output_ols_report:
        model = generate_ols_report(beatmiss_list, alpha_list)
        summary_text = model.summary().as_text()
        summary_text = ""
        # 生成 PDF 并写入文本
        pdf_filename = REPORT_PATH + title + ".pdf"
        pdf = canvas.Canvas(pdf_filename)
        pdf.setFont("Helvetica", 10)

        # 按行写入 PDF，自动换行
        y_position = 800  # 起始位置
        for line in summary_text.split("\n"):
            pdf.drawString(50, y_position, line)
            y_position -= 15  # 逐行下移
        pdf.save()

def plot_stock_price(security_name, stock_price_list, situation, region, sector, influence_period, output_folder_path):
    plt.figure(figsize=(15, 8))
    for item in stock_price_list:
        plt.plot(item[0], marker='o', linestyle='-', label=item[1])
    
    title = f"Stock Price [{security_name} {situation}, Region-{region}, Sector-{sector} trading days-{influence_period}]"
    plt.title(title)
    plt.xlabel("Time (Trading Days)")
    plt.ylabel("Stock Return")

    plt.axhline(y=0, color='black', linestyle='--', linewidth=1)

    # **设置图例到最右侧**
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0) 

    # **调整布局，确保图例不会被裁剪**
    plt.tight_layout(rect=[0, 0, 0.85, 1]) 
    file_path = output_folder_path + PLT_PATH + title + f".png"
    plt.savefig(file_path)

def report_overall_feature_cross_region_sector(overall_beatmiss_list, situation_list, region, sector, output_folder_path):
    title = f"Overall Statistical Feature [{region}, {sector}]"
    overall_content = ""
    for beat_miss_list, situation in zip(overall_beatmiss_list, situation_list):
        beatmiss_count = len(beat_miss_list)
        surprise_size = []
        for beatmiss in beat_miss_list:
            surprise_size.append(np.abs(beatmiss["%Surp"]))
        beatmiss_df = pd.DataFrame(surprise_size, columns=["Surprise"])
        features = beatmiss_df.describe()
        overall_content = overall_content + f"***********************situation: {situation}***********************\n" + f"number of {situation} is: {beatmiss_count}\n" + features.to_string() + "\n"    
    file_name = output_folder_path + REPORT_PATH + title + ".pdf"
    pdf_write(file_name, title, overall_content)

def plot_retrace_distribution(retrace, situation, region, sector, output_folder_path):
    '''
        Input: retrace list (if there's no retrace: -1)
        Type: Numpy
    '''
    filtered_data = retrace[retrace != -1]
    counts = Counter(filtered_data)
    x_values = np.array(list(counts.keys())) 
    y_values = np.array(list(counts.values())) 

    plt.figure(figsize=(12, 5))
    plt.bar(x_values, y_values, color="b", alpha=0.7, edgecolor="black")
    title = f"Distribution of Retrace, Situation-{situation}, Region-{region}, Sector-{sector}"
    plt.title(title)
    plt.xlabel("Retrace (Days)")
    plt.ylabel("Counts")
    plt.xticks(x_values)  # 确保 X 轴显示整数
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    file_name = output_folder_path + PLT_PATH + title + ".png"
    plt.savefig(file_name)

def plot_retrace_distribution_list(retrace_list, situation_list, region, sector, output_folder_path):
    retrace_statistics = []
    title = "Distribution of Retrace [Region-{region}, Sector-{sector}]"
    for situation, retrace in zip(situation_list, retrace_list):
        retrace = np.array(retrace)
        # Calculating Retrace Statistics
        retrace_statistics.append({"situation name": situation, "number of retrace": np.sum(retrace!=-1), "number of not retrace": np.sum(retrace==-1)})
        # Ploting Retrace Distribution
        filtered_data = retrace[retrace != -1]
        counts = Counter(filtered_data)
        x_values = np.array(list(counts.keys())) 
        y_values = np.array(list(counts.values())) 

        plt.figure(figsize=(12, 5))
        plt.bar(x_values, y_values, color="b", alpha=0.7, edgecolor="black")
        title = f"Distribution of Retrace, Situation-{situation}, Region-{region}, Sector-{sector}"
        plt.title(title)
        plt.xlabel("Retrace (Days)")
        plt.ylabel("Counts")
        plt.xticks(x_values)  # 确保 X 轴显示整数
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        file_name = output_folder_path + PLT_PATH + title + ".png"
        plt.savefig(file_name)

def plot_retrace_beatsize(retrace, situation, region, sector, output_ols_report, output_folder_path):
    title = f"Relationship of Retrace-Beat_size [{situation}-{region}-{sector}]"
    retrace = retrace[retrace[:,0]!=-1]
    retrace_data = retrace[:,0]
    beatmiss_size = retrace[:,1]
    plt.figure(figsize=(8, 6))
    plt.scatter(beatmiss_size, retrace_data, color='b', alpha=0.7) 
    plt.ylabel("Retrace Period")
    plt.xlabel(f"{situation} size")
    plt.title(title)
    plt.ylim(0, 60)
    plt.xlim(-0.5, 0.5)
    plt.grid(True)
    plt.savefig(output_folder_path + PLT_PATH + title + ".png")
    if output_ols_report:
        model = generate_ols_report(beatmiss_size, retrace_data)
        summary_text = model.summary().as_text()
        pdf_path = output_folder_path + REPORT_PATH + title + ".pdf"
        pdf_write(pdf_path, title, summary_text)


def report_retrace_statistics(retrace_statistics, region, sector, output_folder_path):
    title = f"Retrace Statistics [{region}, {sector}]"
    file_name = output_folder_path + REPORT_PATH + title + ".pdf"
    content = ""
    for situation in retrace_statistics:
        situation_content = f"{situation['situation name']}:\n" + f"number of retrace: {situation['number of retrace']}\n" + f"number of not retrace: {situation['number of not retrace']}\n"
        content += situation_content
    pdf_write(file_name, title, content)
    

def security_revenue_data_to_xlsx(security_revenue_data, region, sector, influence_period, output_folder_path, surprise_beat_threshold, surprise_miss_threshold):
    overall_beat_up_list, overall_beat_down_list, overall_miss_up_list, overall_miss_down_list = [], [], [], []
    overall_beatmiss_list = [overall_beat_up_list, overall_beat_down_list, overall_miss_up_list, overall_miss_down_list]
    situation_list = ["beat_up", "beat_down", "miss_up", "miss_down"]
    for security in security_revenue_data:
        security_name = security["name"]
        for situation, beatlist in zip(situation_list, overall_beatmiss_list):
            for each_beat_up in security[situation]:
                ann_date = each_beat_up["Ann Date"]
                next_ann_date = each_beat_up["Next Ann Date"]
                surprise = each_beat_up["%Surp"]
                retrace_date = each_beat_up["Retrace"]["retrace_date"]
                retrace_rate = each_beat_up["Retrace"]["retrace_rate"]
                recover_date = each_beat_up["Retrace"]["recover_date"]
                trend = each_beat_up["Retrace"]["trend"]
                dict_data = {"security_name": security_name, 
                            "ann_date": ann_date,
                            "next_ann_date": next_ann_date,
                            "surprise": surprise,
                            "retrace_date": retrace_date,
                            "retrace_rate": retrace_rate,
                            "recover_date": recover_date,
                            "trend": trend}
                beatlist.append(dict_data)
    path = output_folder_path + f"{region} {sector} {surprise_beat_threshold*100}beat {surprise_miss_threshold*100}miss {influence_period}days analysis result.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for situation, beatlist in zip(situation_list, overall_beatmiss_list):
            df = pd.DataFrame(beatlist)
            df.to_excel(writer, sheet_name=situation, index=False)

def beat_analysis_data_to_xlsx(security_revenue_data, sector, influence_period, output_folder_path, surprise_beat_threshold):
    situation_1, situation_2, situation_3, situation_4 = [], [], [], []
    overall_beat_list = [situation_1, situation_2, situation_3, situation_4]
    situation_list = [1, 2, 3, 4]

    for idx, row in security_revenue_data.iterrows():
        security_name = row["equity_name"]
        for situation, beatlist in zip(situation_list, overall_beat_list):
            if row["situation_flag"] == situation:
                ann_date = row["ann_date"]
                next_ann_date = row["next_ann_date"]
                surprise = row["sup"]
                beat_detail = row["beat_detail"]
                
                if situation == 1:
                    retrace_date = beat_detail["retrace_date"]
                    trough_date = beat_detail["trough_date"]
                    trough_loss = beat_detail["trough_loss"]
                    peak_date = beat_detail["peak_date"]
                    peak_gain = beat_detail["peak_gain"]
                    full_time_peak_date = beat_detail["full_time_peak_date"]
                    full_time_peak_gain = beat_detail["full_time_peak_gain"]

                else:
                    retrace_date = ""
                    trough_date = ""
                    trough_loss = ""
                    peak_date = ""
                    peak_gain = ""
                    full_time_peak_date = ""
                    full_time_peak_gain = ""

                stock_price_data = row["stock_price"]["Price"].to_numpy()
                stock_price_dict = {}
                for i in range(stock_price_data.shape[0]):
                    stock_price_dict[f"day{i}"] = stock_price_data[i]
                dict_data = {"security_name": security_name, 
                            "ann_date": ann_date,
                            "next_ann_date": next_ann_date,
                            "surprise": surprise,
                            "retrace_date": retrace_date,
                            "trough_date": trough_date,
                            "trough_loss": trough_loss,
                            "peak_date": peak_date,
                            "peak_gain": peak_gain,
                            "full_time_peak_date":full_time_peak_date, 
                            "full_time_peak_gain":full_time_peak_gain}
                dict_data.update(stock_price_dict)
                beatlist.append(dict_data)
    path = output_folder_path + f"Beat Analysis {sector} {surprise_beat_threshold*100}beat {influence_period}days.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for situation, beatlist in zip(situation_list, overall_beat_list):
            df = pd.DataFrame(beatlist)
            df.to_excel(writer, sheet_name=f"situation {situation}", index=False)
    

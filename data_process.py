import os
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import pickle
from renamedict import *

def judge_df_id_is_equal(df1, df2, colname="eid"):
    #将eid这一列转换成list，然后判定两张表中的eid是否完全一致
    if set(df1[colname]) == set(df2[colname]):
        return True
    else:
        return False

def merged_event2_and_deadly_disease():
    #合并事件2和死亡疾病表格，只提取一些必要的列
    df1=pd.read_csv('data_original/事件2.csv',encoding="gbk")
    df2=pd.read_csv('data_original/死亡疾病.csv',encoding="gbk")
    is_equal= judge_df_id_is_equal(df1,df2)
    if is_equal:
        merged_df = pd.merge(df1, df2, on='eid')
        reser_col = ['eid','全因死亡','随访时间-全因死亡','卒中_','心衰_','Miocardial_infarction_','CVDD','卒中_ICD',"心衰_ICD","Miocardial_infarction_ICD"]
        merged_df = merged_df[reser_col]
        file_name = "data_process/mace_mace_min_time.csv"
        merged_df.to_csv(file_name,index=False)
        print(merged_df)
        return merged_df

def calculate_min(row):
    #计算'卒中_'，'心衰_'，'Miocardial_infarction_'，'随访时间-全因死亡'中的最小时间
    values = [row['卒中_'], row['心衰_'], row['Miocardial_infarction_'],row['随访时间-全因死亡']]
    valid_values = [value for value in values if not pd.isnull(value)]
    if valid_values:
        return min(valid_values)
    else:
        return row['随访时间-全因死亡']

def calculate_mace(row):
    #卒中_、心衰_、Miocardial_infarction_中不为空 || CVDD为1，MACE就为1
    values = [row['卒中_'], row['心衰_'], row['Miocardial_infarction_']]
    valid_values = [value for value in values if not pd.isnull(value)]
    if valid_values or row['CVDD'] == 1:
        return 1
    return 0

def get_mace_and_min_time():
    file_name = "data_process/mace_mace_min_time.csv"
    if(os.path.exists(file_name)):
        mace_data=pd.read_csv(file_name)
    else:        
        mace_data=merged_event2_and_deadly_disease()
    mace_data['MACE'] = mace_data.apply(calculate_mace, axis=1)
    mace_data['MIN_TIME'] = mace_data.apply(calculate_min, axis=1)
    mace_data.to_csv(file_name,index=False)
    return mace_data

def get_behavioral_data(misspercent=0.1):
    #行为数据合并成一个pkl
    #获取路径下的所有文件
    folder_path = 'data_original/行为数据'
    all_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".xlsx") or file.endswith(".csv"):
                all_files.append(os.path.join(root, file))
    print(all_files)
    #读取所有文件并合并成一个DataFrame
    data_frame_list = []
    merged_df = pd.DataFrame()
    for file_path in all_files:
        if file_path.endswith(".xlsx"):
            df = pd.read_excel(file_path)
        elif file_path.endswith(".csv"):
            df = pd.read_csv(file_path,encoding='utf-8')
        data_frame_list.append(df)
    merged_df = pd.concat(data_frame_list,axis=1)
    # 去除列名中带有Unnamed的列
    columns_to_drop = merged_df.columns[merged_df.columns.str.contains('Unnamed')]
    merged_df = merged_df.drop(columns=columns_to_drop)
    # 计算每行的缺失值比例
    row_missing_percentages = merged_df.isnull().mean(axis=1)
    merged_df['MISS_FLAG'] = row_missing_percentages.apply(lambda x: 1 if x > misspercent else 0)
    print("get_behavioral_data\n",merged_df)
    # 保存合并后的 DataFrame 为 pkl 文件
    merged_df.to_pickle(f"data_process/behavioral_data_{misspercent}.pkl")
    return merged_df

def calculate_mace_history(row):
    return (row["心衰_history"] | row["心肌梗死_history"] | row["卒中_history"])

def get_mace_history_data():
    #需要疾病史汇总.csv转成疾病史汇总.xlsx
    mace_history_data=pd.read_excel('data_original/疾病史汇总.xlsx')
    mace_history_data['MACE_HISTORY'] = mace_history_data.apply(calculate_mace_history, axis=1)
    mace_history_data_process = "data_process/mace_history_data.csv"
    mace_history_data.to_csv(mace_history_data_process,index=False)
    return mace_history_data

def get_mace_all_data(misspercent=0.1):
    behavioral_data_path = f'data_process/behavioral_data_{misspercent}.pkl'
    mace_min_time_path = "data_process/mace_mace_min_time.csv"
    mace_history_dt_path = "data_process/mace_history_data.csv"

    if(os.path.exists(behavioral_data_path)):
        behavioral_data = pd.read_pickle(behavioral_data_path)
    else:
        behavioral_data = get_behavioral_data(misspercent)
    if(os.path.exists(mace_min_time_path)):
        mace_min_time_data = pd.read_csv(mace_min_time_path)
    else:
        mace_min_time_data = get_mace_and_min_time()

    if(os.path.exists(mace_history_dt_path)):
        mace_history_data = pd.read_csv(mace_history_dt_path)
    else:
        mace_history_data = get_mace_history_data()

    mace_min_time_data.rename(columns={"eid": "Participant ID"}, inplace=True)
    #去除重复的列
    mace_min_time_data = mace_min_time_data.iloc[:, ~mace_min_time_data.columns.duplicated()]
    merged_df = pd.concat([mace_min_time_data,behavioral_data,mace_history_data],axis=1)
    merged_df.to_pickle(f"data_process/mace_all_data_{misspercent}.pkl")
    print("get_mace_all_data\n",merged_df)
    return merged_df


def mace_all_data_add_cvd_prs(misspercent=0.1):
    all_data_path = f"data_process/mace_all_data_{misspercent}.pkl"
    cvd_prs_path = "basedata/PRS_participant.csv"
    mace_all_data = pd.read_pickle(all_data_path)
    cvd_prs_data = pd.read_csv(cvd_prs_path)

    cvd_prs_data.rename(columns={"eid": "Participant ID"}, inplace=True)
    #去除重复的列Participant ID
    mace_all_data = mace_all_data.iloc[:, ~mace_all_data.columns.duplicated()]
    mace_all_data = mace_all_data.merge(cvd_prs_data[['Participant ID', 'CVD_PRS']], on='Participant ID', how='left')
    mace_all_data.to_pickle(f"data_process/mace_all_data_{misspercent}.pkl")
    print("mace_all_data_add_cvd_prs\n",mace_all_data)
    return mace_all_data

def mace_all_data_add_base_info(misspercent=0.1):
    all_data_path = f"data_process/mace_all_data_{misspercent}.pkl"
    base_info_path = "basedata/基线资料.csv"
    mace_all_data = pd.read_pickle(all_data_path)
    base_data = pd.read_csv(base_info_path)

    base_data.rename(columns={"eid": "Participant ID"}, inplace=True)
    merged_df = pd.merge(mace_all_data, base_data, on='Participant ID')
    #去除重复的列Participant ID
    merged_df = merged_df.iloc[:, ~merged_df.columns.duplicated()]
    merged_df.to_pickle(f"data_process/mace_all_data_{misspercent}.pkl")
    print("mace_all_data_add_base_info\n",merged_df)
    return merged_df

def delete_part_crowds(misspercent=0.1,min_time=365):
    all_data_path = f"data_process/mace_all_data_{misspercent}.pkl"
    mace_all_data = pd.read_pickle(all_data_path)
    all_num = mace_all_data.shape[0]
    print("all_num:",all_num)
    #去除min_time 小于0的人或者是小于365的人
    mace_all_data = mace_all_data.drop(mace_all_data[mace_all_data.MIN_TIME < min_time].index)
    min_num = all_num - mace_all_data.shape[0]
    print("delete min num:",min_num)

    #去除已经得过MACE的人
    old_num = mace_all_data.shape[0]
    mace_all_data = mace_all_data.drop(mace_all_data[mace_all_data.MACE_HISTORY == 1].index)
    history_num = old_num - mace_all_data.shape[0]
    print("delete history num:",history_num)

    #去除数据NA率大于10%的人
    old_num = mace_all_data.shape[0]
    mace_all_data = mace_all_data.drop(mace_all_data[mace_all_data.MISS_FLAG == 1].index)
    NA_num = old_num - mace_all_data.shape[0]
    print("delete NA_%_num:",NA_num)

    #去除这些列，后面不需要了
    mace_all_data = mace_all_data.drop(columns=['id','eid', '卒中_', '心衰_', 'Miocardial_infarction_'])
    
    #去除重复的列Participant ID
    mace_all_data = mace_all_data.iloc[:, ~mace_all_data.columns.duplicated()]   

    #为了R语言，不能有空格
    #mace_all_data.rename(columns={"Participant ID": "ParticipantID"}, inplace=True)
    mace_all_data.to_pickle(f"data_process/mace_all_data_{misspercent}_delete_part_crowds.pkl")
    print("delete_part_crowds\n",mace_all_data)
    return(all_num,min_num,history_num,NA_num)

def rename_the_all_data(misspercent=0.1):
    all_data_path = f"data_process/mace_all_data_{misspercent}_delete_part_crowds.pkl"
    supple_data=pd.read_excel('basedata/supplementary date 1.xlsx')
    mace_all_data = pd.read_pickle(all_data_path)
    fields = mace_all_data.columns.values
    Bh_Dsc_list = supple_data['Behavior_DESCRIPTION'].values.tolist()
    for col_name in fields:
        if col_name in Bh_Dsc_list:
            Bh_Code = supple_data.loc[supple_data['Behavior_DESCRIPTION'] == col_name, 'Behavior code'].iloc[0]
            #print(Bh_Code, col_name)
            mace_all_data.rename(columns={col_name: Bh_Code}, inplace=True)
        if col_name in chname_dict.keys():
            #print(chname_dict[col_name], col_name)
            mace_all_data.rename(columns={col_name: chname_dict[col_name]}, inplace=True)
    mace_all_data.to_pickle(f"data_process/mace_all_data_{misspercent}_delete_part_crowds.pkl")
    print(mace_all_data)
    return mace_all_data

def get_cvdd_data(misspercent=0.1):
    before_data_path = f"data_process/mace_mace_min_time.csv"
    before_data = pd.read_csv(before_data_path)
    before_counts = before_data['CVDD'].value_counts()
    before_cvdd = before_counts[0]+before_counts[1]

    after_data_path = f"data_process/mace_all_data_{misspercent}_delete_part_crowds.pkl"
    after_data = pd.read_pickle(after_data_path)
    after_counts = after_data['CVDD'].value_counts()
    after_cvdd = after_counts[0]+after_counts[1]
    return(before_counts, before_cvdd, after_counts,after_cvdd)

def print_multiple_interpolation(misspercent=0.1):
    data_path = f"data_process/mace_all_data_{misspercent}_multiple_interpolation.pkl"
    data = pd.read_pickle(data_path)
    print(data)

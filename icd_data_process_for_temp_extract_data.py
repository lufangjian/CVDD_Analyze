import os
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import pickle
from datetime import datetime, timedelta

def df1_merge_df2_without_duplicate_columns(df1, df2, column_to_keep="Participant ID"):
    # 获取重复列名
    duplicate_columns = df1.columns.intersection(df2.columns).tolist()
    # 删除除了 column_to_keep 列之外的其他重复列,也就是说要保留Participant ID
    duplicate_columns.remove(column_to_keep)
    # df2中先删除重复的列
    df2 = df2.drop(columns=duplicate_columns)
    # 执行 merge 操作，只保留一个副本的重复列
    result = pd.merge(df1, df2, on=column_to_keep)
    return result

#定义icd数据获取函数
def select_cols(df, cols):
    selected = []
    for x in df.columns.tolist():
        for y1 in cols:
            if y1 in x:
                selected.append(x)
    return selected

#读取/生成icd10和死亡的pkl文件
def process_icd_data():
    if(os.path.exists('data_target/icd.pkl')):
        icd_data=pd.read_pickle('data_target/icd.pkl')
    else:        
        icd_data=pd.read_csv('basedata/ICD10_and_Death.csv',dtype=object,index_col=0)
        # 选择需要的字段
        fields=['53-0','40000-0','40001-0','40002-0','41270','41280']
        icd_data=icd_data[select_cols(icd_data,fields)]
        # 将主要死因列重命名，与次要死因合并为一般死因字段40002
        icd_data=icd_data.rename(columns={'40001-0.0': '40002-0.0'})

        #转换时间格式
        fields=['53-0','40000','41280']
        icd_data[select_cols(icd_data,fields)]=\
            icd_data[select_cols(icd_data,fields)].\
                apply(pd.to_datetime,format='%Y-%m-%d')
        # 保存
        icd_data.to_pickle('data_target/icd.pkl')  
    #print(icd_data)
    return icd_data

def predict_illness(df, disease_type= 'disease'):
    if disease_type == 'disease':
        name, time = '41270', '41280'
    elif disease_type == 'death':
        name, time = '40002', '40000'
    columns_name = [col for col in df.columns if name in col]
    columns_time = [col for col in df.columns if time in col]
    D_name = df[columns_name]
    D_time = df[columns_time]
    return D_name,D_time

#获取CombinedICD和NEW_DESCRIPTION映射关系表
def get_combinedicd_new_descpt_dict():
    supplementary =pd.read_excel('basedata/supplementary date 1.xlsx')
    result_dict = {}
    for index, row in supplementary.iterrows():
        result_dict[row['Combined ICD code']] = row['NEW_DESCRIPTION']
    # 去重字典，以确保每个键只出现一次
    result_dict = {key: value for key, value in result_dict.items()}
    #print(result_dict.keys())
    return(result_dict)

#获取OriginalICD编码和CombinedICD编码映射关系表
def get_original_icd_code_combined_icd_code():
    supplementary =pd.read_excel('basedata/supplementary date 1.xlsx')
    result_dict = {}
    for index, row in supplementary.iterrows():
        result_dict[row['Original ICD code']] = row['Combined ICD code']
    #去重字典以确保每个键只出现一次
    result_dict = {key: value for key, value in result_dict.items()}
    #print(result_dict)
    return(result_dict)

#获取CombinedICD编码和分类的关系映射表
def get_combined_icd_code_and_class_descpt_dict():
    supplementary =pd.read_excel('basedata/supplementary date 1.xlsx')
    result_dict = {}
    for index, row in supplementary.iterrows():
        result_dict[row['Combined ICD code']] = row['Classification of medical conditions']
    #去重字典以确保每个键只出现一次
    result_dict = {key: value for key, value in result_dict.items()}
    #print(result_dict)
    return(result_dict)

#获取Behavior Code和Behavior_DESCRIPTION映射关系表
def get_behavior_code_and_descpt_dict():
    supplementary =pd.read_excel('basedata/supplementary date 1.xlsx')
    ret_descrip = {}
    ret_class = {}
    for index, row in supplementary.iterrows():
        if pd.notnull(row['Behavior code']):
            ret_descrip[row['Behavior code']] = row['Behavior_DESCRIPTION']
            ret_class[row['Behavior code']] = row['Classification_E']
    # 去重字典，以确保每个键只出现一次
    ret_descrip = {key: value for key, value in ret_descrip.items()}
    ret_class = {key: value for key, value in ret_class.items()}
    return(ret_descrip, ret_class)


#把df中旧的value全部替换成新的combined_icd_code
def value_change_to_combined_icd_code(disease,disease_time):
    result_dict = get_original_icd_code_combined_icd_code()
    #遍历每一行并处理数据
    for index, row in disease.iterrows():
        for col in disease.columns:
            value = row[col]
            if pd.notnull(value):
                if value[:3] in result_dict.keys():
                    #替换原始数据
                    disease.at[index, col] = result_dict[value[:3]]
                else:
                    disease.at[index, col] = np.nan
                    disease_time_col = str(col).replace("41270","41280")
                    disease_time.at[index, disease_time_col] = np.nan
    return(disease, disease_time)

#icd.pkl比基线资料.csv多得多，需要保证和基线资料.csv的ID进行对齐即可
def filter_icd_pkl_data():
    icd_data=pd.read_pickle('data_target/icd.pkl')
    basedata = pd.read_csv("basedata/基线资料.csv")
    #获取pkl文件中的索引列，这是你想要保留的ID
    desired_ids = basedata['Participant ID']
    #使用isin方法过滤出pkl文件中与csv文件匹配的行
    filtered_df_pkl = icd_data[icd_data.index.isin(desired_ids)]
    filtered_df_pkl.to_pickle('data_target/icd.pkl')

def get_disease_and_disease_time_data():
    icd_data=pd.read_pickle('data_target/icd.pkl')
    disease,disease_time = predict_illness(icd_data,disease_type='disease')
    disease,disease_time = value_change_to_combined_icd_code(disease, disease_time)
    disease.to_pickle('data_target/disease_data.pkl')
    disease_time.to_pickle('data_target/disease_time_data.pkl')
    return(disease, disease_time)

def write_occur_join_time(occurrence_time, join_time, end_study_time, df_new, index, row, col, flag="one"):
    if flag == "one":
        se = row[col]
    else:
        se = col  
    if occurrence_time:
        df_new.at[index, f"%s_Occur_Join"%se] = (occurrence_time - join_time).days
    else:
        df_new.at[index, f"%s_Occur_Join"%se] = (end_study_time - join_time).days
    return(df_new)

def write_death_join_time(death_time, join_time, end_study_time, df_new, index, row, col, flag="one"):
    if flag == "one":
        se = row[col]
    else:
        se = col  

    if death_time:
        df_new.at[index, f"%s_Death_Join"%se] = (death_time - join_time).days
    else:
        df_new.at[index, f"%s_Death_Join"%se] = (end_study_time - join_time).days
    return(df_new)

def write_death_occur_time(death_time, occurrence_time, end_study_time, df_new, index, row, col, join_time,flag="one"):
    if flag == "one":
        se = row[col]
    else:
        se = col  
    if death_time and occurrence_time:
        df_new.at[index, f"%s_Death_Occur"%se] = (death_time - occurrence_time).days
    elif end_study_time and occurrence_time:
        df_new.at[index, f"%s_Death_Occur"%se] = (end_study_time - occurrence_time).days
    else:
        #没有发病的人，最后死亡了，但并不是发病死的，是处于其他原因死的，这段逻辑只会在E10=0的时候出现
        df_new.at[index, f"%s_Death_Occur"%se] = (death_time - join_time).days   
    return(df_new)

def create_an_empty_form():
    disease=pd.read_pickle('data_target/disease_data.pkl')
    df_new = pd.DataFrame(index=disease.index)
    # 读取 Excel 文件  
    dfspd = pd.read_excel("basedata/supplementary date 1.xlsx", index_col=0)
    # 将 "Combined ICD code" 列转换为列表  
    combined_icd_codes = []  
    for index, row in dfspd.iterrows():  
        combined_icd_codes.append(row["Combined ICD code"])
    combined_icd_codes = list(set(combined_icd_codes))
    for icd in combined_icd_codes:
        df_new[icd] = np.nan
        df_new[f"%s_Occur_Join"%icd] = np.nan
        df_new[f"%s_Death_Join"%icd] = np.nan
        df_new[f"%s_Death_Occur"%icd] = np.nan
    df_new.to_pickle("data_target/icd_data_and_disease_time_with_join_occur_death.pkl")
    #print(df_new)
    return(df_new)

def get_disease_and_disease_time_with_join_occur_death():
    disease=pd.read_pickle('data_target/disease_data.pkl')
    disease_time = pd.read_pickle('data_target/disease_time_data.pkl')
    #去除整列都是空的列
    disease = disease.dropna(axis=1, how='all')
    disease_time = disease_time.dropna(axis=1, how='all')

    basedata = pd.read_csv("basedata/基线资料.csv")
    deathdata = pd.read_csv("data_process/mace_mace_min_time.csv")
    df_new = create_an_empty_form()

    # 逐个遍历每一行再遍历每一列  
    for index, row in disease.iterrows():  
        for col in disease.columns:  
            #print(f"index {index}, column {col}: {row[col]}") 
            if not pd.isna(row[col]):
                df_new.at[index, row[col]] = 1
                disease_time_col = str(col).replace("41270","41280")
                join_time = basedata.loc[basedata['Participant ID'] == index, 'Date of attending assessment centre | Instance 0'].iloc[0] #加入研究的时间
                join_time = datetime.strptime(join_time, "%Y/%m/%d")
                end_study_time = datetime.strptime("2022/2/2", "%Y/%m/%d") #结束研究的时间
                death_time = deathdata.loc[deathdata['eid'] == index, '随访时间-全因死亡'].iloc[0] #死亡的时间，目前是天数
                death_time = join_time + timedelta(days=int(death_time)) #死亡的时间=加入研究的时间+天数
                occurrence_time = disease_time.at[index, disease_time_col] #发病的时间

                if np.isnan(df_new.at[index, f"%s_Occur_Join"%row[col]]):  
                    df_new = write_occur_join_time(occurrence_time, join_time, end_study_time, df_new, index, row, col)

                if np.isnan(df_new.at[index, f"%s_Death_Join"%row[col]]):    
                    df_new = write_death_join_time(death_time, join_time, end_study_time, df_new, index, row, col)
                    
                if np.isnan(df_new.at[index, f"%s_Death_Occur"%row[col]]):  
                    df_new = write_death_occur_time(death_time, occurrence_time, end_study_time, df_new, index, row, col, join_time)

    df_new.to_pickle("data_target/icd_data_and_disease_time_with_join_occur_death.pkl")
  
def get_disease_and_disease_time_with_join_occur_death2():
    icd_data =pd.read_pickle('./data_target/icd_data_and_disease_time_with_join_occur_death.pkl')
    disease=pd.read_pickle('data_target/disease_data.pkl')
    disease_time = pd.read_pickle('data_target/disease_time_data.pkl')
    #去除整列都是空的列
    disease = disease.dropna(axis=1, how='all')
    disease_time = disease_time.dropna(axis=1, how='all')
    basedata = pd.read_csv("basedata/基线资料.csv")
    deathdata = pd.read_csv("data_process/mace_mace_min_time.csv")

    icd_code = get_original_icd_code_combined_icd_code()
    # 逐个遍历每一行再遍历每一列  
    for index, row in icd_data.iterrows():  

        join_time = basedata.loc[basedata['Participant ID'] == index, 'Date of attending assessment centre | Instance 0'].iloc[0] #加入研究的时间
        join_time = datetime.strptime(join_time, "%Y/%m/%d")
        end_study_time = datetime.strptime("2022/2/2", "%Y/%m/%d") #结束研究的时间
        try:
            death_time = deathdata.loc[deathdata['eid'] == index, '随访时间-全因死亡'].iloc[0] #死亡的时间，目前是天数
            death_time = join_time + timedelta(days=int(death_time)) #死亡的时间=加入研究的时间+天数
        except Exception:
            death_time = None
        occurrence_time = None #发病的时间

        for col in icd_data.columns:  
            #print(f"index {index}, column {col}: {row[col]}") 
            if col in list(icd_code.values()):
                if pd.isna(row[col]):
                    icd_data.at[index, col] = 0

                    if np.isnan(icd_data.at[index, f"%s_Occur_Join"%col]):  
                        icd_data = write_occur_join_time(occurrence_time, join_time, end_study_time, icd_data, index, row, col, flag="two")

                    if np.isnan(icd_data.at[index, f"%s_Death_Join"%col]):   
                        icd_data = write_death_join_time(death_time, join_time, end_study_time, icd_data, index, row, col, flag="two")
                        
                    if np.isnan(icd_data.at[index, f"%s_Death_Occur"%col]):  
                        icd_data = write_death_occur_time(death_time, occurrence_time, end_study_time, icd_data, index, row, col, join_time,flag="two")

                    #print(f"index {index}, column {col}: {row[col]}") 
    icd_data.to_pickle("data_target/icd_data_and_disease_time_with_join_occur_death_all.pkl")
    print("end")


def create_an_empty_form_for_disease_and_time():
    disease=pd.read_pickle('data_target/disease_data.pkl')
    df_new = pd.DataFrame(index=disease.index)
    # 读取 Excel 文件  
    dfspd = pd.read_excel("basedata/supplementary date 1.xlsx", index_col=0)
    # 将 "Combined ICD code" 列转换为列表  
    combined_icd_codes = []  
    for index, row in dfspd.iterrows():  
        combined_icd_codes.append(row["Combined ICD code"])
    combined_icd_codes = list(set(combined_icd_codes))
    for icd in combined_icd_codes:
        df_new[icd] = np.nan
        df_new[f"%s_Occur"%icd] = np.nan
    df_new.to_pickle("data_target/icd_data_and_disease_time_with_occur_time.pkl")
    #print(df_new)
    return(df_new)


def get_disease_and_disease_time_with_occur_time():
    disease=pd.read_pickle('data_target/disease_data.pkl')
    disease_time = pd.read_pickle('data_target/disease_time_data.pkl')
    #去除整列都是空的列
    disease = disease.dropna(axis=1, how='all')
    disease_time = disease_time.dropna(axis=1, how='all')
    df_new = create_an_empty_form_for_disease_and_time()
    print(df_new)

    # 逐个遍历每一行再遍历每一列  
    for index, row in disease.iterrows():  
        for col in disease.columns:  
            #print(f"index {index}, column {col}: {row[col]}") 
            if not pd.isna(row[col]):
                df_new.at[index, row[col]] = 1
                disease_time_col = str(col).replace("41270","41280")
                occurrence_time = disease_time.at[index, disease_time_col] #发病的时间
                
                if pd.isnull(df_new.at[index, f"%s_Occur"%row[col]]):
                    df_new.at[index, f"%s_Occur"%row[col]] = occurrence_time

    df_new.to_pickle("data_target/icd_data_and_disease_time_with_occur_time.pkl")


def non_cancer_illness_deal_with():
    df=pd.read_csv('basedata/non_cancer_illness.csv',index_col=1)
    name = "20002-0."
    columns_name = [col for col in df.columns if  name in col]
    disease = df[columns_name]

    #遍历每一行并处理数据
    for index, row in disease.iterrows(): 

        row_list = row.tolist()
        if 1263 in row_list:  
        #如果包含，把 df[1263] 置为 1  
            disease.at[index, "1263"] = 1
        else:
            disease.at[index, "1263"] = 0

    disease.to_csv('test/test_1263.csv')
    print(disease)


#####################################################################
#############################临时抓取数据#############################
#####################################################################

def get_disease_and_disease_time_data_for_temp_extract_data():
    icd_data=pd.read_pickle('data_target/icd.pkl')
    disease,disease_time = predict_illness(icd_data,disease_type='disease')
    #disease,disease_time = value_change_to_combined_icd_code(disease, disease_time)
    disease.to_pickle('temp_extract_data/disease_data_for_temp_extract_data.pkl')
    disease_time.to_pickle('temp_extract_data/disease_time_data_for_temp_extract_data.pkl')
    print(disease, disease_time)
    return(disease, disease_time)

def create_an_empty_form_for_temp_extract_data():
    B_list = [f"B{i}" for i in range(35, 50)]
    icd_data=pd.read_csv('basedata/ICD10_and_Death.csv')
    df_new = pd.DataFrame(index=icd_data.eid)
    for icd in B_list:
        df_new[icd] = np.nan
        df_new[f"T%s"%icd] = np.nan   #发病时间-加入时间
    df_new.to_pickle("./temp_extract_data/icd_data_and_disease_time_with_join_occur_for_temp_extract_data.pkl")
    return(df_new, B_list)


def get_disease_and_disease_time_with_join_occur_death_for_temp_extract_data():
    disease=pd.read_pickle('temp_extract_data/disease_data_for_temp_extract_data.pkl')
    disease_time = pd.read_pickle('temp_extract_data/disease_time_data_for_temp_extract_data.pkl')
    #去除整列都是空的列
    disease = disease.dropna(axis=1, how='all')
    disease_time = disease_time.dropna(axis=1, how='all')
    basedata = pd.read_csv("basedata/基线资料.csv")
    (df_new, B_list) = create_an_empty_form_for_temp_extract_data()


    print("start")
    for index, row in disease.iterrows():  
        for col in disease.columns:  
            #print(f"index {index}, column {col}: {row[col]}") 
            if not pd.isna(row[col]) and str(row[col])[:3] in B_list:
                B_Disease = str(row[col])[:3]
                print(B_Disease)

                df_new.at[index, B_Disease] = 1
                disease_time_col = str(col).replace("41270","41280")
                join_time = basedata.loc[basedata['Participant ID'] == index, 'Date of attending assessment centre | Instance 0'].iloc[0] #加入研究的时间
                join_time = datetime.strptime(join_time, "%Y/%m/%d")
                end_study_time = datetime.strptime("2022/2/2", "%Y/%m/%d")    #结束研究的时间
                occurrence_time = disease_time.at[index, disease_time_col]    #发病的时间

                if np.isnan(df_new.at[index, f"T%s"%B_Disease]):  
                    if occurrence_time:
                        df_new.at[index, f"T%s"%B_Disease] = (occurrence_time - join_time).days
                    else:
                        df_new.at[index, f"T%s"%B_Disease] = (end_study_time - join_time).days

    df_new.to_pickle("./temp_extract_data/icd_data_and_disease_time_with_join_occur_for_temp_extract_data1.pkl")
    print("end")

def get_disease_and_disease_time_with_join_occur_death_for_temp_extract_data2():
    B_list = [f"B{i}" for i in range(35, 50)]
    TB_list = [f"TB{i}" for i in range(35, 50)]

    icd_data=pd.read_pickle('temp_extract_data/icd_data_and_disease_time_with_join_occur_for_temp_extract_data1.pkl')
    disease=pd.read_pickle('temp_extract_data/disease_data_for_temp_extract_data.pkl')
    disease_time = pd.read_pickle('temp_extract_data/disease_time_data_for_temp_extract_data.pkl')
    #去除整列都是空的列
    disease = disease.dropna(axis=1, how='all')
    disease_time = disease_time.dropna(axis=1, how='all')
    basedata = pd.read_csv("basedata/基线资料.csv")
    #print(basedata)

    print("start")
    for index, row in icd_data.iterrows():  
        for col in icd_data.columns:  
            print(f"index {index}, column {col}: {row[col]}") 
            if str(col)[:3] in B_list:
                B_Disease = str(col)[:3]
                if pd.isna(row[col]):
                    print(B_Disease)
                    try:
                        join_time = basedata.loc[basedata['Participant ID'] == index, 'Date of attending assessment centre | Instance 0'].iloc[0] #加入研究的时间
                        join_time = datetime.strptime(join_time, "%Y/%m/%d")
                    except Exception:
                        join_time = datetime.strptime("2008/2/2", "%Y/%m/%d")    #如果超出了index边界指定一个加入时间
                    end_study_time = datetime.strptime("2022/2/2", "%Y/%m/%d")    #结束研究的时间
                    occurrence_time = None #发病的时间
                    icd_data.at[index, B_Disease] = 0
                    if np.isnan(icd_data.at[index, f"T%s"%B_Disease]): 
                        if occurrence_time:
                            icd_data.at[index, f"T%s"%B_Disease] = (occurrence_time - join_time).days
                        else:
                            icd_data.at[index, f"T%s"%B_Disease] = (end_study_time - join_time).days
    icd_data.to_pickle("./temp_extract_data/icd_data_and_disease_time_with_join_occur_for_temp_extract_data2.pkl")
    print("end")

def add_statistics_and_min_value():
    icd_data=pd.read_pickle('temp_extract_data/icd_data_and_disease_time_with_join_occur_for_temp_extract_data2.pkl')
    B_icd_data = icd_data[[f"B{i}" for i in range(35, 50)]]
    TB_icd_data = icd_data[[f"TB{i}" for i in range(35, 50)]]

    # 新增一列，根据每一行是否有1来赋值
    B_icd_data['B35-49'] = B_icd_data.apply(lambda row: 1 if 1 in row.values else 0, axis=1)
    TB_icd_data['TB35-49'] = TB_icd_data.apply(lambda row: min(row), axis=1)
    

    # 使用 concat 函数按照索引进行合并
    result = pd.concat([B_icd_data, TB_icd_data],axis=1)
    result.to_csv("./temp_extract_data/icd_data_and_disease_time_with_join_occur_for_temp_extract_data_all.csv")
import pandas as pd
import numpy as np
import math,os
from tqdm.auto import tqdm
from icd_data_process import *
from scipy.stats import binomtest
import re
import statsmodels.api as sm
import numpy as np
import glob
from renamedict import domain_dict

def get_disease_d1d2d3_relation():
    df = pd.read_excel("./binomial/logic_regression_result_for_disease_with_condition.xlsx")
    df[['D1', 'D2']] = df["D1→D2 code"].str.split('→', expand=True)
    #print(df[['D1', 'D2']])
    newlist = []
    for indexa, row in df.iterrows(): 
        d1,d2 = row["D1"], row["D2"]
        for indexb, row in df.iterrows(): 
            d3,d4 = row["D1"], row["D2"]
            if d2 == d3:
                #print(d1, d2, d3, d4)
                newlist.append((d1,d2,d4))
    newlist = list(set(newlist))
    print("disease d1d2d3 relation:",len(newlist))
    return newlist

def get_domain_d1d2d3_relation():
    domain = pd.read_excel("./binomial/logic_regression_result_for_domain_with_condition.xlsx")
    disease = pd.read_excel("./binomial/logic_regression_result_for_disease_with_condition.xlsx")
    domain[['DM', 'D']] = domain["Domain→Disease code"].str.split('→', expand=True)
    disease[['D1', 'D2']] = disease["D1→D2 code"].str.split('→', expand=True)

    newlist = []
    for indexa, row_d in domain.iterrows(): 
        dm, d= row_d["DM"], row_d["D"]
        for indexb, row_dm in disease.iterrows(): 
            d1, d2= row_dm["D1"], row_dm["D2"]
            if d == d1:
                #print(dm, d, d2)
                newlist.append((dm, d, d2))
    newlist = list(set(newlist))
    print("domain d1d2d3 relation:",len(newlist))
    return newlist

def dichotomy_for_domain():
    #对domain进行二分法
    df_data = pd.read_pickle("./binomial/fox_binomial_data_with_fillna0.pkl")
    #print(df_data[["Drinking","S00","I25"]])
    domainlist = ["Drinking","Diet","Living environment","Obesity","Physical exercise","Sleep","Smoking"]
    for domain in domainlist:
        # 计算均值
        mean_value = df_data[domain].mean()
        # 将大于均值的值设为1，小于均值的值设为0
        df_data[domain] = (df_data[domain] > mean_value).astype(int)
    df_data = df_data.rename(columns=domain_dict)#去掉空格
    df_data.to_pickle("./binomial/fox_binomial_data_with_fillna0_and_dichotomy.pkl")

    
def fox_binomial_data_fillna0():
    df_data = pd.read_pickle("./binomial/fox_binomial_data.pkl")
    df_data = df_data.fillna(0)
    df_data.to_pickle(f"./binomial/fox_binomial_data_with_fillna0.pkl")
    print(df_data.isnull().sum())  

def get_or_value_and_ci(effect,stderror):
    effect = float(effect)
    stderror = float(stderror)
    # 计算 Odds Ratio
    odds_ratio = np.exp(effect)
    # 获取系数的标准误差
    standard_errors = stderror
    # 计算 95% 置信区间
    lower_ci = np.exp(effect - 1.96 * standard_errors)
    upper_ci = np.exp(effect + 1.96 * standard_errors)
    return(odds_ratio, lower_ci, upper_ci)

def deal_with_mediate_txt_file(txtpath, tofile, flag = "disease"):
    file_paths = glob.glob(txtpath)
    icd_dscp = get_combinedicd_new_descpt_dict()
    df_rlt = pd.DataFrame()
    for file_path in file_paths:
        #print(file_path)
        with open(file_path, 'r') as file:
            content = file.read()

        #print(content)
        # 使用正则表达式提取数据
        if flag ==  "disease":
            pattern = r'(\w+\s+\w+)\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)\s+([\d.e\-<>]+.*)' 
        elif flag == "domain":
            pattern = r'(\w+\s+\w+)\s+(-?[\d.e-]+)\s+(-?[\d.e-]+)\s+(-?[\d.e-]+)\s+(<\s*[\d.e-]+|\*{3}|\d+\.\d+e[-+]\d+|\-?\d+\.\d+|\d+)' 

        matches = re.findall(pattern, content)
        # 将提取的数据放入DataFrame
        df = pd.DataFrame(matches, columns=['Effect', 'Estimate', 'Std. Error', 'z value', 'Pr(>|z|)'])
        
        # 去掉 Pr(>|z|) 列中的 ***
        df['Pr(>|z|)'] = df['Pr(>|z|)'].str.replace('\*+', '', regex=True)
        filename = os.path.basename(file_path)
        total_eff = df.loc[df['Effect'] == 'total effect', 'Estimate'].values[0]
        total_eff_err = df.loc[df['Effect'] == 'total effect', 'Std. Error'].values[0]
        total_or, total_lci, total_uci = get_or_value_and_ci(total_eff,total_eff_err)

        ind_eff = df.loc[df['Effect'] == 'indirect effect', 'Estimate'].values[0]
        ind_eff_err = df.loc[df['Effect'] == 'indirect effect', 'Std. Error'].values[0]
        int_or, int_lci, int_uci = get_or_value_and_ci(ind_eff,ind_eff_err)

        filename = filename.split(".")[0]
        pecent = float(ind_eff)/float(total_eff) * 100
        pvalue_str = df.loc[df['Effect'] == 'indirect effect', 'Pr(>|z|)'].values[0]
        pvalue_str = pvalue_str.replace('<', '').replace(' ', '')
        pvalue_str = pvalue_str.strip('.')

        pvalue = float(pvalue_str)
        if total_or < 1 or total_lci < 1 or int_or < 1 or int_lci < 1 or pecent < 0 or pecent > 100 or pvalue > 0.05:
            continue

        d1 = filename.split("-")[0]
        d2 = filename.split("-")[1]
        d3 = filename.split("-")[2]

        if flag == "disease":
            row_data = {'D1→D2→D3': '%s→%s→%s'%(d1,d2,d3),        
            'D1 description': icd_dscp[d1],
            "D2 description": icd_dscp[d2],
            "D3 description": icd_dscp[d3],
            "Overall effect OR (95% CI)": "%.2f(%.2f-%.2f)" % (total_or, total_lci, total_uci),
            "Indirect effect OR (95% CI)": "%.2f(%.2f-%.2f)" % (int_or, int_lci, int_uci),
            "Percent Mediation": "%.2f" % float(pecent), 
            "Pvalue Mediation": pvalue_str,
            }

        elif flag == "domain":
            row_data = {'DM→D1→D2': '%s→%s→%s'%(d1,d2,d3),        
            "D1 description": icd_dscp[d2],
            "D2 description": icd_dscp[d3],
            "Overall effect OR (95% CI)": "%.2f(%.2f-%.2f)" % (total_or, total_lci, total_uci),
            "Indirect effect OR (95% CI)": "%.2f(%.2f-%.2f)" % (int_or, int_lci, int_uci),
            "Percent Mediation": "%.2f" % float(pecent), 
            "Pvalue Mediation": pvalue_str,
            }
        
        df_rlt = pd.concat([df_rlt, pd.DataFrame([row_data])], ignore_index=True)
    df_rlt.to_excel(tofile,index=False)

def deal_with_mediate_txt_file_for_disease():
    txtpath = './mediation/mediate_deal_with_for_disease/*.txt'
    tofile = "./mediation/mediate_deal_with_for_disease.xlsx"
    deal_with_mediate_txt_file(txtpath, tofile, flag="disease")

def deal_with_mediate_txt_file_for_domain():
    txtpath = './mediation/mediate_deal_with_for_domain/*.txt'
    tofile = "./mediation/mediate_deal_with_for_domain.xlsx"
    deal_with_mediate_txt_file(txtpath, tofile, flag="domain")


def add_level_for_disease():
    df = pd.read_excel("./binomial/logic_regression_result_for_disease_with_condition.xlsx")
    df[['D1', 'D2']] = df["D1→D2 code"].str.split('→', expand=True)
    disease_data = pd.read_pickle("./binomial/fox_binomial_data_with_fillna0.pkl")

    # 遍历每一行
    for index, row in df.iterrows():
        #print(f"Index: {index}, D1: {row['D1']}, D2: {row['D2']}")
        D1,D2 = row['D1'], row['D2']
        D1Time = disease_data[D1+"_Occur_Join"].sum()
        D2Time = disease_data[D2+"_Occur_Join"].sum()
        if D1Time > D2Time:
            #print(D1Time, type(D1Time), D2Time, type(D2Time))
            df = df.drop(index)

    # 将D1和D2的数据整合成一个列表
    combined_list = df[['D1', 'D2']].values.flatten().tolist()
    combined_list = list(set(combined_list))

    leveldict = {}
    for value in combined_list:
        leveldict[value] = None
    parentlist = []
    n = 1
    filtered_df = df

    while(1):
        if not filtered_df.empty:
            # 筛选出D2中不包含D1的D1数据
            level1df = filtered_df[~filtered_df['D1'].isin(filtered_df['D2'])]
            #level1df = df[~df['D1'].isin(df['D2'])]
            level1list = list(set(level1df["D1"].tolist()))
            level2list = list(set(level1df["D2"].tolist()))
            #print(level2list)

            for value in level1list:
                if value not in parentlist:
                    leveldict[value] = n
            n = n + 1
            parentlist = parentlist + level1list
            # 使用isin方法筛选,把子类的作为父类筛选出来
            filtered_df = df[df['D1'].isin(level2list)]
            # 使用isin方法筛选去掉父类的行
            filtered_df = filtered_df[~filtered_df['D2'].isin(parentlist)]
        else:
            for value in combined_list:
                if value not in parentlist:
                    leveldict[value] = n
            break

    # 添加Level
    df['LevelD1'] = df['D1'].apply(lambda x: leveldict.get(x, 'unknown'))
    df['LevelD2'] = df['D2'].apply(lambda x: leveldict.get(x, 'unknown'))
    #删除多余的列
    df = df.drop(['D1', 'D2'], axis=1)
    df.to_excel("./binomial/logic_regression_result_for_disease_with_condition_and_level.xlsx",index=False)
    return(leveldict)

def add_level_for_mediation_disease():
    leveldict = add_level_for_disease()
    df = pd.read_excel("./mediation/mediate_deal_with_for_disease.xlsx")
    df[['D1', 'D2', 'D3']] = df["D1→D2→D3"].str.split('→', expand=True)
    df['LevelD1'] = df['D1'].apply(lambda x: leveldict.get(x, 'unknown'))
    df['LevelD2'] = df['D2'].apply(lambda x: leveldict.get(x, 'unknown'))
    df['LevelD3'] = df['D3'].apply(lambda x: leveldict.get(x, 'unknown'))

    # 去掉不满足条件的行
    df = df[df['LevelD1'] != 'unknown']

    df['LevelD1'] = df['LevelD1'].astype(int)
    df['LevelD2'] = df['LevelD2'].astype(int)
    df['LevelD3'] = df['LevelD3'].astype(int)

    df = df[(df['LevelD1'] < df['LevelD2']) & (df['LevelD2'] < df['LevelD3'])]
    #删除多余的列
    df = df.drop(['D1', 'D2', 'D3'], axis=1)
    df.to_excel("./mediation/mediate_deal_with_for_disease_with_conditon.xlsx",index=False)


def add_level_for_mediation_domain():
    leveldict = add_level_for_disease()
    df = pd.read_excel("./mediation/mediate_deal_with_for_domain.xlsx")
    df[['DM', 'D1', 'D2']] = df["DM→D1→D2"].str.split('→', expand=True)
    df['LevelDM'] = 0
    df['LevelD1'] = df['D1'].apply(lambda x: leveldict.get(x, 'unknown'))
    df['LevelD2'] = df['D2'].apply(lambda x: leveldict.get(x, 'unknown'))

    # 去掉不满足条件的行
    df = df[df['LevelD2'] != 'unknown']
    df['LevelDM'] = df['LevelDM'].astype(int)
    df['LevelD1'] = df['LevelD1'].astype(int)
    df['LevelD2'] = df['LevelD2'].astype(int)

    df = df[(df['LevelDM'] < df['LevelD1']) & (df['LevelD1'] < df['LevelD2'])]
    #删除多余的列
    df = df.drop(['DM', 'D1', 'D2'], axis=1)
    df.to_excel("./mediation/mediate_deal_with_for_domain_with_conditon.xlsx",index=False)

def add_level_for_binomial_domain():
    leveldict = add_level_for_disease()
    df = pd.read_excel("./binomial/logic_regression_result_for_domain_with_condition.xlsx")
    df[['DM', 'D1']] = df["Domain→Disease code"].str.split('→', expand=True)
    df['LevelDM'] = 0
    df['LevelD1'] = df['D1'].apply(lambda x: leveldict.get(x, 'unknown'))

    # 去掉不满足条件的行
    df = df[df['LevelD1'] != 'unknown']

    #删除多余的列
    df = df.drop(['DM', 'D1'], axis=1)
    df.to_excel("./binomial/logic_regression_result_for_domain_with_condition_and_level.xlsx",index=False)

def get_first_half_and_second_half(input_str):
    substrings = input_str.split("→")
    result = [substrings[i] + "→" + substrings[i+1] for i in range(len(substrings)-1)]
    return(result)

def get_meet_condition_pair_after_mediation():
    pair_list = []
    df_disease = pd.read_excel("./mediation/mediate_deal_with_for_disease_with_conditon.xlsx")
    df_domain = pd.read_excel("./mediation/mediate_deal_with_for_domain_with_conditon.xlsx")
    for index, row in df_disease.iterrows():
        #print(f"Index: {index}, D1: {row['D1']}, D2: {row['D2']}")
        input_str = row["D1→D2→D3"]
        result = get_first_half_and_second_half(input_str)
        pair_list = pair_list + result

        # 遍历每一行
    for index, row in df_domain.iterrows():
        #print(f"Index: {index}, D1: {row['D1']}, D2: {row['D2']}")
        input_str = row["DM→D1→D2"]
        result1 = get_first_half_and_second_half(input_str)
        pair_list = pair_list + result1

    pair_list = list(set(pair_list))
    return(pair_list)


def get_meet_condition_pair_after_mediation_for_disease():
    pair_list = get_meet_condition_pair_after_mediation()
    df_disease = pd.read_excel("./binomial/logic_regression_result_for_disease_with_condition_and_level.xlsx")
    for index, row in df_disease.iterrows():
        d1d2code = row["D1→D2 code"]
        if d1d2code not in pair_list:
            # 删除索引为'Y'的行
            df_disease = df_disease.drop(index)
    df_disease.to_excel("./binomial/logic_regression_result_for_disease_with_condition_and_level_for_drop.xlsx",index=False)

def get_meet_condition_pair_after_mediation_for_domain():
    pair_list = get_meet_condition_pair_after_mediation()
    df_domain = pd.read_excel("./binomial/logic_regression_result_for_domain_with_condition_and_level.xlsx")
    for index, row in df_domain.iterrows():
        d1d2code = row["Domain→Disease code"]
        #保留0-->9
        if row["LevelDM"] == 0 and row["LevelD1"] == 9:
            continue
        if d1d2code not in pair_list:
            # 删除索引为'Y'的行
            df_domain = df_domain.drop(index)
    df_domain.to_excel("./binomial/logic_regression_result_for_domain_with_condition_and_level_for_drop.xlsx",index=False)












import pandas as pd
import numpy as np
import math,os
from tqdm.auto import tqdm
from icd_data_process import *
from scipy.stats import binomtest
#from copy import deepcopy
import warnings
from statsmodels.discrete.conditional_models import ConditionalLogit
from statsmodels.api import Logit
import scipy.stats as stats
import statsmodels.api as sm
from renamedict import domain_dict

#数据预处理
def binomial_pre_process():
    icd_data_occur =pd.read_pickle('./data_target/icd_data_and_disease_time_with_occur_time.pkl')
    icd_data_all =pd.read_pickle('./data_target/icd_data_and_disease_time_with_join_occur_death_all.pkl')
    mace_all_data =pd.read_pickle('./data_process/mace_all_data_0.1_multiple_interpolation.pkl')
    basedata = pd.read_csv("basedata/基线资料.csv")
    merged_df1 = df1_merge_df2_without_duplicate_columns(mace_all_data, basedata)
    #icd_data的index变成一个普通数据
    icd_data1 = icd_data_occur.reset_index() 
    icd_data1 = icd_data1.rename(columns={'eid': 'Participant ID'})
    merged_df2 = df1_merge_df2_without_duplicate_columns(merged_df1, icd_data1)
    #icd_data的index变成一个普通数据
    icd_data2 = icd_data_all.reset_index() 
    icd_data2 = icd_data2.rename(columns={'eid': 'Participant ID'})
    merged_df3 = df1_merge_df2_without_duplicate_columns(merged_df2, icd_data2)
    #去除重复的列
    merged_df3 = merged_df3.iloc[:, ~merged_df3.columns.duplicated()]
    merged_df3 = merged_df3.rename(columns=domain_dict)#去掉空格
    merged_df3.to_pickle("./binomial/fox_binomial_data.pkl")
    print(merged_df3)

binomial_pre_process()

###############################################疾病部分###############################################

def get_cox_data_with_p_meet_conditions_for_disease(diseasecox="./cox/disease_cox_data_all.csv"):
    cox_data = pd.read_csv(diseasecox)
    # 获取OriginalICD编码和CombinedICD编码映射关系表
    result_dict = get_original_icd_code_combined_icd_code()
    newicdlst = result_dict.values()
    newicdlst = list(set(newicdlst))
    # 把Cox Data中的数据进行筛选，Cox完以后需要根据p值进行筛选
    # 如果这里是疾病那么就是0.05/新的疾病分类总数；如果这里是初始因素，那么这里就是0.05/初始因素的个数，初始因素是65个
    cox_data = cox_data[cox_data['p'] < 0.05/len(newicdlst)]
    print("meet_conditions_for_disease:",len(cox_data))
    return(cox_data)


def get_binomtest_cal_data_with_disease(D1, D2, disease_data):
    #print(D1,D2)
    cp_ds_data = disease_data[["CVDD", D1, D2, D1+'_Occur', D2+'_Occur']].copy()
    #过滤出同时患有D1和D2的人,这里返回的是一个Series，bool类型
    indexes=((cp_ds_data[D1]== True) & (cp_ds_data[D2]==True))
    # 将日期列转换为时间差
    cp_ds_data = pd.DataFrame({D1 + '_Occur': pd.to_datetime(cp_ds_data[D1 + '_Occur']),D2 + '_Occur': pd.to_datetime(cp_ds_data[D2 + '_Occur'])})
    # 统计发病时间不同的数量，这里只是不相同的，不包括D1D2同一天发病的
    total = sum((cp_ds_data.loc[indexes, D2 + '_Occur'] - cp_ds_data.loc[indexes, D1 + '_Occur']) != pd.Timedelta(0))
    # 比较时间差并计算d1在前的次数
    d1_first = sum((cp_ds_data.loc[indexes, D1 + '_Occur'] - cp_ds_data.loc[indexes, D2 + '_Occur']) < pd.Timedelta(0))
    # 比较时间差并计算d2在前的次数
    d2_first = sum((cp_ds_data.loc[indexes, D2 + '_Occur'] - cp_ds_data.loc[indexes, D1 + '_Occur']) < pd.Timedelta(0))
    # D1和D2同一天发病的，目前是预留在这里的
    d1_eq_d2 = sum((cp_ds_data.loc[indexes, D2 + '_Occur'] - cp_ds_data.loc[indexes, D1 + '_Occur']) == pd.Timedelta(0))

    del cp_ds_data
    return(indexes, total, d1_first, d2_first,d1_eq_d2)

def get_logic_regression_modle_or_value(model):
    # 获取系数估计值
    coefficients = model.params
    # 计算 Odds Ratio
    odds_ratio = np.exp(coefficients)
    # 获取系数的标准误差
    standard_errors = model.bse
    # 计算 95% 置信区间
    lower_ci = np.exp(coefficients - 1.96 * standard_errors)
    upper_ci = np.exp(coefficients + 1.96 * standard_errors)
    # 打印结果
    for i in range(len(coefficients)):
        print(f"Variable: {coefficients.index[i]}, Odds Ratio: {odds_ratio[i]}, 95% CI: ({lower_ci[i]}, {upper_ci[i]})")

def get_disease_occur_time_d1d2t(cp_ds_data,D1,D2):
    cp_ds_data2 = cp_ds_data[[D1, D2, D1+'_Occur', D2+'_Occur']].copy()
    D1D2cdtion = (cp_ds_data2[D1] == 1) & (cp_ds_data2[D2] == 1)
    D1D2T = cp_ds_data2.loc[D1D2cdtion, D2+"_Occur"] - cp_ds_data2.loc[D1D2cdtion, D1+"_Occur"]
    # 使用布尔索引来筛选出不包含负数的元素
    D1D2T_Dict = {}
    D1D2T = D1D2T[D1D2T > pd.Timedelta('0 days')]
    D1D2T_Dict["D1D2T_SUM"] = D1D2T.dt.days.sum() / 30
    D1D2T_Dict["D1D2T_MD"]  = D1D2T.median() / pd.Timedelta(days=30)
    D1D2T_Dict["D1D2T_Q1"]  = D1D2T.quantile(0.25) / pd.Timedelta(days=30)
    D1D2T_Dict["D1D2T_Q3"]  = D1D2T.quantile(0.75) / pd.Timedelta(days=30)
    del cp_ds_data2
    #print(D1D2T_Dict)
    return D1D2T_Dict

def calculate_confidence_interval_of_single_rate(rate,n):
    ratedict = {}
    # 使用二项分布计算95%置信区间
    alpha = 0.05  # 95%的置信水平
    SE = math.sqrt((rate * (1 - rate)) / n)
    z_critical = stats.norm.ppf(1 - alpha / 2)  # 正态分布的Z值
    margin_of_error = z_critical * SE

    # 计算置信区间的下限和上限
    lower_limit = rate - margin_of_error
    upper_limit = rate + margin_of_error
    ratedict["rate"] = rate
    ratedict["n"] = n #样本量
    ratedict["lower_limit"] = lower_limit
    ratedict["upper_limit"] = upper_limit
    return(ratedict)

def calculate_two_rate_with_confidence_interval(rate1, rate2, n1, n2):
    ratedict = {}
    diff_rate = rate1 - rate2
    SE_diff = math.sqrt((rate1 * (1 - rate1) / n1) + (rate2 * (1 - rate2) / n2))
    alpha = 0.05  # 95%的置信水平
    z_critical = stats.norm.ppf(1 - alpha / 2)  # 正态分布的Z值
    margin_of_error = z_critical * SE_diff
    lower_limit = diff_rate - margin_of_error
    upper_limit = diff_rate + margin_of_error
    ratedict["rate"] = diff_rate
    ratedict["lower_limit"] = lower_limit
    ratedict["upper_limit"] = upper_limit
    return(ratedict)

def get_d1_nd2_rate_per_1000(cp_ds_data,D1,D2):
    cp_ds_data2 = cp_ds_data[[D1, D2, D1+"_Occur_Join", "AllCauseDeath"]].copy()
    D1ND2cdtion = (cp_ds_data2[D1] == 1) & (np.isnan(cp_ds_data2[D2])) 
    D1ND2 = cp_ds_data2.loc[D1ND2cdtion]
    # 去掉随访时间里面为0
    D1ND2 = D1ND2[D1ND2[D1+"_Occur_Join"] > 0] 
    # 将随访时间转换为年
    D1ND2[D1+"_Occur_Join"] = D1ND2[D1+"_Occur_Join"] / 365
    total_deaths = np.sum(D1ND2["AllCauseDeath"])
    # 计算总的随访时间
    total_time = np.sum(D1ND2[D1+"_Occur_Join"])
    # 计算每1000人年的死亡率
    rate = (total_deaths /total_time  )  * 1000  #单位是人年
    rate = rate / 1000  #一个人的概率，必须转化成概率[介于0和1之间]
    D1ND2RDict= calculate_confidence_interval_of_single_rate(rate, len(D1ND2))
    del cp_ds_data2
    return D1ND2RDict

def get_d1_d2_rate_per_1000(cp_ds_data,D1,D2):
    cp_ds_data2 = cp_ds_data[[D1, D2, D2+"_Occur_Join", "AllCauseDeath"]].copy()
    D1D2cdtion = (cp_ds_data2[D1] == 1) & (cp_ds_data2[D2] == 1) 
    D1D2 = cp_ds_data2.loc[D1D2cdtion]
    # 去掉随访时间里面小于0的值
    D1D2 = D1D2[D1D2[D2+"_Occur_Join"] > 0] 
    # 将随访时间转换为年
    D1D2[D2+"_Occur_Join"] = D1D2[D2+"_Occur_Join"] / 365
    total_deaths = np.sum(D1D2["AllCauseDeath"])
    # 计算总的随访时间
    total_time = np.sum(D1D2[D2+"_Occur_Join"])

    # 计算每1000人年的死亡率
    rate = (total_deaths /total_time )  * 1000  #单位是人年
    rate = rate / 1000 #一个人的概率，必须转化成概率[介于0和1之间]
    D1D2RDict= calculate_confidence_interval_of_single_rate(rate, len(D1D2))
    del cp_ds_data2
    return D1D2RDict

def deal_with_logic_regression_for_disease(disease_data, D1, D2):
    #print(disease_data)
    #只获取需要的列，节省计算时间
    cp_ds_data = disease_data[["CVDD", D1, D2, D1+"_Occur",D2+"_Occur", D1+"_Occur_Join", D2+"_Occur_Join", "Male","Age","education","white","TownsendDeprivationIndexAtRecruitment","AllCauseDeath"]].copy()
    cp_ds_data['D1D2'] = (cp_ds_data[D1] == 1) & (cp_ds_data[D2] == 1)
    cp_ds_data['D1D2'] = cp_ds_data['D1D2'].astype(int)

    D1D2T_Dict = get_disease_occur_time_d1d2t(cp_ds_data,D1,D2)
    D1ND2RDict = get_d1_nd2_rate_per_1000(cp_ds_data,D1,D2)
    D1D2RDict = get_d1_d2_rate_per_1000(cp_ds_data,D1,D2)
    D1D2DetaDict = calculate_two_rate_with_confidence_interval(D1D2RDict["rate"], D1ND2RDict["rate"], D1D2RDict["n"], D1ND2RDict["n"])

    #endog 是因变量（也被称为响应变量/结果变量），而 exog 是自变量（也被称为解释变量）,D1D2作为自变量，其他的参数作为了校正参数
    logit_model = Logit(endog=cp_ds_data["CVDD"], exog=cp_ds_data[['D1D2', "Male", "Age", "education", "white","TownsendDeprivationIndexAtRecruitment"]])
    #进行逻辑回归
    model = logit_model.fit(disp=0)
    #print(model.summary())
    interval=np.exp(model.conf_int(alpha=0.05).loc['D1D2'])
    OR=np.exp(model.params['D1D2'])
    p_value=model.pvalues['D1D2'] 
    del cp_ds_data
    return(OR, interval, p_value,D1D2T_Dict, D1ND2RDict, D1D2RDict, D1D2DetaDict)
    
def new_row_data_for_disease_logic_regresion_rlt(D1,D2,icd_dscp,first,OR, interval, d1_eq_d2, total, p_value,D1D2T_Dict, D1ND2RDict, D1D2RDict, D1D2DetaDict):
    row_data = {'D1→D2 code': '%s→%s'%(D1,D2), 
        'D1 description': icd_dscp[D1], 
        'D2 description': icd_dscp[D2],
        "No.#": first,
        "OR (95% CI)": "%.2f(%.2f-%.2f)" % (OR, interval[0], interval[1]),
        "Percentage": (d1_eq_d2 / (total + d1_eq_d2)) * 100,
        "pvalue":p_value,
        "D1D2T(95% CI)":"%.2f(%.2f-%.2f)" % (D1D2T_Dict["D1D2T_MD"], D1D2T_Dict["D1D2T_Q1"], D1D2T_Dict["D1D2T_Q3"]), #单位月
        "D1ND2R(95% CI)":"%.2f(%.2f-%.2f)" % (D1ND2RDict["rate"]*1000, D1ND2RDict["lower_limit"]*1000, D1ND2RDict["upper_limit"]*1000),#单位人年
        "D1D2R(95% CI)":"%.2f(%.2f-%.2f)" % (D1D2RDict["rate"]*1000, D1D2RDict["lower_limit"]*1000, D1D2RDict["upper_limit"]*1000), #单位人年
        "D1D2DetaR(95% CI)":"%.2f(%.2f-%.2f)" % (D1D2DetaDict["rate"]*1000, D1D2DetaDict["lower_limit"]*1000, D1D2DetaDict["upper_limit"]*1000),#单位人年
        }
    return row_data

# 定义疾病对计算方法，返回二项分布检验后满足条件带顺序的疾病对
def disease_pairs_for_disease():
    cox_data = get_cox_data_with_p_meet_conditions_for_disease()
    icd_dscp = get_combinedicd_new_descpt_dict()
    disease_id_list = cox_data["covariate"].to_list()
    disease_data = pd.read_pickle("./binomial/fox_binomial_data.pkl")

    bino_count = 0
    logic_count= 0
    df_rlt = pd.DataFrame(columns=['D1→D2 code', 'D1 description', 'D2 description', "No.#", "OR (95% CI)", "Percentage","pvalue"])

    n = len(disease_data) * 0.0025

    for D1 in disease_id_list:
        for D2 in disease_id_list:
            if D2 != D1:
                indexes, total, d1_first, d2_first,d1_eq_d2 = get_binomtest_cal_data_with_disease(D1, D2, disease_data)
                # 首先这里需要先满足PPT中所定义的（0.25%），然后类似于抛硬币d1_first / total > 0.5
                if d1_first > n and (d1_first / total) > 0.5 :
                    bino=binomtest(d1_first,total,p=0.5, alternative='greater')
                    bino_count = bino_count + 1

                    if(bino.pvalue < 0.05):
                        OR, interval, p_value,D1D2T_Dict, D1ND2RDict, D1D2RDict, D1D2DetaDict = deal_with_logic_regression_for_disease(disease_data, D1, D2)
                        row_data = new_row_data_for_disease_logic_regresion_rlt(D1,D2,icd_dscp,d1_first,OR, interval, d1_eq_d2, total, p_value,D1D2T_Dict, D1ND2RDict, D1D2RDict, D1D2DetaDict)
                        df_rlt = pd.concat([df_rlt, pd.DataFrame([row_data])], ignore_index=True)
                        logic_count = logic_count + 1

                elif d2_first > n and (d2_first / total) > 0.5:
                    bino=binomtest(d2_first,total,p=0.5, alternative='greater')
                    bino_count = bino_count + 1
                    
                    if(bino.pvalue < 0.05):
                        OR, interval, p_value,D1D2T_Dict, D1ND2RDict, D1D2RDict, D1D2DetaDict = deal_with_logic_regression_for_disease(disease_data, D2, D1)
                        row_data = new_row_data_for_disease_logic_regresion_rlt(D2,D1,icd_dscp,d2_first,OR, interval, d1_eq_d2, total, p_value,D1D2T_Dict,D1ND2RDict, D1D2RDict, D1D2DetaDict)
                        df_rlt = pd.concat([df_rlt, pd.DataFrame([row_data])], ignore_index=True)
                        logic_count = logic_count + 1

    #对行数据进行去重
    df_rlt = df_rlt.drop_duplicates(subset=['D1→D2 code'])
    df_rlt.to_excel("./binomial/logic_regression_result_for_disease.xlsx",index=False)
    print("bino_count:" , bino_count/2)
    print("logic_count:", logic_count/2)


###############################################初始因素###############################################

def get_cox_data_with_p_meet_conditions_for_initial_factors(cox_data="./cox/initial_factors_cox_data_with_part_inverted01_all.csv"):
    cox_data = pd.read_csv(cox_data)
    # 把Cox Data中的数据进行筛选，Cox完以后需要根据p值进行筛选
    # 如果这里是疾病那么就是0.05/新的疾病分类总数；如果这里是初始因素，那么这里就是0.05/初始因素的个数，初始因素是65个
    cox_data = cox_data[cox_data['p'] < 0.05/len(cox_data)]
    print("meet_conditions_for_initial_factors:",len(cox_data))
    return(cox_data)


def get_binomtest_cal_data_with_initial_factors(I, D, disease_data):
    #print(I,D)
    cp_ds_data = disease_data[["CVDD", I, D, "DateOfAttendingAssessmentCentreInstance0", D+'_Occur']].copy()
    #过滤出同时有行为数据I（初始因素）和患病D的人,这里返回的是一个Series，bool类型
    indexes=((cp_ds_data[I]== True) & (cp_ds_data[D]==True))

    # 将日期列转换为时间差
    cp_ds_data = pd.DataFrame({"Join": pd.to_datetime(cp_ds_data["DateOfAttendingAssessmentCentreInstance0"]),D+ '_Occur': pd.to_datetime(cp_ds_data[D+'_Occur'])})
    # 统计发病时间不同的数量，这里只是不相同的，不包括初始因素和疾病同一天发生的
    total = sum((cp_ds_data.loc[indexes, "Join"] - cp_ds_data.loc[indexes, D+'_Occur']) != pd.Timedelta(0))
    # 比较时间差并计算初始因素在前的次数
    I_first = sum((cp_ds_data.loc[indexes, "Join"] - cp_ds_data.loc[indexes, D+'_Occur']) < pd.Timedelta(0))
    # 比较时间差并计算疾病在前的次数
    D_first = sum((cp_ds_data.loc[indexes, D+'_Occur'] - cp_ds_data.loc[indexes, "Join"]) < pd.Timedelta(0))
    # 初始因素和疾病同一天发生的
    I_eq_D = sum((cp_ds_data.loc[indexes, "Join"] - cp_ds_data.loc[indexes, D+'_Occur']) == pd.Timedelta(0))
    del cp_ds_data
    return(indexes, total, I_first, D_first, I_eq_D)


def deal_with_logic_regression_for_initial_factors(disease_data, I, D):
    cp_ds_data = disease_data[["CVDD", I, D,"Male","Age","education","white","TownsendDeprivationIndexAtRecruitment"]].copy()
    cp_ds_data['I-D'] = (cp_ds_data[I] == 1) & (cp_ds_data[D] == 1)
    cp_ds_data['I-D'] = cp_ds_data['I-D'].astype(int)

    #endog 是因变量（也被称为响应变量/结果变量），而 exog 是自变量（也被称为解释变量）
    logit_model = Logit(endog=cp_ds_data["CVDD"], exog=cp_ds_data[['I-D', "Male","Age","education","white","TownsendDeprivationIndexAtRecruitment" ]]) 
    model = logit_model.fit(disp=0)
    #print(model.summary())

    interval=np.exp(model.conf_int(alpha=0.05).loc['I-D'])
    OR=np.exp(model.params['I-D'])
    p_value=model.pvalues['I-D']  

    del cp_ds_data
    return(OR, interval, p_value)

def new_row_data_for_initial_factors_logic_regresion_rlt(I,D,icd_dscp,first,OR, interval, I_eq_D, total, pvalue ,initial_data):
    row_data = {'I→D code': '%s→%s'%(I,D),  
        'I description': initial_data[initial_data['covariate'] == I]['description'].values[0],
        'class': initial_data[initial_data['covariate'] == I]['class'].values[0],
        'D description': icd_dscp[D],
        "No.#": first,
        "OR (95% CI)": "%.3f(%.3f-%.3f)" % (OR, interval[0], interval[1]),
        "Percentage": (I_eq_D / (total + I_eq_D)) * 100,
        "pvalue":pvalue
        }
    return row_data
    
# 定义疾病对计算方法，返回二项分布检验后满足条件带顺序的疾病对
def disease_pairs_for_initial_factors():
    disease_data = get_cox_data_with_p_meet_conditions_for_disease()
    initial_data = get_cox_data_with_p_meet_conditions_for_initial_factors()

    icd_dscp = get_combinedicd_new_descpt_dict()
    initial_id_list = initial_data["covariate"].to_list()
    disease_id_list = disease_data["covariate"].to_list()
    disease_data = pd.read_pickle("./binomial/fox_binomial_data.pkl")
    #print(disease_data)

    bino_count = 0
    logic_count= 0
    df_rlt = pd.DataFrame(columns=['I→D code', 'I description','class', 'D description', "No.#", "OR (95% CI)", "Percentage","pvalue"])

    for I in initial_id_list:
        for D in disease_id_list:
            indexes, total, I_first, D_first, I_eq_D = get_binomtest_cal_data_with_initial_factors(I, D, disease_data)
            # 首先这里需要先满足PPT中所定义的千分之2.5，然后类似于抛硬币I_first / total > 0.5
            if I_first > (len(disease_data) * 0.0025)  and (I_first / total) > 0.5 :
                bino=binomtest(I_first,total,p=0.5, alternative='greater')
                bino_count = bino_count + 1
                if(bino.pvalue < 0.05):
                    OR, interval, p_value = deal_with_logic_regression_for_initial_factors(disease_data, I, D)
                    row_data = new_row_data_for_initial_factors_logic_regresion_rlt(I,D,icd_dscp,I_first,OR, interval, I_eq_D, total, p_value ,initial_data)
                    df_rlt = pd.concat([df_rlt, pd.DataFrame([row_data])], ignore_index=True)
                    logic_count = logic_count + 1
                    
    #对行数据进行去重
    df_rlt = df_rlt.drop_duplicates(subset=['I→D code'])
    df_rlt.to_excel("./binomial/logic_regression_result_for_initial_factors.xlsx",index=False)
    print("bino_count:" , bino_count/2)
    print("logic_count:", logic_count/2)


###############################################Domain###############################################

def accord_init_factor_change_to_domain():
    # 读取数据  
    data = pd.read_csv("./cox/initial_factors_cox_data_with_part_inverted01_all.csv")
    disease_data = pd.read_pickle("./binomial/fox_binomial_data.pkl")
    #把分类所对应的初始因素，进行转换，预测概率填写到原始数据中
    for col_class in data['class'].unique():  
        data_class = data[data["class"]==col_class]
        data_class = data_class[data_class["p"] < 0.01]
        init_factor_lst = data_class['covariate'].unique().tolist()
        x = disease_data[init_factor_lst]   #自变量
        y = disease_data["CVDD"] #因变量
        # 添加截距项  
        x = sm.add_constant(x)
        # 拟合逻辑回归模型  
        model = sm.Logit(y, x)  
        result = model.fit()
        # 计算预测概率  
        predictions = result.predict(x)
        # 将预测概率添加到原始数据中  
        disease_data[col_class] = predictions
    disease_data.to_pickle("./binomial/fox_binomial_data.pkl")

 #domain通过中位数方法变成1和0
def domain_to_1and0_with_median():
    cox_data = pd.read_pickle("./binomial/fox_binomial_data.pkl")
    ret_descrip, ret_class = get_behavior_code_and_descpt_dict()
    class_list = ret_class.values()
    class_list = list(set(class_list))
    print(class_list)
    for cls in class_list:
        # 判断factor列是否只包含0和1, 如果不是才需要进行中位数计算
        if  not (cox_data[cls].unique().size == 2):
            # 计算中位数
            median = np.median(cox_data[cls])
            # 将数据与中位数进行比较，如果大于中位数则设置为1，否则设置为0
            cox_data[cls] = np.where(cox_data[cls] > median, 1, 0)
    cox_data.to_pickle("./binomial/fox_binomial_data_with_domain.pkl")
    return(class_list, cox_data)

 
def domain_add_occur_join_time():
    domain_data = pd.read_pickle("./binomial/fox_binomial_data_with_domain.pkl")
    # 逐个遍历每一行再遍历每一列  
    print(domain_data)
    end_study_time = datetime.strptime("2022/2/2", "%Y/%m/%d") #结束研究的时间
    for index, row in domain_data.iterrows():  
        join_time = row['Date of attending assessment centre | Instance 0'] #加入研究的时间
        join_time = datetime.strptime(join_time, "%Y/%m/%d")
        if row["AllCauseDeath"] == 0:
            domain_data.at[index, "Domain_Occur_Join"] = (end_study_time - join_time).days 
        elif row["AllCauseDeath"] == 1:
            death_time = row['FollowUpAllCauseDeath']#死亡的时间，目前是天数
            death_time = join_time + timedelta(days=int(death_time)) #死亡的时间=加入研究的时间+天数
            domain_data.at[index, "Domain_Occur_Join"] = (death_time - join_time).days
    domain_data.to_pickle("./binomial/fox_binomial_data_with_domain.pkl")


def get_binomtest_cal_data_with_domain(domain, D, domain_data):
    #print(domain,D)
    cp_dm_data = domain_data[["CVDD", domain, D, "DateOfAttendingAssessmentCentreInstance0", D+'_Occur']].copy()
    #过滤出同时有行为数据Domain（初始因素class）和患病D的人,这里返回的是一个Series，bool类型
    indexes=((cp_dm_data[domain]== True) & (cp_dm_data[D]==True))
    # 将日期列转换为时间差
    cp_dm_data = pd.DataFrame({"Join": pd.to_datetime(cp_dm_data["DateOfAttendingAssessmentCentreInstance0"]),D+ '_Occur': pd.to_datetime(cp_dm_data[D+'_Occur'])})
    # 统计发病时间不同的数量，这里只是不相同的，不包括初始因素和疾病同一天发生的
    total = sum((cp_dm_data.loc[indexes, "Join"] - cp_dm_data.loc[indexes, D+'_Occur']) != pd.Timedelta(0))
    # 比较时间差并计算初始因素在前的次数
    domain_first = sum((cp_dm_data.loc[indexes, "Join"] - cp_dm_data.loc[indexes, D+'_Occur']) < pd.Timedelta(0))
    # 比较时间差并计算疾病在前的次数
    D_first = sum((cp_dm_data.loc[indexes, D+'_Occur'] - cp_dm_data.loc[indexes, "Join"]) < pd.Timedelta(0))
    # 初始因素和疾病同一天发生的
    dm_eq_d = sum((cp_dm_data.loc[indexes, "Join"] - cp_dm_data.loc[indexes, D+'_Occur']) == pd.Timedelta(0))
    del cp_dm_data
    return(indexes, total, domain_first, D_first, dm_eq_d)


def get_disease_occur_time_dmd1t(cp_dm_data,domain,D):
    cp_dm_data2 = cp_dm_data[[domain, D, D+'_Occur_Join']].copy()
    DMD1cdtion = (cp_dm_data2[domain] == 1) & (cp_dm_data2[D] == 1)
    DMD1T = cp_dm_data2.loc[DMD1cdtion, D+'_Occur_Join']
    print(DMD1T)
    # 使用布尔索引来筛选出不包含负数的元素
    DMD1T_Dict = {}
    DMD1T = DMD1T[DMD1T > 0]
    DMD1T_Dict["DMD1T_SUM"] = DMD1T / 30
    DMD1T_Dict["DMD1T_MD"]  = DMD1T.median() / 30
    DMD1T_Dict["DMD1T_Q1"]  = DMD1T.quantile(0.25) / 30
    DMD1T_Dict["DMD1T_Q3"]  = DMD1T.quantile(0.75) / 30
    del cp_dm_data2
    return DMD1T_Dict

def get_dm_nd1_rate_per_1000(cp_dm_data,domain,D):
    cp_dm_data = cp_dm_data[[domain, D, "Domain_Occur_Join", "AllCauseDeath"]].copy()
    DMND1cdtion = (cp_dm_data[domain] == 1) & (np.isnan(cp_dm_data[D])) 
    DMND1 = cp_dm_data.loc[DMND1cdtion]
    # 去掉随访时间里面小于0
    DMND1 = DMND1[DMND1["Domain_Occur_Join"] > 0] 
    # 将随访时间转换为年
    DMND1["Domain_Occur_Join"] = DMND1["Domain_Occur_Join"] / 365
    total_deaths = np.sum(DMND1["AllCauseDeath"])
    # 计算总的随访时间
    total_time = np.sum(DMND1["Domain_Occur_Join"])
    # 计算每1000人年的死亡率
    rate = (total_deaths /total_time  )  * 1000  #单位是人年
    rate = rate / 1000  #一个人的概率，必须转化成概率[介于0和1之间]
    D1ND2RDict= calculate_confidence_interval_of_single_rate(rate, len(DMND1))
    del cp_dm_data
    return D1ND2RDict

def get_dm_d1_rate_per_1000(cp_dm_data,domain,D):
    cp_dm_data2 = cp_dm_data[[domain, D, D+"_Occur_Join", "AllCauseDeath"]].copy()
    DMD1cdtion = (cp_dm_data2[domain] == 1) & (cp_dm_data2[D] == 1) 
    DMD1 = cp_dm_data2.loc[DMD1cdtion]
    # 去掉随访时间里面小于0的值
    DMD1 = DMD1[DMD1[D+"_Occur_Join"] > 0] 
    # 将随访时间转换为年
    DMD1[D+"_Occur_Join"] = DMD1[D+"_Occur_Join"] / 365
    total_deaths = np.sum(DMD1["AllCauseDeath"])
    # 计算总的随访时间
    total_time = np.sum(DMD1[D+"_Occur_Join"])

    # 计算每1000人年的死亡率
    rate = (total_deaths /total_time )  * 1000  #单位是人年
    rate = rate / 1000 #一个人的概率，必须转化成概率[介于0和1之间]
    DMD1RDict= calculate_confidence_interval_of_single_rate(rate, len(DMD1))
    del cp_dm_data2
    return DMD1RDict
   
def deal_with_logic_regression_for_domain(domain_data, domain, D):
    cp_dm_data = domain_data[["CVDD", domain ,D, D+"_Occur_Join", D+'_Occur', "Domain_Occur_Join", "Male","Age","education","white","TownsendDeprivationIndexAtRecruitment","AllCauseDeath"]].copy()
    #endog 是因变量（也被称为响应变量/结果变量），而 exog 是自变量（也被称为解释变量）,domain作为自变量，其他的参数作为了校正参数
    cp_dm_data['DM-D'] = (cp_dm_data[domain] == 1) & (cp_dm_data[D] == 1)
    cp_dm_data['DM-D'] = cp_dm_data['DM-D'].astype(int)

    DMD1T_Dict = get_disease_occur_time_dmd1t(cp_dm_data,domain,D)
    DMND1RDict = get_dm_nd1_rate_per_1000(cp_dm_data,domain,D)
    DMD1RDict = get_dm_d1_rate_per_1000(cp_dm_data,domain,D)
    DMD1DetaDict = calculate_two_rate_with_confidence_interval(DMD1RDict["rate"], DMND1RDict["rate"], DMD1RDict["n"], DMND1RDict["n"])

    logit_model = Logit(endog=cp_dm_data["CVDD"], exog=cp_dm_data[["DM-D", "Male", "Age", "education", "white","TownsendDeprivationIndexAtRecruitment"]])
    #进行逻辑回归
    model = logit_model.fit(disp=0)
    #print(model.summary())
    interval=np.exp(model.conf_int(alpha=0.05).loc["DM-D"])
    OR=np.exp(model.params["DM-D"])
    p_value=model.pvalues["DM-D"] 
    del cp_dm_data
    return(OR, interval, p_value, DMD1T_Dict, DMND1RDict, DMD1RDict,DMD1DetaDict)

def new_row_data_for_initial_factors_domain_rlt(domain,D,icd_dscp,domain_first,OR, interval, dm_eq_d, total, p_value , DMD1T_Dict, DMND1RDict, DMD1RDict,DMD1DetaDict):
    row_data = {'Domain→Disease code': '%s→%s'%(domain,D),  
        'D description': icd_dscp[D],
        "No.#": domain_first,
        "OR (95% CI)": "%.2f(%.2f-%.2f)" % (OR, interval[0], interval[1]),
        "Percentage": (dm_eq_d / (total + dm_eq_d)) * 100,
        "pvalue":p_value,
        "DMD1T(95% CI)":"%.2f(%.2f-%.2f)" % (DMD1T_Dict["DMD1T_MD"], DMD1T_Dict["DMD1T_Q1"], DMD1T_Dict["DMD1T_Q3"]), #单位月
        "DMND1R(95% CI)":"%.2f(%.2f-%.2f)" % (DMND1RDict["rate"]*1000, DMND1RDict["lower_limit"]*1000, DMND1RDict["upper_limit"]*1000),#单位人年
        "DMD1R(95% CI)":"%.2f(%.2f-%.2f)" % (DMD1RDict["rate"]*1000, DMD1RDict["lower_limit"]*1000, DMD1RDict["upper_limit"]*1000), #单位人年
        "DMD1DetaR(95% CI)":"%.2f(%.2f-%.2f)" % (DMD1DetaDict["rate"]*1000, DMD1DetaDict["lower_limit"]*1000, DMD1DetaDict["upper_limit"]*1000),#单位人年
        }
    return row_data


# 定义Domain对疾病的计算方法
def disease_pairs_for_domains():
    disease_data = get_cox_data_with_p_meet_conditions_for_disease()
    icd_dscp = get_combinedicd_new_descpt_dict()
    disease_id_list = disease_data["covariate"].to_list()

    domain_data = pd.read_pickle("./binomial/fox_binomial_data_with_domain.pkl")
    domain_data = domain_data.rename(columns=domain_dict)#去掉空格

    ret_descrip, ret_class = get_behavior_code_and_descpt_dict()
    domain_clslst = ret_class.values()
    domain_clslst = list(set(domain_clslst))

    bino_count = 0
    logic_count= 0
    df_rlt = pd.DataFrame()

    for domain in domain_clslst:
        for D in disease_id_list:
            indexes, total, domain_first, D_first, dm_eq_d = get_binomtest_cal_data_with_domain(domain, D, domain_data)
            # 首先这里需要先满足PPT中所定义的千分之2.5，然后类似于抛硬币domain_first / total > 0.5
            if domain_first > (len(domain_data) * 0.0025)  and (domain_first / total) > 0.5 :
                bino=binomtest(domain_first,total,p=0.5, alternative='greater')
                bino_count = bino_count + 1
                if(bino.pvalue < 0.05):
                    OR, interval, p_value, DMD1T_Dict, DMND1RDict, DMD1RDict,DMD1DetaDict = deal_with_logic_regression_for_domain(domain_data, domain, D)
                    row_data = new_row_data_for_initial_factors_domain_rlt(domain,D,icd_dscp,domain_first,OR, interval, dm_eq_d, total, p_value , DMD1T_Dict, DMND1RDict, DMD1RDict,DMD1DetaDict)
                    df_rlt = pd.concat([df_rlt, pd.DataFrame([row_data])], ignore_index=True)
                    logic_count = logic_count + 1
                    
    
    df_rlt.to_excel("./binomial/logic_regression_result_for_domain.xlsx",index=False)
    print("bino_count:" , bino_count)
    print("logic_count:", logic_count)


def filter_meet_cditions(df, orv = 1, n=120):
    # 概率需要小于0.05/120
    fdf = df[df['pvalue'] < 0.05/n]
    # OR值大于1
    fdf['start_value'] = fdf['OR (95% CI)'].str.extract(r'^([\d.]+)')
    fdf['start_value'] = pd.to_numeric(fdf['start_value'], errors='coerce')
    fdf2 = fdf[fdf['start_value'] > orv]
    return fdf2

def filter_meet_cditions_for_domains():
    df = pd.read_excel("./binomial/logic_regression_result_for_domain.xlsx")
    fdf2 = filter_meet_cditions(df, orv = 1, n=120)
    fdf2.to_excel("./binomial/logic_regression_result_for_domain_with_condition.xlsx")

def filter_meet_cditions_for_disease():
    df = pd.read_excel("./binomial/logic_regression_result_for_disease.xlsx")
    fdf2 = filter_meet_cditions(df, orv = 1, n=120)
    fdf2.to_excel("./binomial/logic_regression_result_for_disease_with_condition.xlsx")




















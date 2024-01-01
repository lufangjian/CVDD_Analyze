import numpy as np
from datetime import timedelta
import pandas as pd 
from tqdm.auto import tqdm
from icd_data_process import *
from lifelines import CoxPHFitter
from binomial_logicregresion import *
from renamedict import domain_dict


#Cox回归前，数据需要预处理
def cox_data_pre_process():
    icd_data =pd.read_pickle('./data_target/icd_data_and_disease_time_with_join_occur_death_all.pkl')
    mace_all_data =pd.read_pickle('./data_process/mace_all_data_0.1_multiple_interpolation.pkl')
    basedata = pd.read_csv("basedata/基线资料.csv")

    merged_df1 = df1_merge_df2_without_duplicate_columns(mace_all_data, basedata) 
    #icd_data的index变成一个普通数据
    icd_data1 = icd_data.reset_index() 
    icd_data1 = icd_data1.rename(columns={'eid': 'Participant ID'})
    merged_df2 = df1_merge_df2_without_duplicate_columns(merged_df1, icd_data1)
    #去除重复的列
    merged_df2 = merged_df2.iloc[:, ~merged_df2.columns.duplicated()]
    merged_df2.to_pickle("./cox/fox_cox_data.pkl")
    print(merged_df2)

#疾病和CVDD-通过Cox回归
def for_cox_disease(Male=None, Age=None):
    cox_data = pd.read_pickle("./cox/fox_cox_data.pkl")
    result_dict = get_original_icd_code_combined_icd_code()
    descrp_dict = get_combinedicd_new_descpt_dict()
    class_dict = get_combined_icd_code_and_class_descpt_dict()
    new_disease_icd = result_dict.values()
    new_disease_icd = list(set(new_disease_icd))
    all_cox_df = pd.DataFrame()
    for disease in new_disease_icd:
        try:
            cox_data_disease = cox_data[["CVDD",f"{disease}", f"{disease}_Death_Occur","Male","Age","education","white","TownsendDeprivationIndexAtRecruitment"]]

            if Male is not None:
                cox_data_disease = cox_data_disease[cox_data_disease["Male"]==Male]
            if Age is not None:
                if Age >= 60:
                    cox_data_disease = cox_data_disease[cox_data_disease["Age"]>=60]
                elif Age < 60:
                    cox_data_disease = cox_data_disease[cox_data_disease["Age"]<60]

            # 计算疾病在整个人群中的概率   
            frequency_counts = cox_data_disease[disease].value_counts()[1]
            frequency = frequency_counts / len(cox_data_disease)

            if Male is None and Age is None:
            # 这个病人在整个人群中的概率低于千分之五将会被抛掉，抛掉的前提是All这种类型，亚组分析的时候不需要这个条件；
                if frequency < 0.005:
                    continue
                else:
                    pass

            # 创建Cox回归对象
            cph = CoxPHFitter()
            # cox_data_disease中性别、年龄、教育程度、贫穷指数、白皮肤人群作为了校正数据
            df_tmp = pd.DataFrame()
            cox_dfs = cph.fit(cox_data_disease, formula=disease, duration_col=f"{disease}_Death_Occur", event_col="CVDD").summary
            print(cox_dfs)
            cox_dfs["frequency"] = frequency
            cox_dfs["frequency_counts"] = frequency_counts
            cox_dfs["description"] = descrp_dict[disease]
            cox_dfs["class"] = class_dict[disease]
            # 将所有Cox结果合并到一个DataFrame中
            all_cox_df = pd.concat([all_cox_df, cox_dfs])

        except:continue
    print("all cox df:",len(all_cox_df))

    if Male is None and Age is None:
        all_cox_df.to_csv("./cox/cox_disease_to_format/disease_cox_data_all.csv")
    if Male is not None:
        all_cox_df.to_csv(f"./cox/cox_disease_to_format/disease_cox_data_Male_{Male}.csv")
    if Age is not None :
        all_cox_df.to_csv(f"./cox/cox_disease_to_format/disease_cox_data_Age_{Age}.csv")

def get_meet_conditions_for_cox_data(type="disease"):
    if type == "disease":
        all_data=f"./cox/cox_disease_to_format/disease_cox_data_all.csv"
        all_data = get_cox_data_with_p_meet_conditions_for_disease(all_data)
        all_data = all_data[all_data["exp(coef)"] > 1] 
        all_data.to_excel("./cox/cox_disease_to_format/disease_cox_data_all_meet_condition.xlsx")
    elif type == "domain":
        all_data=f"./cox/cox_domain_to_format/domain_cox_data_all.csv"
        all_data = get_cox_data_with_p_meet_conditions_for_disease(all_data)
        all_data = all_data[all_data["exp(coef)"] > 1] 
        all_data.to_excel("./cox/cox_domain_to_format/domain_cox_data_all_meet_condition.xlsx")
    elif type == "init_factors":
        all_data=f"./cox/cox_initial_factors_to_format/initial_factors_cox_data_with_part_inverted01_all.csv"
        all_data = get_cox_data_with_p_meet_conditions_for_initial_factors(all_data)
        all_data = all_data[all_data["exp(coef)"] > 1]
        all_data.to_excel("./cox/cox_initial_factors_to_format/initial_factors_cox_data_with_part_inverted01_all_meet_condition.xlsx")

    covariatelist = all_data["covariate"].to_list()

    diseasecoxtype = ["all","Male_0","Male_1","Age_59","Age_60"]
    for diseasecox in diseasecoxtype:
        if type == "disease":
            dfdata = pd.read_csv(f"./cox/cox_disease_to_format/disease_cox_data_{diseasecox}.csv")
        elif type == "init_factors":
            dfdata = pd.read_csv(f"./cox/cox_initial_factors_to_format/initial_factors_cox_data_with_part_inverted01_{diseasecox}.csv")
        elif type == "domain":
            dfdata = pd.read_csv(f"./cox/cox_domain_to_format/domain_cox_data_{diseasecox}.csv")

        for index, row in dfdata.iterrows():
            cur_covariate = row["covariate"]
            if cur_covariate not in covariatelist:
                dfdata = dfdata.drop(index)

        dfdata["HR (95% CI)"] = dfdata.apply(lambda row: "%.2f(%.2f-%.2f)" % (row["exp(coef)"], row["exp(coef) lower 95%"], row["exp(coef) upper 95%"]), axis=1)
        dfdata = dfdata.rename(columns={'frequency_counts': 'No.#'})
        dfdata = dfdata[["covariate", "description","class", "No.#", "HR (95% CI)"]]
        if type == "disease":
            dfdata.to_excel(f"./cox/cox_disease_to_format/disease_cox_data_{diseasecox}_meet_condition.xlsx")
        elif type == "domain":
            dfdata.to_excel(f"./cox/cox_domain_to_format/domain_cox_data_{diseasecox}_meet_condition.xlsx")
        elif type == "init_factors":    
            dfdata.to_excel(f"./cox/cox_initial_factors_to_format/initial_factors_cox_data_with_part_inverted01_{diseasecox}_meet_condition.xlsx")


def merge_meet_conditions_for_cox_data(type="disease"):
    if type == "disease":
        all_data = pd.read_excel("./cox/cox_disease_to_format/disease_cox_data_all_meet_condition.xlsx",index_col=False)
    elif type == "domain":
        all_data = pd.read_excel("./cox/cox_domain_to_format/domain_cox_data_all_meet_condition.xlsx",index_col=False)
    elif type == "init_factors":    
        all_data = pd.read_excel("./cox/cox_initial_factors_to_format/initial_factors_cox_data_with_part_inverted01_all_meet_condition.xlsx",index_col=False)
    
    Malelist = ["Male_0","Male_1"]
    male_df_data = all_data
    for Maletype in Malelist:
        if type == "disease":
            Maledata = pd.read_excel(f"./cox/cox_disease_to_format/disease_cox_data_{Maletype}_meet_condition.xlsx",index_col=False)
        elif type == "domain":
            Maledata = pd.read_excel(f"./cox/cox_domain_to_format/domain_cox_data_{Maletype}_meet_condition.xlsx",index_col=False)   
        elif type == "init_factors":    
            Maledata = pd.read_excel(f"./cox/cox_initial_factors_to_format/initial_factors_cox_data_with_part_inverted01_{Maletype}_meet_condition.xlsx",index_col=False)

        male_df_data = pd.merge(male_df_data, Maledata, on="covariate", how='left')

    if type == "disease":
        male_df_data.to_excel("./cox/cox_disease_to_format/disease_cox_data_all_meet_condition_for_male.xlsx",index=False)
    elif type == "domain":
        male_df_data.to_excel("./cox/cox_domain_to_format/domain_cox_data_all_meet_condition_for_male.xlsx",index=False)
    elif type == "init_factors": 
        male_df_data.to_excel("./cox/cox_initial_factors_to_format/initial_factors_cox_data_with_part_inverted01_for_male.xlsx",index=False)   

    Agelist = ["Age_59","Age_60"]
    age_df_data = all_data
    for Agetype in Agelist:
        if type == "disease":
            Agedata = pd.read_excel(f"./cox/cox_disease_to_format/disease_cox_data_{Agetype}_meet_condition.xlsx",index_col=False)
        elif type == "domain":
            Agedata = pd.read_excel(f"./cox/cox_domain_to_format/domain_cox_data_{Agetype}_meet_condition.xlsx",index_col=False)   
        elif type == "init_factors": 
            Agedata = pd.read_excel(f"./cox/cox_initial_factors_to_format/initial_factors_cox_data_with_part_inverted01_{Agetype}_meet_condition.xlsx",index_col=False)
        
        age_df_data = pd.merge(age_df_data, Agedata, on="covariate", how='left')
    if type == "disease":
        age_df_data.to_excel("./cox/cox_disease_to_format/disease_cox_data_all_meet_condition_for_age.xlsx",index=False)
    elif type == "domain":
        age_df_data.to_excel("./cox/cox_domain_to_format/domain_cox_data_all_meet_condition_for_age.xlsx",index=False)
    elif type == "init_factors": 
        age_df_data.to_excel("./cox/cox_initial_factors_to_format/initial_factors_cox_data_with_part_inverted01_for_age.xlsx",index=False)


def calculate_cox_data():
    cox_data = pd.read_pickle("./cox/fox_cox_data.pkl")
    print(len(cox_data))
    Male0 = cox_data[cox_data["Male"]==0]
    print("Male0:",len(Male0))
    Male1 = cox_data[cox_data["Male"]==1]
    print("Male1:",len(Male1))
    Age59 = cox_data[cox_data["Age"]<60]
    print("Age<60:",len(Age59))
    Age60 = cox_data[cox_data["Age"]>=60]
    print("Age>=60:",len(Age60))

#初始因素通过中位数方法变成1和0
def initial_factors_to_1and0_with_median():
    cox_data = pd.read_pickle("./cox/fox_cox_data.pkl")
    ret_descrip, ret_class = get_behavior_code_and_descpt_dict()
    initial_factors = ret_descrip.keys()
    initial_factors = list(set(initial_factors))
    for factor in initial_factors:
        # 判断factor列是否只包含0和1, 如果不是才需要进行中位数计算
        if  not (cox_data[factor].unique().size == 2):
            # 计算中位数
            median = np.median(cox_data[factor])
            # 将数据与中位数进行比较，如果大于中位数则设置为1，否则设置为0
            cox_data[factor] = np.where(cox_data[factor] > median, 1, 0)
    cox_data.to_pickle("./cox/fox_cox_data_with_median.pkl")
    print(cox_data)

#初始因素进行Cox回归，时间用的是死亡时间-加入时间
def for_cox_initial_factors():
    cox_data = pd.read_pickle("./cox/fox_cox_data_with_median.pkl")
    ret_descrip, ret_class = get_behavior_code_and_descpt_dict()
    initial_factors = ret_descrip.keys()
    initial_factors = list(set(initial_factors))
    all_cox_df = pd.DataFrame()
    for factor in initial_factors:
        try:
            cox_data_factors = cox_data[["CVDD",f"{factor}", "FollowUpAllCauseDeath","Male","Age","education","white","TownsendDeprivationIndexAtRecruitment"]]
            print(factor)
            # 计算初始因素在整个人群中的概率   
            frequency_counts = cox_data_factors[factor].value_counts()[1]
            frequency = frequency_counts / len(cox_data_factors)
            # 这个初始因素在整个人群中的概率低于千分之五将会被抛掉
            if frequency > 0.005:
                # 创建Cox回归对象
                cph = CoxPHFitter()
                cox_dfs = cph.fit(cox_data_factors, formula=factor, duration_col="FollowUpAllCauseDeath", event_col="CVDD").summary
                cox_dfs["frequency"] = frequency
                cox_dfs["frequency_counts"] = frequency_counts
                cox_dfs["description"] = ret_descrip[factor]
                cox_dfs["class"] = ret_class[factor]
                print(cox_dfs)
                # 将所有Cox结果合并到一个DataFrame中
                all_cox_df = pd.concat([all_cox_df, cox_dfs])
        except:continue
    print(all_cox_df)
    all_cox_df.to_csv("./cox/initial_factors_cox_data.csv")


def get_inverted01_dict():
    cox_data = pd.read_excel("./cox/inverted01_requirement.xlsx")
    # 选择需要翻转数据的df
    cox_data = cox_data[cox_data['inverted01'] == 1]
    result_dict = {}
    for index, row in cox_data.iterrows():
        result_dict[row['covariate']] = row['new description']
    return(result_dict)

#初始因素部分需要进行翻转，然后重新进行初始化因素回归
def for_cox_initial_factors_with_inverted01(Male=None, Age=None):
    cox_data = pd.read_pickle("./cox/fox_cox_data_with_median.pkl")
    ret_descrip, ret_class = get_behavior_code_and_descpt_dict()
    initial_factors = ret_descrip.keys()
    initial_factors = list(set(initial_factors))
    inverted01_dict = get_inverted01_dict()

    all_cox_df = pd.DataFrame()
    for factor in initial_factors:
        try:
            if factor in inverted01_dict.keys():
                #有几个特定的列需要进行翻转
                cox_data[factor] = cox_data[factor].apply(lambda x: 1 - x)

            cox_data_factors = cox_data[["CVDD",f"{factor}", "FollowUpAllCauseDeath","Male","Age","education","white","Townsend deprivation index at recruitment"]]
            if Male is not None:
                cox_data_factors = cox_data_factors[cox_data_factors["Male"]==Male]
            if Age is not None:
                if Age >= 60:
                    cox_data_factors = cox_data_factors[cox_data_factors["Age"]>=60]
                elif Age < 60:
                    cox_data_factors = cox_data_factors[cox_data_factors["Age"]<60]
            
            # 计算初始因素在整个人群中的概率   
            frequency_counts = cox_data_factors[factor].value_counts()[1]
            frequency = frequency_counts / len(cox_data_factors)

            if Male is None and Age is None:
            # 这个病人在整个人群中的概率低于千分之五将会被抛掉，抛掉的前提是All这种类型，亚组分析的时候不需要这个条件；
                if frequency < 0.005:
                    continue
                else:
                    pass

            cph = CoxPHFitter()
            cox_dfs = cph.fit(cox_data_factors, formula=factor, duration_col="FollowUpAllCauseDeath", event_col="CVDD").summary
            cox_dfs["frequency"] = frequency
            cox_dfs["frequency_counts"] = frequency_counts
            
            if factor in inverted01_dict.keys():
                cox_dfs["description"] = inverted01_dict[factor]
            else:
                cox_dfs["description"] = ret_descrip[factor]
            cox_dfs["class"] = ret_class[factor]

            print(cox_dfs)
            # 将所有Cox结果合并到一个DataFrame中
            all_cox_df = pd.concat([all_cox_df, cox_dfs])

        except:continue

    print("all cox df:",len(all_cox_df))

    if Male is None and Age is None:
        all_cox_df.to_csv("./cox/cox_initial_factors_to_format/initial_factors_cox_data_with_part_inverted01_all.csv")
    if Male is not None:
        all_cox_df.to_csv(f"./cox/cox_initial_factors_to_format/initial_factors_cox_data_with_part_inverted01_Male_{Male}.csv")
    if Age is not None :
        all_cox_df.to_csv(f"./cox/cox_initial_factors_to_format/initial_factors_cox_data_with_part_inverted01_Age_{Age}.csv")

def get_min_and_max_value(value):
    #value = 2.58(1.57-4.24)
    value = str(value)
    min = float(value.split("(")[1].split("-")[0])
    max = float(value.split("-")[1].replace(")",""))
    #print(value, min, max)
    return(min, max)

def calculate_overlapping(filepath):
    cox_data = pd.read_excel(filepath)
    # 添加一列overlapping，默认为0
    cox_data['overlapping'] = 0  # 默认为0

    for index, row in cox_data.iterrows():
        if not pd.isna(row['HR (95% CI)_y']) and not pd.isna(row['HR (95% CI)']):
            min1,max1 = get_min_and_max_value(row['HR (95% CI)_y'])
            min2,max2 = get_min_and_max_value(row['HR (95% CI)'])
            print(min1, max1, min2, max2)
            if min2 > max1:
                print("aaaa")
                cox_data.at[index, 'overlapping'] = 0
            elif min1 > max2:
                print("bbbb")
                cox_data.at[index, 'overlapping'] = 0
            else:
                print("cccc")
                cox_data.at[index, 'overlapping'] = 1
    
    cox_data.to_excel(filepath, index=False)

def calculate_overlapping_for_file():
    file_list = [
        "./cox/cox_initial_factors_to_format/initial_factors_cox_data_with_part_inverted01_for_age.xlsx",
        "./cox/cox_initial_factors_to_format/initial_factors_cox_data_with_part_inverted01_for_male.xlsx",
        "./cox/cox_disease_to_format/disease_cox_data_all_meet_condition_for_age.xlsx",
        "./cox/cox_disease_to_format/disease_cox_data_all_meet_condition_for_male.xlsx",
        "./cox/cox_domain_to_format/domain_cox_data_all_meet_condition_for_age.xlsx",
        "./cox/cox_domain_to_format/domain_cox_data_all_meet_condition_for_male.xlsx",
    ]
    for file_path in file_list:
        print(file_path)
        calculate_overlapping(file_path)

calculate_overlapping_for_file()

#疾病提取数据
def extract_cox_data_for_disease():
    cox_data = pd.read_pickle("./cox/fox_cox_data.pkl")
    cox_data_disease = cox_data[["Participant ID","CVDD","Male","Age","education","white","TownsendDeprivationIndexAtRecruitment",
                                 "B49", "B49_Death_Occur","E83","E83_Death_Occur","F05","F05_Death_Occur",
                                 "G45", "G45_Death_Occur","I26","I26_Death_Occur","I48","I48_Death_Occur",
                                 "I50", "I50_Death_Occur","I60","I60_Death_Occur","J44","J44_Death_Occur",
                                 "N08", "N08_Death_Occur",
                                ]]
    cox_data_disease.to_csv("./test/extract_cox_data_for_disease.csv",index=False)


#初始因素提取
def extract_cox_data_for_initial_factors():
    cox_data = pd.read_pickle("./cox/fox_cox_data_with_median.pkl")
    cox_data_factors = cox_data[["Participant ID", "CVDD","Z61", "FollowUpAllCauseDeath","Male","Age","education","white","TownsendDeprivationIndexAtRecruitment"]]
    cox_data_factors.to_csv("./test/extract_cox_data_for_initial_factors.csv",index=False)

#补充DomainCox回归数据
#Domain进行Cox回归，时间用的是死亡时间-加入时间
def for_cox_initial_factors():
    cox_data = pd.read_pickle("./binomial/fox_binomial_data_with_fillna0_and_dichotomy.pkl")
    ret_descrip, ret_class = get_behavior_code_and_descpt_dict()
    domain_clslst = ret_class.values()
    domain_clslst = list(set(domain_clslst))

    #initial_factors = ret_descrip.keys()
    #initial_factors = list(set(initial_factors))
    all_cox_df = pd.DataFrame()
    for factor in domain_clslst:
        try:
            cox_data_factors = cox_data[["CVDD",f"{factor}", "FollowUpAllCauseDeath","Male","Age","education","white","TownsendDeprivationIndexAtRecruitment"]]
            print(factor)
            # 计算初始因素在整个人群中的概率   
            frequency_counts = cox_data_factors[factor].value_counts()[1]
            frequency = frequency_counts / len(cox_data_factors)
            # 这个初始因素在整个人群中的概率低于千分之五将会被抛掉
            if frequency > 0.005:
                # 创建Cox回归对象
                cph = CoxPHFitter()
                cox_dfs = cph.fit(cox_data_factors, formula=factor, duration_col="FollowUpAllCauseDeath", event_col="CVDD").summary
                cox_dfs["frequency"] = frequency
                cox_dfs["frequency_counts"] = frequency_counts
                cox_dfs["description"] = ret_descrip[factor]
                cox_dfs["class"] = ret_class[factor]
                print(cox_dfs)
                # 将所有Cox结果合并到一个DataFrame中
                all_cox_df = pd.concat([all_cox_df, cox_dfs])
        except:continue
    print(all_cox_df)
    all_cox_df.to_csv("./cox/initial_factors_cox_data.csv")


#补充DomainCox回归数据
#Domain进行Cox回归，时间用的是死亡时间-加入时间
def for_cox_domain(Male=None, Age=None):
    cox_data = pd.read_pickle("./binomial/fox_binomial_data_with_fillna0_and_dichotomy.pkl")
    ret_descrip, ret_class = get_behavior_code_and_descpt_dict()
    domain_clslst = ret_class.values()
    domain_clslst = list(set(domain_clslst)) 
    
    cox_data = cox_data.rename(columns=domain_dict)#去掉空格

    all_cox_df = pd.DataFrame()
    for factor in domain_clslst:
        #try:
            factor = factor.replace(" ", "") #去掉空格
            cox_data_factors = cox_data[["CVDD",f"{factor}", "FollowUpAllCauseDeath","Male","Age","education","white","TownsendDeprivationIndexAtRecruitment"]]
            if Male is not None:
                cox_data_factors = cox_data_factors[cox_data_factors["Male"]==Male]
            if Age is not None:
                if Age >= 60:
                    cox_data_factors = cox_data_factors[cox_data_factors["Age"]>=60]
                elif Age < 60:
                    cox_data_factors = cox_data_factors[cox_data_factors["Age"]<60]
            
            # 计算初始因素在整个人群中的概率   
            frequency_counts = cox_data_factors[factor].value_counts()[1]
            frequency = frequency_counts / len(cox_data_factors)

            """
            if Male is None and Age is None:
            # 这个病人在整个人群中的概率低于千分之五将会被抛掉，抛掉的前提是All这种类型，亚组分析的时候不需要这个条件；
                if frequency < 0.005:
                    continue
                else:
                    pass
            """

            cph = CoxPHFitter()
            cox_dfs = cph.fit(cox_data_factors, formula=factor, duration_col="FollowUpAllCauseDeath", event_col="CVDD").summary
            cox_dfs["frequency"] = frequency
            cox_dfs["frequency_counts"] = frequency_counts
            cox_dfs["class"] = factor
            cox_dfs["description"] = factor
            print(cox_dfs)
            # 将所有Cox结果合并到一个DataFrame中
            all_cox_df = pd.concat([all_cox_df, cox_dfs])

        #except:continue
    print("all cox df:",len(all_cox_df))
    if Male is None and Age is None:
        all_cox_df.to_csv("./cox/cox_domain_to_format/domain_cox_data_all.csv")
    if Male is not None:
        all_cox_df.to_csv(f"./cox/cox_domain_to_format/domain_cox_data_Male_{Male}.csv")
    if Age is not None :
        all_cox_df.to_csv(f"./cox/cox_domain_to_format/domain_cox_data_Age_{Age}.csv")

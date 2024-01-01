########################################################
#使用过程中，可能会安装的库
########################################################
#install.packages('mice')
#install.packages('reticulate')
#install.packages('tibble')
#install.packages('lme4')
#install.packages("e1071")
#install.packages("reticulate")
#install.packages("glmnet")  
#install.packages("melt") 
#install.packages("MatchThem") 
#install.packages("devtools")

########################################################
#使用MICE进行多重插补空缺数据
########################################################
library(MatchThem)
library(mice)
library(reticulate)

setwd("D:/03_Project/HX_CVDD/CVDD_Analyze")
pd <- import("pandas")
data <- pd$read_pickle("data_process/mace_all_data_0.1_delete_part_crowds.pkl")

# 遍历每一列
for (col in names(data)) 
{
    print(col)
    # 获取当前列的数据
    column_data <- data[[col]]
    column_data2 <- data[[col]]
    # 获取唯一值，目标是作为判定条件
    unique_values <- unique(column_data2)
    # 去除空值
    unique_values <- unique_values[!is.na(unique_values)]
    # 判断当前列的数据类型
    if (anyNA(column_data)) 
    {
        # 创建包含两列数据的数据帧，其中MACE是用来估计的列，column_data是需要插补的列
        mice_data <- data.frame(MACE = data[["MACE"]], column_data)
        if (all(unique_values %in% c(0, 1))) 
        {
            # 数据列只包含0和1
            mids <- mice(mice_data, m = 5, method = "2l.bin", maxcat = 100)
        }
        else
        {
            # 数据列还包含其他值
           mids <- mice(mice_data, m = 5, method = "norm", maxcat = 100)
        }
        completed_data <- complete(mids)
        data[[col]] <- completed_data[[2]]  # 2表示被插补的列，也就是column_data
    }
    else 
    {
        data[[col]] <- column_data
    }
}
# 保存数据
output_file <- "data_process/mace_all_data_0.1_multiple_interpolation.pkl"
pd$to_pickle(data, output_file)


########################################################
#倾向性匹配-Participants with CVDD
########################################################
library(reticulate)
library(MatchIt)
setwd("D:/03_Project/HX_CVDD/CVDD_Analyze")
pd <- import("pandas")
data_CVDD <- pd$read_pickle("data_process/mace_all_data_0.1_multiple_interpolation.pkl")

data_CVDD$CVDD <- as.logical(data_CVDD$CVDD)
#PSM倾向性匹配
m.out <- matchit(CVDD~Male + Age + TownsendDeprivationIndexAtRecruitment ,data=data_CVDD,m.order = "data",method='nearest',ratio=10,caliper=0.1,discard='both')
plot(m.out,type="jitter",interactive=FALSE)
summary(m.out)
write.csv(match.data(m.out), "r_result/PSM匹配CVDD_0.1.csv",row.names = FALSE)



########################################################
#旧的参考代码-死亡倾向性匹配
########################################################

# 死亡匹配
#data_DEATH <- read.csv("data_target/MAFLD_death.csv",encoding = 'UTF-8')
#data_DEATH$death<- as.logical(data_DEATH$death) 
#PSM倾向性匹配
#m_death.out <- matchit(death~Sex + Age + Townsend.deprivation.index.at.recruitment,data=data_DEATH,m.order = "data",method='nearest',ratio=5,caliper=0.1,discard='control')

#plot(m_death.out,type="jitter",interactive=FALSE)
#summary(m_death.out)
#write.csv(match.data(m_death.out), "r_result/PSM_death.csv",row.names = FALSE)

#library(RItools)
#xBalance(MAFLD~Sex + Age + Townsend.deprivation.index.at.recruitment,data=match.data(m.out),report	= c("chisquare.test"))


########################################################
#倾向性匹配-用于测试
########################################################
library(MatchThem)
library(mice)
library(reticulate)

setwd("D:/03_Project/HX_CVDD/CVDD_Analyze")
pd <- import("pandas")
data <- pd$read_pickle("data_process/mace_all_data_0.1_delete_part_crowds.pkl")

# 遍历每一列
for (col in names(data)) 
{
    print(col)
    # 获取当前列的数据
    column_data <- data[[col]]
    # 获取唯一值，目标是作为判定条件
    unique_values <- unique(column_data)
    # 去除空值
    unique_values <- unique_values[!is.na(unique_values)]
    if (all(unique_values %in% c(0, 1))) 
    {
        # 数据列只包含0和1
        #mice_method <- "mode"
        print("aaaa")
    }
    else
    {
        # 数据列还包含其他值
        #mice_method <- "norm"
        print("bbbb")
    }
}
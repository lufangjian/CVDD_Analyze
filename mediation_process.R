#install.packages("mediation")  
#install.packages("writexl")  
#install.packages("medflex")  
#install.packages("broom")  
#install.packages("doParallel") 

#############################################################
####################使用medflex中介分析疾病###################
#############################################################

library(gtsummary)
library(medflex)
library(writexl)
library(MatchThem)
library(reticulate)
library(MatchIt)
library(doParallel)

cl <- makeCluster(detectCores())
registerDoParallel(cl)

# 读取数据
setwd("D:/03_Project/HX_CVDD/CVDD_Analyze")
source_python("D:/03_Project/HX_CVDD/CVDD_Analyze/mediation_process.py")

pd <- import("pandas")
df <- pd$read_pickle("D:/03_Project/HX_CVDD/CVDD_Analyze/binomial/fox_binomial_data_with_fillna0.pkl")
df[is.na(df)]<- 0 #补充NA数据
df$education <- as.factor(df$education)
# 调用自定义的Python函数
diseaselist <- py$get_disease_d1d2d3_relation()

for (disease in diseaselist) {
    # 访问子列表中的元素
    X <- disease[[1]]
    M <- disease[[2]]
    Y <- disease[[3]]
    cat(X,M,Y,"//")

    df_dealwith <- df[, c(X,M,Y)]  
    result <- tryCatch({

    formula1 <- as.formula(paste(M, "~ factor(", X, ")", sep = ""))
    expData <- neWeight(formula1, data = df_dealwith)
    X0 <- paste(X, "0", sep = "")
    X1 <- paste(X, "1", sep = "")
    formula2 <- as.formula(paste(Y, "~", X0, "+", X1, sep = ""))
    neMod1  <- neModel(formula2, family = binomial("logit"),expData = expData,se = "robust")
    effdecomp <- neEffdecomp(neMod1)
    #print(summary(effdecomp))
    summary_info <- summary(effdecomp)
    # 将结果保存到文本文件中
    filename <- sprintf("mediation/mediate_deal_with_for_disease/%s-%s-%s.txt", X, M, Y)
    capture.output(summary_info, file = filename)

    }, error = function(e) {
    # 异常处理代码,这里的做法是不处理
    return(NULL)
    })

}

stopCluster(cl)



#############################################################
####################使用medflex中介分析domain#################
#############################################################



library(gtsummary)
library(medflex)
library(writexl)
library(MatchThem)
library(reticulate)
library(MatchIt)
library(doParallel)

cl <- makeCluster(detectCores())
registerDoParallel(cl)

# 读取数据
setwd("D:/03_Project/HX_CVDD/CVDD_Analyze")
source_python("D:/03_Project/HX_CVDD/CVDD_Analyze/mediation_process.py")

pd <- import("pandas")
df <- pd$read_pickle("D:/03_Project/HX_CVDD/CVDD_Analyze/binomial/fox_binomial_data_with_fillna0_and_dichotomy.pkl")
df[is.na(df)]<- 0 #补充NA数据
df$education <- as.factor(df$education)
# 调用自定义的Python函数
diseaselist <- py$get_domain_d1d2d3_relation()

for (disease in diseaselist) {
    # 访问子列表中的元素
    X <- disease[[1]]
    M <- disease[[2]]
    Y <- disease[[3]]
    cat(X,M,Y,"//")

    df_dealwith <- df[, c(X,M,Y)]  
    result <- tryCatch({

    formula1 <- as.formula(paste(M, "~ factor(", X, ")", sep = ""))
    expData <- neWeight(formula1, data = df_dealwith)
    X0 <- paste(X, "0", sep = "")
    X1 <- paste(X, "1", sep = "")
    formula2 <- as.formula(paste(Y, "~", X0, "+", X1, sep = ""))
    neMod1  <- neModel(formula2, family = binomial("logit"),expData = expData,se = "robust")
    effdecomp <- neEffdecomp(neMod1)
    #print(summary(effdecomp))
    summary_info <- summary(effdecomp)
    # 将结果保存到文本文件中
    filename <- sprintf("mediation/mediate_deal_with_for_domain/%s-%s-%s.txt", X, M, Y)
    capture.output(summary_info, file = filename)

    }, error = function(e) {
    # 异常处理代码,这里的做法是不处理
    return(NULL)
    })

}

stopCluster(cl)




#############################################################
####################使用medflex中介分析#######################
#############################################################
library(medflex)
library(writexl)
library(MatchThem)
library(reticulate)
library(MatchIt)
library(gtsummary)

# 读取数据
setwd("D:/03_Project/HX_CVDD/CVDD_Analyze")
source_python("D:/03_Project/HX_CVDD/CVDD_Analyze/mediation_process.py")
pd <- import("pandas")
df <- pd$read_pickle("D:/03_Project/HX_CVDD/CVDD_Analyze/binomial/fox_binomial_data.pkl")
df[is.na(df)]<- 0 #补充NA数据，这一步看自我需求 
df$education <- as.factor(df$education)
df_dealwith <- df[, c("W20", "S00", "H25","Male", "Age","education","white")]  
expData <- neWeight(S00 ~ factor(W20) + Male + Age + education + white , data = df_dealwith)
#expData <- neWeight(S00 ~ factor(W20) , data = df_dealwith)
#head(expData, 4)
neMod1 <- neModel(H25 ~ W200 + W201 + Male + Age + education + white, family = binomial("logit"), expData = expData,se = "robust",control = list(tol = 2e-18))
#neMod1 <- neModel(H25 ~ W200 + W201 , family = binomial("logit"), expData = expData,se = "robust",control = list(tol = 2e-18))
effdecomp <- neEffdecomp(neMod1)
# 计算 summary 统计信息
summary_info <- summary(effdecomp)
# 将结果保存到文本文件中
filename <- sprintf("mediation/%s-%s-%s.txt", "W20", "S00", "H25")
capture.output(summary_info, file = filename)



#############################################################
####################使用mediation中介分析#####################
#############################################################
library(mediation)
library(writexl)
library(MatchThem)
library(reticulate)
library(MatchIt)

setwd("D:/03_Project/HX_CVDD/CVDD_Analyze")
pd <- import("pandas")
df <- pd$read_pickle("D:/03_Project/HX_CVDD/CVDD_Analyze/binomial/fox_binomial_data.pkl")
df[is.na(df)]<- 0 #补充NA数据，这一步看自我需求
#X是W20, 是I25, Y是J12（二分类变量）
a <- lm(I25 ~ W20 + Male + Age + education + white,df) #lm(M~X,df)
b <- glm(J12 ~W20+I25+Male + Age + education + white, df,  family = binomial("logit")) #glm(Y~X+M)
set.seed(123) #保证结果可以复现
result = mediate(a,b,treat="W20",mediator = "I25", boot=TRUE, sims=20)#20次抽样
summary(result)


#############################################################
########################使用mediation中介分析#################
#############################################################
library(mediation)
library(writexl)
library(MatchThem)
library(reticulate)
library(MatchIt)

setwd("D:/03_Project/HX_CVDD/CVDD_Analyze")
pd <- import("pandas")
df <- pd$read_excel("D:/03_Project/HX_CVDD/CVDD_Analyze/test/test3.xlsx")
df[is.na(df)]<- 0 #补充NA数据，这一步看自我需求
df$education <- as.factor(df$education)
df$IncomeClass <- as.factor(df$IncomeClass)
#X是kdm_advance, M是CCM, Y是Allduetodeath（二分类变量）
a <- lm(CCM ~ kdm_advance+Sexmale1+Ageatrecruitment+education+IncomeClass,df) #lm(M~X,df)
#a<- glm(CCM ~ kdm_advance+Sexmale1+Ageatrecruitment+education+IncomeClass,df,family = binomial("logit")) #glm(M~X,df)
b <- glm(Allduetodeath ~CCM+kdm_advance+Sexmale1+Ageatrecruitment+education+IncomeClass, df, family = binomial("logit")) #glm(Y~X+M)
set.seed(123) #保证结果可以复现
result = mediate(a,b,treat="kdm_advance",mediator = "CCM", boot=TRUE, sims=100)#100次抽样
summary(result)


#############################################################
####################使用medflex中介分析#######################
#############################################################
library(medflex)
library(writexl)
library(MatchThem)
library(reticulate)
library(MatchIt)

# 读取数据
setwd("D:/03_Project/HX_CVDD/CVDD_Analyze")
df$education <- as.factor(df$education)  
df$IncomeClass <- as.factor(df$IncomeClass) 
AnalysisData <- pd$read_excel("D:/03_Project/HX_CVDD/CVDD_Analyze/test/test3.xlsx")
expData <- neWeight(CCM ~ factor(kdm_advance) + Sexmale1 + Ageatrecruitment + education + IncomeClass, data = AnalysisData)
neMod1 <- neModel(Allduetodeath ~ kdm_advance0 + kdm_advance1 + Sexmale1 + Ageatrecruitment + education + IncomeClass, family = binomial("logit"), expData = expData,se = "robust")
#summary(neMod1)
effdecomp <- neEffdecomp(neMod1)
summary(effdecomp)
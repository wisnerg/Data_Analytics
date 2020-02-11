# Title: Course 3 Task 3.R

# Last update: 2019/09/10

# File: Course 3 Task 3.R
# Project name: Evaluate Techniques for Wifi Locationing


#################
# Project Notes #
#################

# Summarize project: Using wifi signals to predict location of phone inside buildings  

# Summarize top model and/or filtered dataset
# The top model was rf_OOB_CS -- random forest model using out of box data that was centered and scaled.
# The top model2 was rf_OOB2_CS -- random forest model using out of box data that was centered and scaled.

################
# Housekeeping #
################

# Clear objects if necessary
rm(list = ls())

# get working directory
getwd()

# set working directory
setwd()
setwd("C:/Users/gwisner/Desktop/UT_Austin/Course 3/Task 3/")
dir()

################
# Load packages
################
install.packages('arules')
install.packages('arulesViz')
install.packages('caret')
install.packages('corrplot')
install.packages('readr')
install.packages('mlbench')
install.packages('dplyr')
install.packages('caret')
install.packages('tidyr')
install.packages('data.table')
install.packages('doParallel')
install.packages('ggplot2')
install.packages('BBmisc')
library(arules)
library(arulesViz)
library(caret)
library(corrplot)
library(readr)
library(mlbench)
library(dplyr)
library(tidyr)
library(data.table)
library(doParallel)
library(ggplot2)
library(BBmisc)

#######################
# Parallel processing #
#######################

#--- for Win ---#
#nstall.packages("doParallel") # install in 'Load packages' section above
library(doParallel)  # load in the 'Load Packages' section above
detectCores()  # detect number of cores
cl <- makeCluster(7)  # select number of cores
registerDoParallel(cl) # register cluster
getDoParWorkers()  # confirm number of cores being used by RStudio
# Stop Cluster. After performing your tasks, make sure to stop your cluster. 
stopCluster(cl); registerDoSEQ()


###############
# Import data #
###############

#--- Load raw datasets ---#

## Load Train/Existing data (Dataset 1)
wifi_training <- read.csv("trainingData.csv", stringsAsFactors = FALSE, header=T)
class(wifi_training)  # "data.frame"
str(wifi_training) #19937 obs. of  529 variables


## Load Predict/New data (Dataset 2) ---#
wifi_validation <- read.csv("validationData.csv", stringsAsFactors = FALSE, header=T)
class(wifi_validation)  # "data.frame"
str(wifi_validation) #1111 obs. of  529 variables

#--- Load preprocessed datasets that have been saved ---#

#################
# Evaluate data #
#################

#--- Dataset 1 ---#
str(wifi_training)  # 19937 obs. of  529 variable
names(wifi_training)
head(wifi_training)
tail(wifi_training)
summary(wifi_training)
# plot
hist(wifi_training$BUILDINGID)
plot(wifi_training$LONGITUDE, wifi_training$LATITUDE) #Building Views
#qqnorm( # Be familiar with this plot, but don't spend a lot of time on it
# check for missing values 
anyNA(wifi_training) #FALSE
is.na(wifi_training)

# remove or exclude missing values
#na.omit(DatasetName$ColumnName) # Drops any rows with missing values and omits them forever.
#na.exclude(DatasetName$ColumnName) # Drops any rows with missing values, but keeps track of where they were.
# remove outliers if necessary
#DatasetName <- DatasetName[-c(1,2),] 

#--- Dataset 2 ---#

str(wifi_validation)  #1111 obs. of  529 variables
names(wifi_validation)
head(wifi_validation)
tail(wifi_validation)
summary(wifi_validation)
# plot
hist(wifi_validation$BUILDINGID)
plot(wifi_validation$LONGITUDE, wifi_validation$LATITUDE) #Building Views
#qqnorm( # Be familiar with this plot, but don't spend a lot of time on it
# check for missing values 
anyNA(wifi_validation) #FALSE
is.na(wifi_validation)

# remove or exclude missing values
#na.omit(DatasetName$ColumnName) # Drops any rows with missing values and omits them forever.
#na.exclude(DatasetName$ColumnName) # Drops any rows with missing values, but keeps track of where they were.
# remove outliers if necessary
#DatasetName <- DatasetName[-c(1,2),] 

##############
# Preprocess #
##############

#--- Dataset 1 ---#

# change data types

wifi_training_OOB <- wifi_training %>% 
  mutate_all(funs(as.numeric(as.character(.)))) %>%
  unite("POSITION", BUILDINGID, FLOOR, sep = "_", remove = F) %>%
  mutate_at(vars(POSITION, BUILDINGID, FLOOR, SPACEID, RELATIVEPOSITION), 
            funs(as.factor))

class(wifi_training_OOB$POSITION) #[1] "factor"

unique(wifi_training_OOB$POSITION) #[1] Levels: 0_0 0_1 0_2 0_3 1_0 1_1 1_2 1_3 2_0 2_1 2_2 2_3 2_4

# WAPXXX: Intensity value --Negative integer values from -104 (Weakest) to 0 (Strongest) and +100. Positive value 100 used if WAP001 was not detected.
# Longitude. Negative real values from -7695.9387549299299000 to -7299.786516730871000 
# Latitude. Positive real values from 4864745.7450159714 to 4865017.3646842018. 
# Floor: Altitude in floors inside the building. Integer values from 0 to 4. 
# BuildingID: ID to identify the building. Measures were taken in three different buildings. Categorical integer values from 0 to 2. 
# SpaceID: Internal ID number to identify the Space (office, corridor, classroom) where the capture was taken. Categorical integer values. 
# RelativePosition: Relative position with respect to the Space (1 - Inside, 2 - Outside in Front of the door). Categorical integer values. 

# rename a column
#names(ds)<-c("ColumnName","ColumnName","ColumnName") 
# handle missing values (if applicable) 
#na.omit(ds$ColumnName)
#na.exclude(ds$ColumnName)        
#ds$ColumnName[is.na(ds$ColumnName)] <- mean(ds$ColumnName,na.rm = TRUE)

#--- Dataset 2 ---#

wifi_validation_OOB <- wifi_validation %>% 
  mutate_all(funs(as.numeric(as.character(.)))) %>%
  unite("POSITION", BUILDINGID, FLOOR, sep = "_", remove = F) %>%
  mutate_at(vars(POSITION, BUILDINGID, FLOOR, SPACEID, RELATIVEPOSITION), 
            funs(as.factor))

class(wifi_validation_OOB$POSITION) #[1] "factor"

unique(wifi_validation_OOB$POSITION) #[1] Levels: 0_0 0_1 0_2 0_3 1_0 1_1 1_2 1_3 2_0 2_1 2_2 2_3 2_4
                                      
#Is the position variable the same in both Validation and Testing? Yes
##[1] Levels: 0_0 0_1 0_2 0_3 1_0 1_1 1_2 1_3 2_0 2_1 2_2 2_3 2_4
##[1] Levels: 0_0 0_1 0_2 0_3 1_0 1_1 1_2 1_3 2_0 2_1 2_2 2_3 2_4

#################
# Feature removal
#################

#function to convert 100s(no signal) to -100
convert <- function(x) {if(any(x == 100)) { x <- -105} else {x <- x}}

#--- Dataset 1 ---#

#creating out of box training dataset
trainset_OOB <- wifi_training_OOB %>% 
  distinct() %>% #removing duplicate rows
  mutate_at(vars(starts_with("WAP")), funs(sapply(.,convert))) #converting '100s' -- no signal to -105

str(trainset_OOB) # 'data.frame':	19300 obs. of  530 variables:

#create summary statistics table for training dataset
stats_trainset <- trainset_OOB %>% 
  select(starts_with("WAP")) %>% 
  summarise_each(list(mean = mean,
                      sd = sd,
                      min = min,
                      max = max)) %>% 
  gather(stat, val) %>%
  separate(stat, into = c("var", "stat"), sep = "_") %>%
  spread(stat, val) %>%
  select(var, min, max, mean, sd)

head(stats_trainset)
# var  min  max      mean        sd
# 1 WAP001 -105  -93 -104.9910 0.2970075
# 2 WAP002 -105  -86 -104.9832 0.5400930
# 3 WAP003 -105 -105 -105.0000 0.0000000
# 4 WAP004 -105 -105 -105.0000 0.0000000
# 5 WAP005 -105  -89 -104.9741 0.5755119
# 6 WAP006 -105  -58 -104.6927 2.632271

# remove ID and obvious features

### Unnecessary variables in Building / Floor predictions
not_used <- c("LONGITUDE", "LATITUDE", "FLOOR", "BUILDINGID", "SPACEID", "RELATIVEPOSITION", "USERID",
              "PHONEID", "TIMESTAMP")

## Create OOB for Building / Floor predictions
trainset_OOB <- trainset_OOB[,!names(trainset_OOB) %in% c(not_used)]

str(trainset_OOB) # 'data.frame':	19300 obs. of  521 variables:

#create new dataset where mean = -105 (no responses from wifi at any location in dataset)
no_signal <- stats_trainset[stats_trainset$mean == -105,1]

#trainset Remove No Signal Building / Floor predictions
trainset_R_NS <- trainset_OOB[,!names(trainset_OOB) %in% c(no_signal)]

str(trainset_R_NS) # 'data.frame':	19300 obs. of  466 variables
head(trainset_R_NS)

#--- Dataset 2 ---#

#creating out of box training dataset
testset_OOB <- wifi_validation_OOB %>% 
  distinct() %>% #removing duplicate rows
  mutate_at(vars(starts_with("WAP")), funs(sapply(.,convert))) #converting '100s' -- no signal to -105

str(testset_OOB) # 'data.frame':	1111 obs. of  531 variables:

#create summary statistics table for training dataset
stats_testset <- testset_OOB %>% 
  select(starts_with("WAP")) %>% 
  summarise_each(list(mean = mean,
                      sd = sd,
                      min = min,
                      max = max)) %>% 
  gather(stat, val) %>%
  separate(stat, into = c("var", "stat"), sep = "_") %>%
  spread(stat, val) %>%
  select(var, min, max, mean, sd)

head(stats_testset)
# var  min  max      mean        sd
# 1 WAP001 -105  -85 -104.8965 1.2480963
# 2 WAP002 -105 -105 -105.0000 0.0000000
# 3 WAP003 -105  -85 -104.9658 0.8068961
# 4 WAP004 -105  -84 -104.9658 0.8102387
# 5 WAP005 -105 -105 -105.0000 0.0000000
# 6 WAP006 -105 -105 -105.0000 0.0000000

# remove ID and obvious features

## Create OOB for Building / Floor predictions
testset_OOB <- testset_OOB[,!names(testset_OOB) %in% c(not_used)]

str(testset_OOB) # 'data.frame':	1111 obs. of  521 variables:

#trainset Remove No Signal Building / Floor predictions
testset_R_NS <- testset_OOB[,!names(testset_OOB) %in% c(no_signal)]

str(testset_R_NS) # 'data.frame':	1111 obs. of  466 variables:
head(testset_R_NS)

#################
# Save datasets #
#################

# after ALL preprocessing, save a new version of the dataset
#write.csv()

##########################
# Feature Selection (FS) #
##########################

# Three primary methods
# 1. Filtering
# 2. Wrapper methods (e.g., RFE caret)
# 3. Embedded methods (e.g., varImp)

############# 
# Filtering #
#############
testcorr <- trainset_R_NS
# good for num/int data 
testcorr[] <- lapply(testcorr, as.integer)
# calculate correlation matrix for all vars
corr_R_NS <- cor(testcorr[,1:466], use = "complete.obs")
# summarize the correlation matrix
corr_R_NS
# plot correlation matrix
corrplot(corr_R_NS)
#corrplot(corr_R_NS, order = "hclust") # sorts based on level of collinearity
corrplot(corr_R_NS, method = "circle")

# find IVs that are highly corrected (ideally >0.90)
highlyCorrelated_R_NS <- findCorrelation(corr_R_NS, cutoff=0.90)
# summarize the correlation matrix
highlyCorrelated_R_NS
# [1]  14  22  24  26  28  30  38  40  42  46  48  52  56  66  70  74  79  87  89  91  98 100 102 108
# [25] 110 112 114 120 124 131 133 135 137 143 145 149 157 159 161 163 167 169 171 173 175 177 179 182
# [49] 194 229 236 282 283 285 302 318 335 339 369 444 445 459 460   9  11  17  31  33  35  43  49  92
# [73]  95 103 105 115 117 121 125 128 140 151 154 164 198 233 237 235 238 249 263 257 259 261 286 298
# [97] 306 308 312

# get var name of high corr
HighCorr <- colnames(testcorr[c(highlyCorrelated_R_NS)]) 
HighCorr
# [1] "WAP016" "WAP024" "WAP026" "WAP028" "WAP030" "WAP032" "WAP040" "WAP042" "WAP044" "WAP048"
# [11] "WAP050" "WAP054" "WAP058" "WAP068" "WAP072" "WAP076" "WAP081" "WAP089" "WAP091" "WAP097"
# [21] "WAP104" "WAP106" "WAP108" "WAP114" "WAP116" "WAP118" "WAP120" "WAP126" "WAP130" "WAP137"
# [31] "WAP139" "WAP141" "WAP143" "WAP149" "WAP151" "WAP156" "WAP167" "WAP169" "WAP171" "WAP173"
# [41] "WAP177" "WAP179" "WAP181" "WAP183" "WAP185" "WAP187" "WAP189" "WAP192" "WAP204" "WAP253"
# [51] "WAP261" "WAP313" "WAP314" "WAP316" "WAP334" "WAP351" "WAP371" "WAP375" "WAP405" "WAP498"
# [61] "WAP499" "WAP513" "WAP514" "WAP011" "WAP013" "WAP019" "WAP033" "WAP035" "WAP037" "WAP045"
# [71] "WAP051" "WAP098" "WAP101" "WAP109" "WAP111" "WAP121" "WAP123" "WAP127" "WAP131" "WAP134"
# [81] "WAP146" "WAP161" "WAP164" "WAP174" "WAP208" "WAP258" "WAP262" "WAP260" "WAP263" "WAP274"
# [91] "WAP288" "WAP282" "WAP284" "WAP286" "WAP317" "WAP329" "WAP338" "WAP340" "WAP344"

trainset_R_Cor <- trainset_R_NS[,!names(trainset_R_NS) %in% c(HighCorr)]
testset_R_Cor <- testset_R_NS[,!names(testset_R_NS) %in% c(HighCorr)]

str(trainset_R_Cor) #'data.frame':	19300 obs. of  367 variables:
str(testset_R_Cor)  #data.frame':	1111 obs. of  367 variables:

############
# Sampling #
############

# ---- Sampling ---- #

set.seed(123) # set random seed
trainset_OOB_5k <- trainset_OOB[sample(1:nrow(trainset_OOB), 5000, replace=FALSE),]
nrow(trainset_OOB_5k)
head(trainset_OOB_5k)

set.seed(123) # set random seed
trainset_R_NS_5k <- trainset_R_NS[sample(1:nrow(trainset_R_NS), 5000, replace=FALSE),]
nrow(trainset_R_NS_5k)
head(trainset_R_NS_5k)

set.seed(123) # set random seed
trainset_R_Cor_5k <- trainset_R_Cor[sample(1:nrow(trainset_R_Cor), 5000, replace=FALSE),]
nrow(trainset_R_Cor_5k)
head(trainset_R_Cor_5k)

# create the training partition that is 75% of total obs
set.seed(123) # set random seed
inTraining_OOB <- createDataPartition(trainset_OOB_5k$POSITION, p=0.75, list=FALSE)

# create training/testing dataset
ts.trainSet_OOB <- trainset_OOB_5k[inTraining_OOB,]   
ts.testSet_OOB <- trainset_OOB_5k[-inTraining_OOB,]  

# verify number of obs 
str(ts.trainSet_OOB) # 'data.frame':	3755 obs. of  521 variables:
str(ts.testSet_OOB) # 'data.frame':	1245 obs. of  521 variables:

# sample again after removing any features

# create the training partition that is 75% of total obs
set.seed(123) # set random seed
inTraining_R_NS <- createDataPartition(trainset_R_NS_5k$POSITION, p=0.75, list=FALSE)

# create training/testing dataset
ts.trainSet_R_NS <- trainset_R_NS_5k[inTraining_R_NS,]   
ts.testSet_R_NS <- trainset_R_NS_5k[-inTraining_R_NS,] 

# verify number of obs 
str(ts.trainSet_R_NS) # 'data.frame':	3755 obs. of  466 variables:
str(ts.testSet_R_NS) # 'data.frame':	1245 obs. of  466 variables:

# sample again after removing any features

# create the training partition that is 75% of total obs
set.seed(123) # set random seed
inTraining_R_Cor <- createDataPartition(trainset_R_Cor_5k$POSITION, p=0.75, list=FALSE)

# create training/testing dataset
ts.trainSet_R_Cor <- trainset_R_Cor_5k[inTraining_R_Cor,]   
ts.testSet_R_Cor <- trainset_R_Cor_5k[-inTraining_R_Cor,] 

# verify number of obs 
str(ts.trainSet_R_Cor) # 'data.frame':	3755 obs. of  367 variables:
str(ts.testSet_R_Cor) # 'data.frame':	1245 obs. of  367 variables:

#############
# caret RFE # 
#############

##################################
# Datasets created above to test #
##################################
ts.trainSet_OOB   # 'data.frame':	3755 obs. of  521 variables
ts.trainSet_R_NS  # 'data.frame':	3755 obs. of  466 variables:
ts.trainSet_R_Cor # 'data.frame':	3755 obs. of  367 variables:

#################
# Train control #
#################

# set 10 fold cross validation
fitControl <- trainControl(method = "repeatedcv", 
                           number = 10, 
                           repeats = 1)


#######################################
# ----------- Train model ----------- #
#######################################

##########################
## ------- C5.0 ------- ##
##########################

# --------------- C5.0 train ts.trainset_OOB  --------------- #
set.seed(123)
system.time(C5.0_OOB <- train(POSITION~.,
                  data=ts.trainSet_OOB,
                  method="C5.0",
                  trControl=fitControl))

# user  system elapsed 
# 27.02    0.64  187.67 

C5.0_OOB 
# model  winnow  trials  Accuracy   Kappa 
# tree   FALSE   20      0.9800303  0.9781309

varImp(C5.0_OOB)

# test against test set
C5.0_OOB_pred <- predict(C5.0_OOB, ts.testSet_OOB)
#performace measurment
postResample(C5.0_OOB_pred, ts.testSet_OOB$POSITION)

# Accuracy     Kappa 
# 0.9823293 0.9806504

# --------------- C5.0 train ts.trainset_OOB Center/Scale  --------------- #
set.seed(123)
system.time(C5.0_OOB_CS <- train(POSITION~.,
                              data=ts.trainSet_OOB,
                              method="C5.0",
                              trControl=fitControl,
                              preProc = c("zv", "center", "scale")))
# method = "center" subtracts the mean of the predictor's data from the predictor values 
# method = "scale" divides by the standard deviation.
# method = "zv" excluded zero variance columns

# user  system elapsed 
# 19.90    0.61  156.48 

C5.0_OOB_CS 
# model  winnow  trials  Accuracy   Kappa 
#  tree   FALSE   20      0.9802934  0.9784193

varImp(C5.0_OOB_CS)

# test against test set
C5.0_OOB_CS_pred <- predict(C5.0_OOB_CS, ts.testSet_OOB)
#performace measurment
postResample(C5.0_OOB_CS_pred, ts.testSet_OOB$POSITION)

# Accuracy     Kappa 
# 0.9831325 0.9815294 

# --------------- C5.0 train ts.trainset_R_NS --------------- #
set.seed(123)
system.time(C5.0_R_NS <- train(POSITION~.,
                              data=ts.trainSet_R_NS, 
                              method="C5.0", 
                              trControl=fitControl))

# user  system elapsed 
# 22.50    0.33  149.09

C5.0_R_NS

# model  winnow  trials  Accuracy   Kappa    
# rules  FALSE   20      0.9848183  0.9833742

varImp(C5.0_R_NS)

# test against test set
C5.0_R_NS_pred <- predict(C5.0_R_NS, ts.testSet_R_NS)
#performace measurment
postResample(C5.0_R_NS_pred, ts.testSet_R_NS$POSITION)

# Accuracy     Kappa 
#  0.9815261 0.9797729

# --------------- C5.0 train ts.trainset_R_NS Center/Scale --------------- #
set.seed(123)
system.time(C5.0_R_NS_CS <- train(POSITION~.,
                   data=ts.trainSet_R_NS, 
                   method="C5.0", 
                   trControl=fitControl,
                   preProc = c("zv", "center", "scale")))

# user  system elapsed 
# 19.63    0.38  163.02 

C5.0_R_NS_CS
# model  winnow  trials  Accuracy   Kappa    
# tree   FALSE   20      0.9802934  0.9784193

varImp(C5.0_R_NS_CS)

# test against test set
C5.0_R_NS_CS_pred <- predict(C5.0_R_NS_CS, ts.testSet_R_NS)
#performace measurment
postResample(C5.0_R_NS_CS_pred, ts.testSet_R_NS$POSITION)

# Accuracy     Kappa 
#  0.9831325 0.9815294 

# --------------- C5.0 train ts.trainset_R_Cor --------------- #
set.seed(123)
system.time(C5.0_R_Cor <- train(POSITION~.,
                               data=ts.trainSet_R_Cor, 
                               method="C5.0", 
                               trControl=fitControl))

# user  system elapsed 
# 12.09    0.25  101.75 

C5.0_R_Cor  
# model  winnow  trials  Accuracy   Kappa    
# rules  FALSE   20      0.9800332  0.9781390

varImp(C5.0_R_Cor)

# test against test set
C5.0_R_Cor_pred <- predict(C5.0_R_Cor , ts.testSet_R_Cor)
#performace measurment
postResample(C5.0_R_Cor_pred, ts.testSet_R_Cor$POSITION)

# Accuracy    Kappa 
# 0.9783133 0.9762586 

# --------------- C5.0 train ts.trainset_R_NS Center/Scale--------------- #
set.seed(123)
system.time(C5.0_R_Cor_CS <- train(POSITION~.,
                                data=ts.trainSet_R_Cor, 
                                method="C5.0", 
                                trControl=fitControl,
                                preProc = c("zv", "center", "scale")))

# user  system elapsed 
# 13.98    0.34  117.06

C5.0_R_Cor_CS  
# model  winnow  trials  Accuracy   Kappa    
# rules  FALSE   20      0.9800324  0.9781354

varImp(C5.0_R_Cor_CS)

# test against test set
C5.0_R_Cor_CS_pred <- predict(C5.0_R_Cor_CS , ts.testSet_R_Cor)
#performace measurment
postResample(C5.0_R_Cor_CS_pred, ts.testSet_R_Cor$POSITION)

# Accuracy    Kappa 
# 0.9839357 0.9824129

# --------------- Comparing C5.0 models --------------- #

set.seed(123)
C5.0ModelFitResults <- resamples(list(C5OOB = C5.0_OOB, C5OOB_CS = C5.0_OOB_CS, 
                                      C5_R_NS = C5.0_R_NS, C5_R_NS_CS = C5.0_R_NS_CS, 
                                      C5_R_Cor = C5.0_R_Cor, C5_R_Cor_CS = C5.0_R_Cor_CS))
# output summary metrics for tuned models 
summary(C5.0ModelFitResults)

# Call:
#   summary.resamples(object = C5.0ModelFitResults)
# 
# Models: C5OOB, C5OOB_CS, C5_R_NS, C5_R_NS_CS, C5_R_Cor, C5_R_Cor_CS 
# Number of resamples: 10 
# 
# Accuracy 
#                 Min.    1st Qu.    Median      Mean   3rd Qu.     Max.    NA's
# C5OOB       0.9706667 0.9759353 0.9787551 0.9800303 0.9839893 0.9893617    0
# C5OOB_CS    0.9706667 0.9765984 0.9800316 0.9802934 0.9833457 0.9893617    0
# C5_R_NS     0.9733333 0.9820212 0.9840849 0.9848183 0.9866843 0.9973333    0
# C5_R_NS_CS  0.9706667 0.9765984 0.9800316 0.9802934 0.9833457 0.9893617    0
# C5_R_Cor    0.9600000 0.9747728 0.9801061 0.9800332 0.9886326 0.9893617    0
# C5_R_Cor_CS 0.9706667 0.9761273 0.9799783 0.9800324 0.9833900 0.9893048    0
# 
# Kappa 
#                 Min.   1st Qu.    Median      Mean   3rd Qu.      Max.    NA's
# C5OOB       0.9678716 0.9736497 0.9767459 0.9781309 0.9824651 0.9883451    0
# C5OOB_CS    0.9678716 0.9743762 0.9781374 0.9784193 0.9817619 0.9883451    0
# C5_R_NS     0.9707876 0.9803097 0.9825784 0.9833742 0.9854177 0.9970797    0
# C5_R_NS_CS  0.9678716 0.9743762 0.9781374 0.9784193 0.9817619 0.9883451    0
# C5_R_Cor    0.9562226 0.9723793 0.9782206 0.9781390 0.9875553 0.9883481    0
# C5_R_Cor_CS 0.9678781 0.9738652 0.9780795 0.9781354 0.9818034 0.9882879    0

#plotting model comparison
bwplot(C5.0ModelFitResults)

#Best model is C5_R_NS

########################
## ------- RF ------- ##
########################

# #manual tuning
#rfGrid <- expand.grid(mtry=c(28, 30, 32, 34, 36))

# --------------- RF train ts.trainset_OOB  --------------- #
set.seed(123)
system.time(rf_OOB <- train(POSITION~., 
                            data = ts.trainSet_OOB, 
                            method="rf", 
                            trControl=fitControl,
                            tuneLength = 5))

# user  system elapsed 
# 78.02    0.32  960.28

rf_OOB
# mtry  Accuracy   Kappa    
#  32   0.9912056  0.9903568

varImp(rf_OOB)

# test against test set
rf_OOB_pred <- predict(rf_OOB, ts.testSet_OOB)
#performace measurment
postResample(rf_OOB_pred, ts.testSet_OOB$POSITION)

# Accuracy     Kappa 
# 0.9935795 0.9929610  

# --------------- RF train ts.trainset_OOB Center/Scale  --------------- #
set.seed(123)
system.time(rf_OOB_CS <- train(POSITION~., 
                            data = ts.trainSet_OOB, 
                            method="rf", 
                            trControl=fitControl,
                            tuneLength = 5,
                            preProc = c("zv", "center", "scale")))

# user  system elapsed 
# 71.35    0.36 1345.57 

rf_OOB_CS
# mtry  Accuracy   Kappa    
#   32   0.9882879  0.9871722

varImp(rf_OOB_CS)

# test against test set
rf_OOB_CS_pred <- predict(rf_OOB_CS, ts.testSet_OOB)
#performace measurment
postResample(rf_OOB_CS_pred, ts.testSet_OOB$POSITION)

# Accuracy     Kappa 
# 0.9887550 0.9876882   

# --------------- RF train ts.trainset_R_NS  --------------- #
set.seed(123)
system.time(rf_R_NS <- train(POSITION~., 
                            data = ts.trainSet_R_NS, 
                            method="rf", 
                            trControl=fitControl,
                            tuneLength = 5))

#    user  system elapsed 
#  99.53    0.29 2069.18 

rf_R_NS
# mtry  Accuracy   Kappa    
#   117   0.9888112  0.9877335

varImp(rf_R_NS)

# test against test set
rf_R_NS_pred <- predict(rf_R_NS, ts.testSet_R_NS)
#performace measurment
postResample(rf_R_NS_pred, ts.testSet_R_NS$POSITION)

# Accuracy     Kappa 
# 0.9911717 0.9903212 

# --------------- RF train ts.trainset_R_NS Center/Scale  --------------- #
set.seed(123)
system.time(rf_R_NS_CS <- train(POSITION~., 
                               data = ts.trainSet_R_NS, 
                               method="rf", 
                               trControl=fitControl,
                               tuneLength = 5,
                               preProc = c("zv", "center", "scale")))

#    user  system   elapsed 
#    88.14    0.30 1815.24

rf_R_NS_CS
# mtry  Accuracy   Kappa    
# 117   0.9880091  0.9868542

varImp(rf_R_NS_CS)

# test against test set
rf_R_NS_CS_pred <- predict(rf_R_NS_CS, ts.testSet_R_NS)
#performace measurment
postResample(rf_R_NS_CS_pred, ts.testSet_R_NS$POSITION)

# Accuracy     Kappa 
# 0.9903692 0.9894403

# --------------- RF train ts.trainset_R_Cor  --------------- #
set.seed(123)
system.time(rf_R_Cor <- train(POSITION~., 
                                 data = ts.trainSet_R_Cor, 
                                 method="rf", 
                                 trControl=fitControl,
                                 tuneLength = 5))

# user  system elapsed 
# 68.26    1.00 1351.84 

rf_R_Cor
# mtry  Accuracy   Kappa    
#  93   0.9869431  0.9856839

varImp(rf_R_Cor)

# test against test set
rf_R_Cor_pred <- predict(rf_R_Cor, ts.testSet_R_Cor)
#performace measurment
postResample(rf_R_Cor_pred, ts.testSet_R_Cor$POSITION)

# Accuracy     Kappa 
# 0.9895666 0.9885604

# --------------- RF train ts.trainset_R_Cor Center/Scale --------------- #
set.seed(123)
system.time(rf_R_Cor_CS <- train(POSITION~., 
                               data = ts.trainSet_R_Cor, 
                               method="rf", 
                               trControl=fitControl,
                               tuneLength = 5,
                               preProc = c("zv", "center", "scale")))

# user  system elapsed 
# 54.16    0.30 1140.07 

rf_R_Cor_CS
# mtry  Accuracy   Kappa    
#     93   0.9853445  0.9839315

varImp(rf_R_Cor_CS)

# test against test set
rf_R_Cor_CS_pred <- predict(rf_R_Cor_CS, ts.testSet_R_Cor)
#performace measurment
postResample(rf_R_Cor_CS_pred, ts.testSet_R_Cor$POSITION)

# Accuracy     Kappa 
# 0.9903692 0.9894403

# --------------- Comparing rf models --------------- #

set.seed(123)
RFModelFitResults <- resamples(list(rf_OOB = rf_OOB, rf_OOB_CS = rf_OOB_CS, 
                                    rf_R_NS = rf_R_NS, rf_R_NS_CS = rf_R_NS_CS, 
                                    rf_R_Cor = rf_R_Cor, rf_R_Cor_CS = rf_R_Cor_CS))

# output summary metrics for tuned models 
summary(RFModelFitResults)

# Call:
#   summary.resamples(object = RFModelFitResults)
# 
# Models: rf_OOB, rf_OOB_CS, rf_R_NS, rf_R_NS_CS, rf_R_Cor, rf_R_Cor_CS 
# Number of resamples: 10 
# 
# Accuracy 
# Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
# rf_OOB      0.9839572 0.9873528 0.9919998 0.9912056 0.9946809 0.9973333    0
# rf_OOB_CS   0.9839572 0.9900106 0.9919999 0.9914737 0.9920424 0.9973404    0
# rf_R_NS     0.9839572 0.9867109 0.9893047 0.9888112 0.9920106 0.9920424    0
# rf_R_NS_CS  0.9813830 0.9866488 0.9879891 0.9880091 0.9913493 0.9920424    0
# rf_R_Cor    0.9787234 0.9839250 0.9879822 0.9869431 0.9893617 0.9946950    0
# rf_R_Cor_CS 0.9812332 0.9820636 0.9853155 0.9853445 0.9867021 0.9920424    0
# 
# Kappa 
#                  Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
# rf_OOB      0.9824035 0.9861340 0.9912278 0.9903568 0.9941679 0.9970773    0
# rf_OOB_CS   0.9824035 0.9890418 0.9912319 0.9906513 0.9912806 0.9970836    0
# rf_R_NS     0.9824035 0.9854311 0.9882750 0.9877335 0.9912391 0.9912739    0
# rf_R_NS_CS  0.9795939 0.9853702 0.9868308 0.9868542 0.9905128 0.9912822    0
# rf_R_Cor    0.9766735 0.9823723 0.9868310 0.9856839 0.9883328 0.9941881    0
# rf_R_Cor_CS 0.9794217 0.9803297 0.9839086 0.9839315 0.9854164 0.9912822    0

#plotting model comparison
bwplot(RFModelFitResults)

#Best model is rf_OOB_CS

########################
## ------- KNN ------- ##
########################

# #manual tuning
# rfGrid <- expand.grid(mtry=c(28, 30, 32, 34, 36))

# --------------- knn train ts.trainset_OOB  --------------- #
set.seed(123)
system.time(knn_OOB <- train(POSITION~., 
                            data = ts.trainSet_OOB, 
                            method="knn", 
                            trControl=fitControl,
                            tuneLength = 5))

# user  system elapsed 
# 1.50    0.39   48.01 

knn_OOB
# k   Accuracy   Kappa    
# 5  0.9800247  0.9781324

varImp(knn_OOB)

# test against test set
knn_OOB_pred <- predict(knn_OOB, ts.testSet_OOB)
#performace measurment
postResample(knn_OOB_pred, ts.testSet_OOB$POSITION)

# Accuracy     Kappa 
# 0.9694779 0.9665931  

# --------------- RF train ts.trainset_OOB Center/Scale  --------------- #
set.seed(123)
system.time(knn_OOB_CS <- train(POSITION~., 
                               data = ts.trainSet_OOB, 
                               method="knn", 
                               trControl=fitControl,
                               tuneLength = 5,
                               preProc = c("zv", "center", "scale")))

# user  system elapsed 
# 1.67    0.28   35.14 

knn_OOB_CS
# k   Accuracy   Kappa    
#  5  0.9456520  0.9405272

varImp(rf_OOB_CS)

# test against test set
knn_OOB_CS_pred <- predict(knn_OOB_CS, ts.testSet_OOB)
#performace measurment
postResample(knn_OOB_CS_pred, ts.testSet_OOB$POSITION)

# Accuracy     Kappa 
# 0.9477912 0.9428356   

# --------------- knn train ts.trainset_R_NS  --------------- #
set.seed(123)
system.time(knn_R_NS <- train(POSITION~., 
                             data = ts.trainSet_R_NS, 
                             method="knn", 
                             trControl=fitControl,
                             tuneLength = 5))

# user  system elapsed 
#   1.22    0.31   35.31 

knn_R_NS
# k   Accuracy   Kappa    
# 5  0.9800247  0.9781324

varImp(knn_R_NS)

# test against test set
knn_R_NS_pred <- predict(knn_R_NS, ts.testSet_R_NS)
#performace measurment
postResample(knn_R_NS_pred, ts.testSet_R_NS$POSITION)

# Accuracy     Kappa 
# 0.9694779 0.9665931 

# --------------- knn train ts.trainset_R_NS Center/Scale  --------------- #
set.seed(123)
system.time(knn_R_NS_CS <- train(POSITION~., 
                                data = ts.trainSet_R_NS, 
                                method="knn", 
                                trControl=fitControl,
                                tuneLength = 5,
                                preProc = c("zv", "center", "scale")))

# user  system elapsed 
# 1.48    0.31   34.81  

knn_R_NS_CS
# k   Accuracy   Kappa    
#  5  0.9456520  0.9405272

varImp(knn_R_NS_CS)

# test against test set
knn_R_NS_CS_pred <- predict(knn_R_NS_CS, ts.testSet_R_NS)
#performace measurment
postResample(knn_R_NS_CS_pred, ts.testSet_R_NS$POSITION)

# Accuracy     Kappa 
# 0.9477912 0.9428356

# --------------- knn train ts.trainset_R_Cor  --------------- #
set.seed(123)
system.time(knn_R_Cor <- train(POSITION~., 
                              data = ts.trainSet_R_Cor, 
                              method="knn", 
                              trControl=fitControl,
                              tuneLength = 5))

# user  system elapsed 
#  1.08    0.24   19.34

knn_R_Cor
#  k   Accuracy   Kappa    
#   5  0.9725579  0.9699635

varImp(rf_R_Cor)

# test against test set
knn_R_Cor_pred <- predict(knn_R_Cor, ts.testSet_R_Cor)
#performace measurment
postResample(knn_R_Cor_pred, ts.testSet_R_Cor$POSITION)

# Accuracy     Kappa 
# 0.9598394 0.9560388 

# --------------- knn train ts.trainset_R_Cor Center/Scale --------------- #
set.seed(123)
system.time(knn_R_Cor_CS <- train(POSITION~., 
                                 data = ts.trainSet_R_Cor, 
                                 method="knn", 
                                 trControl=fitControl,
                                 tuneLength = 5,
                                 preProc = c("zv", "center", "scale")))

# user  system elapsed 
#  1.34    0.46   22.30 

knn_R_Cor_CS
# k   Accuracy   Kappa    
# 5  0.9363221  0.9303003

varImp(knn_R_Cor_CS)

# test against test set
knn_R_Cor_CS_pred <- predict(knn_R_Cor_CS, ts.testSet_R_Cor)
#performace measurment
postResample(knn_R_Cor_CS_pred, ts.testSet_R_Cor$POSITION)

# Accuracy     Kappa 
# 0.9293173 0.9226220  

# --------------- Comparing knn models --------------- #

set.seed(123)
knnModelFitResults <- resamples(list(knn_OOB = knn_OOB, knn_OOB_CS = knn_OOB_CS, 
                                    knn_R_NS = knn_R_NS, knn_R_NS_CS = knn_R_NS_CS, 
                                    knn_R_Cor = knn_R_Cor, knn_R_Cor_CS = knn_R_Cor_CS))

# output summary metrics for tuned models 
summary(knnModelFitResults)

# Call:
#   summary.resamples(object = knnModelFitResults)
# 
# Models: knn_OOB, knn_OOB_CS, knn_R_NS, knn_R_NS_CS, knn_R_Cor, knn_R_Cor_CS 
# Number of resamples: 10 
# 
# Accuracy 
# Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
# knn_OOB      0.9706667 0.9767003 0.9787798 0.9800247 0.9840319 0.9920424    0
# knn_OOB_CS   0.9146667 0.9417266 0.9455528 0.9456520 0.9515243 0.9654255    0
# knn_R_NS     0.9706667 0.9767003 0.9787798 0.9800247 0.9840319 0.9920424    0
# knn_R_NS_CS  0.9146667 0.9417266 0.9455528 0.9456520 0.9515243 0.9654255    0
# knn_R_Cor    0.9573333 0.9666862 0.9734395 0.9725579 0.9779142 0.9893899    0
# knn_R_Cor_CS 0.9146667 0.9252440 0.9335988 0.9363221 0.9453723 0.9602122    0
# 
# Kappa 
# Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
# knn_OOB      0.9678756 0.9745111 0.9767655 0.9781324 0.9825116 0.9912919    0
# knn_OOB_CS   0.9067200 0.9362318 0.9404089 0.9405272 0.9469325 0.9621326    0
# knn_R_NS     0.9678756 0.9745111 0.9767655 0.9781324 0.9825116 0.9912919    0
# knn_R_NS_CS  0.9067200 0.9362318 0.9404089 0.9405272 0.9469325 0.9621326    0
# knn_R_Cor    0.9533143 0.9635387 0.9709143 0.9699635 0.9758163 0.9883903    0
# knn_R_Cor_CS 0.9066068 0.9182430 0.9273318 0.9303003 0.9401453 0.9564300    0

#plotting model comparison
bwplot(knnModelFitResults)

#Best model is knn_R_NS

######################
# Compare Top Models #
######################

set.seed(123)
TOPModelFitResults <- resamples(list(knn_top = knn_R_NS, 
                                     RF_top = rf_OOB_CS, 
                                     C5.0_top = C5.0_R_NS))

# output summary metrics for tuned models 
summary(TOPModelFitResults)

# Call:
#   summary.resamples(object = TOPModelFitResults)
# 
# Models: knn_top, RF_top, C5.0_top 
# Number of resamples: 10 
# 
# Accuracy 
#           Min.      1st Qu.    Median      Mean   3rd Qu.      Max.   NA's
# knn_top  0.9706667 0.9767003 0.9787798 0.9800247 0.9840319 0.9920424    0
# RF_top   0.9761273 0.9826578 0.9907056 0.9882879 0.9940106 0.9973190    0
# C5.0_top 0.9733333 0.9820212 0.9840849 0.9848183 0.9866843 0.9973333    0
# 
# Kappa 
#               Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
# knn_top  0.9678756 0.9745111 0.9767655 0.9781324 0.9825116 0.9912919    0
# RF_top   0.9738477 0.9810049 0.9898251 0.9871722 0.9934408 0.9970651    0
# C5.0_top 0.9707876 0.9803097 0.9825784 0.9833742 0.9854177 0.9970797    0

#plotting model comparison
bwplot(TOPModelFitResults)

##############################
# Variable Importance (varImp)
##############################

# varImp is also evaluated in the model train/fit section

varImp(rf_OOB_CS)
# Overall
# WAP065  100.00
# WAP118   94.62
# WAP122   94.28
# WAP501   91.37
# WAP121   89.69
# WAP117   86.73
# WAP066   84.40
# WAP132   81.35
# WAP128   79.96
# WAP096   79.69
# WAP087   77.58
# WAP127   77.10
# WAP011   73.99
# WAP012   70.08
# WAP097   69.81
# WAP070   68.89
# WAP145   65.73
# WAP131   65.50
# WAP144   65.30
# WAP162   61.23

v_imp <- varImp(rf_OOB_CS, scale = T)

plot(v_imp, top = 20)

# ---- Conclusion ---- #

#best model based off of Accuracy and Kappa is rf_OOB_CS

saveRDS(rf_OOB_CS, "rfFit.rds")  

# load and name model
rf_OOB_CS <- readRDS("rfFit.rds")


############################
# Predict testSet/validation
############################

# predict with Top Model
rfPred1 <- predict(rf_OOB_CS, trainset_OOB)
#performace measurment
postResample(rfPred1, trainset_OOB$POSITION)

# Accuracy     Kappa 
# 0.9910363 0.9901946 

# predict validation data with Top Model
rfPred1 <- predict(rf_OOB_CS, testset_OOB)
#performace measurment
postResample(rfPred1, testset_OOB$POSITION)

# Accuracy     Kappa 
# 0.8874887 0.8743117

#plot predicted verses actual
plot(rfPred1,testset_OOB$POSITION)

# print predictions
rfPred1
summary(rfPred1)

CM <- confusionMatrix(rfPred1,testset_OOB$POSITION)
CM

# Confusion Matrix and Statistics
# 
# Reference
# Prediction 0_0 0_1 0_2 0_3 1_0 1_1 1_2 1_3 2_0 2_1 2_2 2_3 2_4
# 0_0  74   1   1   0   0   0   0   0   0   0   0   0   0
# 0_1   2 200   3   0   0   0   0   0   0   0   0   0   0
# 0_2   1   1 155   2   0   0   0   0   0   0   0   0   0
# 0_3   0   0   3  82   0   0   0   0   0   0   0   0   0
# 1_0   0   0   0   0  24   4   0   0   0   0   0   0   0
# 1_1   0   0   0   0   2  92   1   0   0   0   0   0   0
# 1_2   0   0   0   0   2  37  81   3   0   0   0   0   0
# 1_3   1   6   3   1   2  10   5  44   0   1   0   0   1
# 2_0   0   0   0   0   0   0   0   0  21   0   0   0   1
# 2_1   0   0   0   0   0   0   0   0   3 108   1   0   0
# 2_2   0   0   0   0   0   0   0   0   0   2  38   0   0
# 2_3   0   0   0   0   0   0   0   0   0   0  15  40  10
# 2_4   0   0   0   0   0   0   0   0   0   0   0   0  27
# 
# Overall Statistics
# 
# Accuracy : 0.8875          
# 95% CI : (0.8674, 0.9055)
# No Information Rate : 0.1872          
# P-Value [Acc > NIR] : < 2.2e-16       
# 
# Kappa : 0.8743          
# 
# Mcnemar's Test P-Value : NA              
# 
# Statistics by Class:
# 
# Class: 0_0 Class: 0_1 Class: 0_2 Class: 0_3 Class: 1_0 Class: 1_1 Class: 1_2
# Sensitivity             0.94872     0.9615     0.9394    0.96471     0.8000    0.64336    0.93103
# Specificity             0.99806     0.9945     0.9958    0.99708     0.9963    0.99690    0.95898
# Pos Pred Value          0.97368     0.9756     0.9748    0.96471     0.8571    0.96842    0.65854
# Neg Pred Value          0.99614     0.9912     0.9895    0.99708     0.9945    0.94980    0.99393
# Prevalence              0.07021     0.1872     0.1485    0.07651     0.0270    0.12871    0.07831
# Detection Rate          0.06661     0.1800     0.1395    0.07381     0.0216    0.08281    0.07291
# Detection Prevalence    0.06841     0.1845     0.1431    0.07651     0.0252    0.08551    0.11071
# Balanced Accuracy       0.97339     0.9780     0.9676    0.98089     0.8981    0.82013    0.94501
# Class: 1_3 Class: 2_0 Class: 2_1 Class: 2_2 Class: 2_3 Class: 2_4
# Sensitivity             0.93617     0.8750    0.97297     0.7037    1.00000     0.6923
# Specificity             0.97180     0.9991    0.99600     0.9981    0.97666     1.0000
# Pos Pred Value          0.59459     0.9545    0.96429     0.9500    0.61538     1.0000
# Neg Pred Value          0.99711     0.9972    0.99700     0.9851    1.00000     0.9889
# Prevalence              0.04230     0.0216    0.09991     0.0486    0.03600     0.0351
# Detection Rate          0.03960     0.0189    0.09721     0.0342    0.03600     0.0243
# Detection Prevalence    0.06661     0.0198    0.10081     0.0360    0.05851     0.0243
# Balanced Accuracy       0.95399     0.9370    0.98449     0.8509    0.98833     0.8462

## prediction whole validation set, but first must convert 100's to -105
wifi_validation_OOB_updated <- wifi_validation_OOB %>% 
  mutate_at(vars(starts_with("WAP")), funs(sapply(.,convert)))

# predict with Top Model on complete validation dataset
rfPred2 <- predict(rf_OOB_CS, wifi_validation_OOB_updated)
#performace measurment
postResample(rfPred2, wifi_validation_OOB_updated$POSITION)

# Accuracy     Kappa 
# 0.8874887 0.8743117

#plot predicted verses actual
plot(rfPred2,wifi_validation_OOB_updated$POSITION)

# print predictions
rfPred2
summary(rfPred2)

CM2 <- confusionMatrix(rfPred2,wifi_validation_OOB_updated$POSITION)
CM2

# Confusion Matrix and Statistics
# 
# Reference
# Prediction 0_0 0_1 0_2 0_3 1_0 1_1 1_2 1_3 2_0 2_1 2_2 2_3 2_4
# 0_0  74   1   1   0   0   0   0   0   0   0   0   0   0
# 0_1   2 200   3   0   0   0   0   0   0   0   0   0   0
# 0_2   1   1 155   2   0   0   0   0   0   0   0   0   0
# 0_3   0   0   3  82   0   0   0   0   0   0   0   0   0
# 1_0   0   0   0   0  24   4   0   0   0   0   0   0   0
# 1_1   0   0   0   0   2  92   1   0   0   0   0   0   0
# 1_2   0   0   0   0   2  37  81   3   0   0   0   0   0
# 1_3   1   6   3   1   2  10   5  44   0   1   0   0   1
# 2_0   0   0   0   0   0   0   0   0  21   0   0   0   1
# 2_1   0   0   0   0   0   0   0   0   3 108   1   0   0
# 2_2   0   0   0   0   0   0   0   0   0   2  38   0   0
# 2_3   0   0   0   0   0   0   0   0   0   0  15  40  10
# 2_4   0   0   0   0   0   0   0   0   0   0   0   0  27
# 
# Overall Statistics
# 
# Accuracy : 0.8875          
# 95% CI : (0.8674, 0.9055)
# No Information Rate : 0.1872          
# P-Value [Acc > NIR] : < 2.2e-16       
# 
# Kappa : 0.8743          
# 
# Mcnemar's Test P-Value : NA              
# 
# Statistics by Class:
# 
# Class: 0_0 Class: 0_1 Class: 0_2 Class: 0_3 Class: 1_0 Class: 1_1 Class: 1_2
# Sensitivity             0.94872     0.9615     0.9394    0.96471     0.8000    0.64336    0.93103
# Specificity             0.99806     0.9945     0.9958    0.99708     0.9963    0.99690    0.95898
# Pos Pred Value          0.97368     0.9756     0.9748    0.96471     0.8571    0.96842    0.65854
# Neg Pred Value          0.99614     0.9912     0.9895    0.99708     0.9945    0.94980    0.99393
# Prevalence              0.07021     0.1872     0.1485    0.07651     0.0270    0.12871    0.07831
# Detection Rate          0.06661     0.1800     0.1395    0.07381     0.0216    0.08281    0.07291
# Detection Prevalence    0.06841     0.1845     0.1431    0.07651     0.0252    0.08551    0.11071
# Balanced Accuracy       0.97339     0.9780     0.9676    0.98089     0.8981    0.82013    0.94501
# Class: 1_3 Class: 2_0 Class: 2_1 Class: 2_2 Class: 2_3 Class: 2_4
# Sensitivity             0.93617     0.8750    0.97297     0.7037    1.00000     0.6923
# Specificity             0.97180     0.9991    0.99600     0.9981    0.97666     1.0000
# Pos Pred Value          0.59459     0.9545    0.96429     0.9500    0.61538     1.0000
# Neg Pred Value          0.99711     0.9972    0.99700     0.9851    1.00000     0.9889
# Prevalence              0.04230     0.0216    0.09991     0.0486    0.03600     0.0351
# Detection Rate          0.03960     0.0189    0.09721     0.0342    0.03600     0.0243
# Detection Prevalence    0.06661     0.0198    0.10081     0.0360    0.05851     0.0243
# Balanced Accuracy       0.95399     0.9370    0.98449     0.8509    0.98833     0.8462


##############################
# Creating Combined Data Set #
##############################

wifi_training_OOB_updated <- wifi_training_OOB %>%
  mutate_at(vars(starts_with("WAP")), funs(sapply(.,convert))) %>% 
  mutate(DataSet = "Training")

wifi_validation_OOB_updated <- wifi_validation_OOB_updated %>% 
  mutate(DataSet = "Validation")

wifi_combined <- rbind(wifi_training_OOB_updated, wifi_validation_OOB_updated)

# predict with Top Model on complete validation dataset
rfPred3 <- predict(rf_OOB_CS, wifi_combined)
#performace measurment
postResample(rfPred3, wifi_combined$POSITION)

# Accuracy     Kappa 
# 0.9855568 0.9842468

#plot predicted verses actual
plot(rfPred3, wifi_combined$POSITION)

# print predictions
rfPred3
summary(rfPred3)

CM3 <- confusionMatrix(rfPred3, wifi_combined$POSITION)
CM3

# Confusion Matrix and Statistics
# 
# Reference
# Prediction  0_0  0_1  0_2  0_3  1_0  1_1  1_2  1_3  2_0  2_1  2_2  2_3  2_4
# 0_0 1126    5    1    0    0    0    0    0    0    0    0    0    0
# 0_1    8 1551    7    0    0    0    0    0    0    0    0    0    0
# 0_2    1    2 1590    2    0    0    0    0    0    0    0    0    0
# 0_3    0    0    4 1472    0    0    0    0    0    0    0    0    0
# 1_0    0    0    0    0 1377   31    0    0    0    0    0    0    0
# 1_1    0    0    0    0   16 1537    1    0    0    0    0    0    0
# 1_2    0    0    0    0    2   47 1473   16    0    0    0    0    0
# 1_3    2    6    6    2    3   12    9  979   37    2    3    3    1
# 2_0    0    0    0    0    0    0    0    0 1926    6    0    0    1
# 2_1    0    0    0    0    0    0    0    0    3 2260   14    0    1
# 2_2    0    0    0    0    0    0    0    0    0    5 1589    1    0
# 2_3    0    0    0    0    0    0    0    0    0    0   25 2745   19
# 2_4    0    0    0    0    0    0    0    0    0    0    0    0 1119
# 
# Overall Statistics
# 
# Accuracy : 0.9856          
# 95% CI : (0.9839, 0.9871)
# No Information Rate : 0.1306          
# P-Value [Acc > NIR] : < 2.2e-16       
# 
# Kappa : 0.9842          
# 
# Mcnemar's Test P-Value : NA              
# 
# Statistics by Class:
# 
# Class: 0_0 Class: 0_1 Class: 0_2 Class: 0_3 Class: 1_0 Class: 1_1 Class: 1_2
# Sensitivity             0.99033    0.99169    0.98881    0.99729    0.98498    0.94468    0.99326
# Specificity             0.99970    0.99923    0.99974    0.99980    0.99842    0.99912    0.99668
# Pos Pred Value          0.99470    0.99042    0.99687    0.99729    0.97798    0.98906    0.95774
# Neg Pred Value          0.99945    0.99933    0.99907    0.99980    0.99893    0.99538    0.99949
# Prevalence              0.05402    0.07431    0.07640    0.07013    0.06642    0.07730    0.07046
# Detection Rate          0.05350    0.07369    0.07554    0.06994    0.06542    0.07302    0.06998
# Detection Prevalence    0.05378    0.07440    0.07578    0.07013    0.06689    0.07383    0.07307
# Balanced Accuracy       0.99501    0.99546    0.99427    0.99854    0.99170    0.97190    0.99497
# Class: 1_3 Class: 2_0 Class: 2_1 Class: 2_2 Class: 2_3 Class: 2_4
# Sensitivity             0.98392    0.97965     0.9943    0.97425     0.9985    0.98072
# Specificity             0.99571    0.99963     0.9990    0.99969     0.9976    1.00000
# Pos Pred Value          0.91925    0.99638     0.9921    0.99624     0.9842    1.00000
# Neg Pred Value          0.99920    0.99791     0.9993    0.99784     0.9998    0.99890
# Prevalence              0.04727    0.09341     0.1080    0.07749     0.1306    0.05421
# Detection Rate          0.04651    0.09151     0.1074    0.07549     0.1304    0.05316
# Detection Prevalence    0.05060    0.09184     0.1082    0.07578     0.1325    0.05316
# Balanced Accuracy       0.98982    0.98964     0.9967    0.98697     0.9981    0.99036


#reformating new products dataset
wifi_combined_w_pred <- wifi_combined %>% 
  mutate(Position_Predicted = rfPred3) %>% 
  separate(Position_Predicted, into = c("Bldg_pred", "Floor_pred"), remove = F) #adding the predicted volumes to the dataset where 0s were before
          
#writing to working directory
write.csv(wifi_combined_w_pred, "wifi_combined_w_pred.csv", row.names = F) #writing prediction results

#############################
# Accuracy of Floor and Bld #
#############################

validation <- wifi_combined_w_pred %>%
  mutate(Bldg_match = ifelse(Bldg_pred == BUILDINGID, 1, 0),
         Flr_match = ifelse(Floor_pred == FLOOR, 1, 0),
         Pos_match = ifelse(Bldg_match == 1 & Flr_match == 1, 1, 0),
         Denominator = 1,
         RowMean = rowMeans(.[,1:520]))

All_Results <- validation %>% 
  group_by(POSITION, BUILDINGID, FLOOR) %>% 
  summarise(Bld_y = sum(Bldg_match),
            Num = sum(Denominator),
            Bld_pct = (Bld_y / Num) * 100,
            Flr_y = sum(Flr_match),
            Flr_pct = (Flr_y / Num)* 100,
            Pos_y = sum(Pos_match),
            Pos_pct = (Pos_y / Num) * 100)

All_Results_Totals <- validation %>% 
  summarise(Bld_y = sum(Bldg_match),
            Num = sum(Denominator),
            Bld_pct = (Bld_y / Num) * 100,
            Flr_y = sum(Flr_match),
            Flr_pct = (Flr_y / Num)* 100,
            Pos_y = sum(Pos_match),
            Pos_pct = (Pos_y / Num) * 100)

All <- rbindlist(list(All_Results, All_Results_Totals), fill= T)


All_Results_ns <- validation %>% 
  filter(RowMean != -105) %>% 
  group_by(POSITION, BUILDINGID, FLOOR) %>% 
  summarise(Bld_y = sum(Bldg_match),
            Num = sum(Denominator),
            Bld_pct = (Bld_y / Num) * 100,
            Flr_y = sum(Flr_match),
            Flr_pct = (Flr_y / Num)* 100,
            Pos_y = sum(Pos_match),
            Pos_pct = (Pos_y / Num) * 100)

All_Results_Totals_ns <- validation %>% 
  filter(RowMean != -105) %>% 
  summarise(Bld_y = sum(Bldg_match),
            Num = sum(Denominator),
            Bld_pct = (Bld_y / Num) * 100,
            Flr_y = sum(Flr_match),
            Flr_pct = (Flr_y / Num)* 100,
            Pos_y = sum(Pos_match),
            Pos_pct = (Pos_y / Num) * 100)

All_ns <- rbindlist(list(All_Results_ns, All_Results_Totals_ns), fill= T)

#writing to working directory
write.csv(All, "all.csv", row.names = F)

write.csv(All_ns, "all_ns.csv", row.names = F)

######################################################
# Start of predicting Floor_SpaceID_RelativeLocation #
######################################################

# -- Showing analysis could be done to predict more than just BLDG and Floor, but using specific bldg
#  versus a subset of overall data

#################
# Preprocess #2 #
#################

#creating out of box training dataset
trainset2_OOB <- wifi_training %>%
  filter(BUILDINGID == 0) %>% 
  unite("POSITION2", FLOOR, SPACEID, RELATIVEPOSITION, sep = "_", remove = T) %>% 
  select(-LONGITUDE, #do not need these values
         -LATITUDE,
         -USERID,
         -PHONEID,
         -TIMESTAMP,
         -BUILDINGID) %>% 
  mutate_at(vars(starts_with("WAP")), funs(sapply(.,convert))) %>% #converting '100s' -- no signal to -105
  mutate_at(vars(POSITION2), funs(as.factor))

str(trainset2_OOB) # ''data.frame':	5249 obs. of  521 variables

length(unique(trainset2_OOB$POSITION2)) #259

# remove zero variance predictors 
zv <- apply(trainset2_OOB, 2, function(x) length(unique(x))==1)

trainset2_zv <- trainset2_OOB[,!zv]

str(trainset2_zv) # ''data.frame':	5249 obs. of  201 variables

length(unique(trainset2_zv$POSITION2)) #259

#--- Dataset 2 ---#

testset2_OOB <- wifi_validation %>%
  filter(BUILDINGID == 0) %>% 
  unite("POSITION2", FLOOR, RELATIVEPOSITION, SPACEID, sep = "_", remove = T) %>% 
  select(-LONGITUDE,
         -LATITUDE,
         -USERID,
         -PHONEID,
         -TIMESTAMP,
         -BUILDINGID) %>%
  mutate_at(vars(starts_with("WAP")), funs(sapply(.,convert))) %>% #converting '100s' -- no signal to -105
  mutate_at(vars(POSITION2), funs(as.factor))

str(testset2_OOB) #'data.frame':	536 obs. of  521 variables:

length(unique(testset2_OOB$POSITION2)) # 4, but makes complete sense, b/c these are unknown values we're
                                         # trying to predict

# remove same as trainset for zero variance predictors

testset2_zv <- testset2_OOB[,!zv]

str(testset2_zv) #'data.frame':	536 obs. of  201 variables:

length(unique(testset2_zv$POSITION2)) # 4, but makes complete sense, b/c these are unknown values we're
                                      # trying to predict

################ 
# Filtering #2 #
################
testcorr2 <- trainset2_zv
# good for num/int data 
testcorr2[] <- lapply(testcorr2[1:201], as.integer)
# calculate correlation matrix for all vars
corr2_zv <- cor(testcorr2[,1:200], use = "complete.obs")
# summarize the correlation matrix
corr2_zv
# plot correlation matrix
corrplot(corr2_zv)

# find WAPSs that are highly corrected (ideally >0.90)
highlyCorrelated_zv <- findCorrelation(corr2_zv, cutoff=0.90)
# summarize the correlation matrix
highlyCorrelated_zv
#  [1]  7 13 17 19 21 23 27 31 33 35 37 41 43 45 47 50 65 10 14 24 28 38 56 62 71

# get var name of high corr
HighCorr2 <- colnames(testcorr2[c(highlyCorrelated_zv)]) 
HighCorr2
#  [1] "WAP014" "WAP024" "WAP028" "WAP030" "WAP032" "WAP034" "WAP040" "WAP044" "WAP046" "WAP048" "WAP050"
# [12] "WAP054" "WAP058" "WAP072" "WAP076" "WAP081" "WAP162" "WAP019" "WAP025" "WAP035" "WAP041" "WAP051"
# [23] "WAP142" "WAP155" "WAP170"

trainset2_R_Cor <- trainset2_zv[,!names(trainset2_zv) %in% c(HighCorr2)]
testset2_R_Cor <- testset2_zv[,!names(testset2_zv) %in% c(HighCorr2)]

str(trainset2_R_Cor) #'data.frame':	5249 obs. of  176 variables:

length(unique(trainset2_R_Cor$POSITION2)) #259

str(testset2_R_Cor)  #data.frame':	536 obs. of  176 variables:

length(unique(testset2_R_Cor$POSITION2)) #4 -- Makes sense...see comments above

###############
# Sampling #2 #
###############

# ---- Sampling ---- #

# create the training partition that is 75% of total obs
set.seed(123) # set random seed
inTraining_OOB2 <- createDataPartition(trainset2_OOB$POSITION2, p=0.75, list=FALSE)

# create training/testing dataset
ts.trainSet2_OOB <- trainset2_OOB[inTraining_OOB2,]   
ts.testSet2_OOB <- trainset2_OOB[-inTraining_OOB2,]  

# verify number of obs 
str(ts.trainSet2_OOB) # 'data.frame':	3996 obs. of  521 variables:
str(ts.testSet2_OOB) # 'data.frame':	1253 obs. of  521 variables:

length(unique(ts.trainSet2_OOB$POSITION2)) #259
length(unique(ts.testSet2_OOB$POSITION2)) #259

# sample again after removing any features

# create the training partition that is 75% of total obs
set.seed(123) # set random seed
inTraining_zv2 <- createDataPartition(trainset2_zv$POSITION2, p=0.75, list=FALSE)

# create training/testing dataset
ts.trainSet2_zv <- trainset2_zv[inTraining_zv2,]   
ts.testSet2_zv <- trainset2_zv[-inTraining_zv2,]  

# verify number of obs 
str(ts.trainSet2_zv) # 'data.frame':	3996 obs. of  201 variables:
str(ts.testSet2_zv) # 'data.frame':	1253 obs. of  201 variables:

length(unique(ts.trainSet2_zv$POSITION2)) #259
length(unique(ts.testSet2_zv$POSITION2)) #259

# sample again after removing any features

# create the training partition that is 75% of total obs
set.seed(123) # set random seed
inTraining_R_Cor <- createDataPartition(trainset2_R_Cor$POSITION2, p=0.75, list=FALSE)

# create training/testing dataset
ts.trainSet2_R_Cor <- trainset2_R_Cor[inTraining_R_Cor,]   
ts.testSet2_R_Cor <- trainset2_R_Cor[-inTraining_R_Cor,]  

# verify number of obs 
str(ts.trainSet2_R_Cor) # 'data.frame':	3996 obs. of  176 variables:
str(ts.testSet2_R_Cor) # 'data.frame':	1253 obs. of  176 variables:

length(unique(ts.trainSet2_R_Cor$POSITION2)) #259
length(unique(ts.testSet2_R_Cor$POSITION2)) #259

#####################################
# Datasets created above to test #2 #
#####################################
ts.trainSet2_OOB   # 'data.frame':	3755 obs. of  521 variables
ts.trainSet2_zv  # 'data.frame':	3755 obs. of  466 variables:
ts.trainSet2_R_Cor # 'data.frame':	3755 obs. of  367 variables:


#########################################
# ----------- Train model 2 ----------- #
#########################################

##############################
## ------- C5.0 #2 -------  ##
##############################

# --------------- C5.0 train ts.trainSet2_OOB  --------------- #
set.seed(123)
system.time(C5.0_OOB2 <- train(POSITION2~.,
                              data=ts.trainSet2_OOB,
                              method="C5.0",
                              trControl=fitControl))

# user  system elapsed 
# 37.49    7.18  349.64 

C5.0_OOB2 
# model  winnow  trials  Accuracy   Kappa 
# tree   FALSE   20      0.7689478  0.7679477

varImp(C5.0_OOB2)

# test against test set
C5.0_OOB2_pred <- predict(C5.0_OOB2, ts.testSet2_OOB)

options(error=utils::recover)
#performace measurment
postResample(C5.0_OOB2_pred, ts.testSet2_OOB$POSITION2)

# Accuracy     Kappa 
# 0.7797287 0.7788296

# --------------- C5.0 train ts.trainSet2_OOB Center/Scale  --------------- #
set.seed(123)
system.time(C5.0_OOB2_CS <- train(POSITION2~.,
                                 data=ts.trainSet2_OOB,
                                 method="C5.0",
                                 trControl=fitControl,
                                 preProc = c("zv", "center", "scale")))
# method = "center" subtracts the mean of the predictor's data from the predictor values 
# method = "scale" divides by the standard deviation.
# method = "zv" excluded zero variance columns

# user  system elapsed 
# 846.09  114.08  961.64

C5.0_OOB2_CS
# model  winnow  trials  Accuracy   Kappa 
#  tree   FALSE   20     0.7669985  0.7659901

varImp(C5.0_OOB2_CS)

# test against test set
C5.0_OOB2_CS_pred <- predict(C5.0_OOB2_CS, ts.testSet2_OOB)
#performace measurment
postResample(C5.0_OOB2_CS_pred, ts.testSet2_OOB$POSITION2)

# Accuracy     Kappa 
# 0.7837191 0.7828377 

# --------------- C5.0 train ts.trainSet2_zv --------------- #
set.seed(123)
system.time(C5.0_R_zv <- train(POSITION2~.,
                               data=ts.trainSet2_zv, 
                               method="C5.0", 
                               trControl=fitControl))

# user  system elapsed 
# 1688.33    4.35 1703.41 

C5.0_R_zv

# model  winnow  trials  Accuracy   Kappa    
# tree   FALSE   20      0.7689478  0.7679477

varImp(C5.0_R_zv)

# test against test set
C5.0_R_zv_pred <- predict(C5.0_R_zv, ts.testSet2_zv)
#performace measurment
postResample(C5.0_R_zv_pred, ts.testSet2_zv$POSITION2)

# Accuracy     Kappa 
#  0.7797287 0.7788296 

# --------------- C5.0 train ts.trainSet2_zv Center/Scale --------------- #
set.seed(123)
system.time(C5.0_R_zv_CS <- train(POSITION2~.,
                                  data=ts.trainSet2_zv, 
                                  method="C5.0", 
                                  trControl=fitControl,
                                  preProc = c("zv", "center", "scale")))

# user  system elapsed 
# 846.87  115.07  963.76   

C5.0_R_zv_CS
# model  winnow  trials  Accuracy   Kappa    
# rules   TRUE   20      0.7669985  0.7659901

varImp(C5.0_R_zv_CS)

# test against test set
C5.0_R_zv_CS_pred <- predict(C5.0_R_zv_CS, ts.testSet2_zv)
#performace measurment
postResample(C5.0_R_zv_CS_pred, ts.testSet2_zv$POSITION2)

# Accuracy     Kappa 
# 0.7837191 0.7828377

# --------------- C5.0 train ts.trainSet2_R_Cor --------------- #
set.seed(123)
system.time(C5.0_R_Cor2 <- train(POSITION2~.,
                                data=ts.trainSet2_R_Cor, 
                                method="C5.0", 
                                trControl=fitControl))

# user  system elapsed 
# 795.74  140.72  939.55  

C5.0_R_Cor2  
# model  winnow  trials  Accuracy   Kappa    
# rules  FALSE   20      0.7512952  0.7502210

varImp(C5.0_R_Cor2)

# test against test set
C5.0_R_Cor2_pred <- predict(C5.0_R_Cor2 , ts.testSet2_R_Cor)
#performace measurment
postResample(C5.0_R_Cor2_pred, ts.testSet2_R_Cor$POSITION)

# Accuracy    Kappa 
# 0.7685555 0.7676086  

# --------------- C5.0 train ts.trainSet2_R_Cor Center/Scale--------------- #
set.seed(123)
system.time(C5.0_R_Cor2_CS <- train(POSITION2~.,
                                   data=ts.trainSet2_R_Cor, 
                                   method="C5.0", 
                                   trControl=fitControl,
                                   preProc = c("zv", "center", "scale")))

# user  system elapsed 
# 768.04  130.42  900.43

C5.0_R_Cor2_CS  
# model  winnow  trials  Accuracy   Kappa    
#  rules   TRUE   20      0.7500839  0.7490031

varImp(C5.0_R_Cor2_CS)

# test against test set
C5.0_R_Cor2_CS_pred <- predict(C5.0_R_Cor2_CS , ts.testSet2_R_Cor)
#performace measurment
postResample(C5.0_R_Cor2_CS_pred, ts.testSet2_R_Cor$POSITION2)

# Accuracy    Kappa 
# 0.7693536 0.7684116

# --------------- Comparing C5.0 models --------------- #

set.seed(123)
C5.0ModelFitResults2 <- resamples(list(C5OOB2 = C5.0_OOB2, C5OOB2_CS = C5.0_OOB2_CS, 
                                      C5_R_ZV = C5.0_R_zv, C5_R_ZV_CS = C5.0_R_zv_CS, 
                                      C5_R_Cor2 = C5.0_R_Cor2, C5_R_Cor2_CS = C5.0_R_Cor2_CS))
# output summary metrics for tuned models 
summary(C5.0ModelFitResults2)

# Call:
#   summary.resamples(object = C5.0ModelFitResults2)
# 
# Models: C5OOB2, C5OOB2_CS, C5_R_ZV, C5_R_ZV_CS, C5_R_Cor2, C5_R_Cor2_CS 
# Number of resamples: 10 
# 
# Accuracy 
#               Min.      1st Qu.    Median      Mean   3rd Qu.      Max. NA's
# C5OOB2       0.7335025 0.7454236 0.7641515 0.7689478 0.7836780 0.8175182    0
# C5OOB2_CS    0.7208122 0.7600573 0.7655407 0.7669985 0.7738550 0.8102190    0
# C5_R_ZV      0.7335025 0.7454236 0.7641515 0.7689478 0.7836780 0.8175182    0
# C5_R_ZV_CS   0.7208122 0.7600573 0.7655407 0.7669985 0.7738550 0.8102190    0
# C5_R_Cor2    0.7258883 0.7297569 0.7444077 0.7512952 0.7719335 0.7925926    0
# C5_R_Cor2_CS 0.7128713 0.7314727 0.7518519 0.7500839 0.7708821 0.7780612    0
# 
# Kappa 
# Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
# C5OOB2       0.7323543 0.7443072 0.7631453 0.7679477 0.7827286 0.8167328    0
# C5OOB2_CS    0.7195857 0.7590291 0.7645293 0.7659901 0.7728772 0.8094077    0
# C5_R_ZV      0.7323543 0.7443072 0.7631453 0.7679477 0.7827286 0.8167328    0
# C5_R_ZV_CS   0.7195857 0.7590291 0.7645293 0.7659901 0.7728772 0.8094077    0
# C5_R_Cor2    0.7246930 0.7285988 0.7432916 0.7502210 0.7709526 0.7916960    0
# C5_R_Cor2_CS 0.7116523 0.7302974 0.7507748 0.7490031 0.7698915 0.7771068    0

#plotting model comparison
bwplot(C5.0ModelFitResults2)

#Best model is C5OOB2

###########################
## ------- RF #2 ------- ##
###########################

# #manual tuning
#rfGrid <- expand.grid(mtry=c(28, 30, 32, 34, 36))

# --------------- RF train ts.trainset_OOB  --------------- #
set.seed(123)
system.time(rf_OOB2 <- train(POSITION2~., 
                            data = ts.trainSet2_OOB, 
                            method="rf", 
                            trControl=fitControl,
                            tuneLength = 3))

# user  system elapsed 
# 7062.06    2.20 7075.44 

rf_OOB2
# mtry  Accuracy   Kappa    
#  32   0.814229324  0.8134193

varImp(rf_OOB2)

# test against test set
rf_OOB2_pred <- predict(rf_OOB2, ts.testSet2_OOB)
#performace measurment
postResample(rf_OOB2_pred, ts.testSet2_OOB$POSITION2)

# Accuracy     Kappa 
# 0.8236233 0.8228985  

# --------------- RF train ts.trainset_OOB Center/Scale  --------------- #
set.seed(123)
system.time(rf_OOB2_CS <- train(POSITION2~., 
                               data = ts.trainSet2_OOB, 
                               method="rf", 
                               trControl=fitControl,
                               tuneLength = 3,
                               preProc = c("zv", "center", "scale")))

# user  system elapsed 
# 1507.14    2.11 1512.58 

rf_OOB2_CS
# mtry  Accuracy   Kappa    
#  32   0.8342260  0.8335060

varImp(rf_OOB2_CS)

# test against test set
rf_OOB2_CS_pred <- predict(rf_OOB2_CS, ts.testSet2_OOB)
#performace measurment
postResample(rf_OOB2_CS_pred, ts.testSet2_OOB$POSITION2)

# Accuracy     Kappa 
# 0.8355946 0.8349250   

# --------------- RF train ts.trainSet2_zv  --------------- #
set.seed(123)
system.time(rf_R_ZV <- train(POSITION2~., 
                             data = ts.trainSet2_zv, 
                             method="rf", 
                             trControl=fitControl,
                             tuneLength = 3))

#    user  system elapsed 
#  1688.33    4.35 1703.41 

rf_R_ZV
# mtry  Accuracy   Kappa    
#   101   0.8213005  0.8205249

varImp(rf_R_ZV)

# test against test set
rf_R_ZV_pred <- predict(rf_R_ZV, ts.testSet2_zv)
#performace measurment
postResample(rf_R_ZV_pred, ts.testSet2_zv$POSITION2)

# Accuracy     Kappa 
# 0.8316042 0.8309190

# --------------- RF train ts.trainSet2_zv Center/Scale  --------------- #
set.seed(123)
system.time(rf_R_ZV_CS <- train(POSITION2~., 
                             data = ts.trainSet2_zv, 
                             method="rf", 
                             trControl=fitControl,
                             tuneLength = 3,
                             preProc = c("zv", "center", "scale")))

#    user  system   elapsed 
#   1806.87    2.22 1815.23

rf_R_ZV_CS
# mtry  Accuracy   Kappa    
# 101   0.8167670  0.8159721

varImp(rf_R_ZV_CS)

# test against test set
rf_R_ZV_CS_pred <- predict(rf_R_ZV_CS, ts.testSet2_zv)
#performace measurment
postResample(rf_R_ZV_CS_pred, ts.testSet2_zv$POSITION2)

# Accuracy     Kappa 
# 0.8339984 0.8333231

# --------------- RF train ts.trainset_R_Cor  --------------- #
set.seed(123)
system.time(rf_R_Cor2 <- train(POSITION2~., 
                              data = ts.trainSet2_R_Cor, 
                              method="rf", 
                              trControl=fitControl,
                              tuneLength = 3))

# user  system elapsed 
# 1619.39    1.72 1625.48 

rf_R_Cor2
# mtry  Accuracy   Kappa    
#  88   0.8044972  0.8036459

varImp(rf_R_Cor2)

# test against test set
rf_R_Cor2_pred <- predict(rf_R_Cor2, ts.testSet2_R_Cor)
#performace measurment
postResample(rf_R_Cor2_pred, ts.testSet2_R_Cor$POSITION2)

# Accuracy     Kappa 
# 0.8148444 0.8140892 

# --------------- RF train ts.trainset_R_Cor Center/Scale --------------- #
set.seed(123)
system.time(rf_R_Cor2_CS <- train(POSITION2~., 
                                 data = ts.trainSet2_R_Cor, 
                                 method="rf", 
                                 trControl=fitControl,
                                 tuneLength = 3,
                                 preProc = c("zv", "center", "scale")))

# user  system elapsed 
# 1490.86    3.00 1495.77 

rf_R_Cor2_CS
# mtry  Accuracy   Kappa    
# 88   0.8064925  0.8056482

varImp(rf_R_Cor2_CS)

# test against test set
rf_R_Cor2_CS_pred <- predict(rf_R_Cor2_CS, ts.testSet2_R_Cor)
#performace measurment
postResample(rf_R_Cor2_CS_pred, ts.testSet2_R_Cor$POSITION2)

# Accuracy     Kappa 
# 0.8164405 0.8156904

# --------------- Comparing rf models --------------- #

set.seed(123)
RFModelFitResults2 <- resamples(list(rf_OOB2 = rf_OOB2, rf_OOB2_CS = rf_OOB2_CS, 
                                    rf_R_ZV = rf_R_ZV, rf_R_ZV_CS = rf_R_ZV_CS, 
                                    rf_R_Cor2 = rf_R_Cor2, rf_R_Cor2_CS = rf_R_Cor2_CS))

# output summary metrics for tuned models 
summary(RFModelFitResults2)

# Call:
#   summary.resamples(object = RFModelFitResults2)
# 
# Models: rf_OOB2, rf_OOB2_CS, rf_R_ZV, rf_R_ZV_CS, rf_R_Cor2, rf_R_Cor2_CS 
# Number of resamples: 10 
# 
# Accuracy 
# Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
# rf_OOB2      0.7945545 0.8085915 0.8126551 0.8142293 0.8225605 0.8329238    0
# rf_OOB2_CS   0.7995050 0.8201810 0.8365163 0.8342260 0.8478185 0.8673219    0
# rf_R_ZV      0.7896040 0.8072499 0.8199589 0.8213005 0.8389237 0.8543210    0
# rf_R_ZV_CS   0.7747525 0.8025154 0.8197335 0.8167670 0.8295332 0.8543210    0
# rf_R_Cor2    0.7772277 0.7816613 0.8036764 0.8044972 0.8251595 0.8345679    0
# rf_R_Cor2_CS 0.7783375 0.7847898 0.8049171 0.8064925 0.8251595 0.8469136    0
# 
# Kappa 
# Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
# rf_OOB2      0.7936594 0.8077570 0.8118388 0.8134193 0.8217918 0.8321863    0
# rf_OOB2_CS   0.7986438 0.8193927 0.8358083 0.8335060 0.8471552 0.8667402    0
# rf_R_ZV      0.7887042 0.8064208 0.8191764 0.8205249 0.8382219 0.8536769    0
# rf_R_ZV_CS   0.7737962 0.8016482 0.8189510 0.8159721 0.8287953 0.8536778    0
# rf_R_Cor2    0.7762751 0.7806976 0.8028280 0.8036459 0.8243917 0.8338365    0
# rf_R_Cor2_CS 0.7773614 0.7838504 0.8040677 0.8056482 0.8243931 0.8462386    0

#plotting model comparison
bwplot(RFModelFitResults2)

#Best model is rf_OOB2_CS

############################
## ------- KNN #2 ------- ##
############################


# --------------- knn train ts.trainset_OOB  --------------- #
set.seed(123)
system.time(knn_OOB2 <- train(POSITION2~., 
                             data = ts.trainSet2_OOB, 
                             method="knn", 
                             trControl=fitControl,
                             tuneLength = 3))

# user  system elapsed 
# 68.45    0.47   69.06 

knn_OOB2
# k   Accuracy   Kappa    
#   5  0.6981501  0.6968605

varImp(knn_OOB2)

# test against test set
knn_OOB2_pred <- predict(knn_OOB2, ts.testSet2_OOB)
#performace measurment
postResample(knn_OOB2_pred, ts.testSet2_OOB$POSITION2)

# Accuracy     Kappa 
# 0.6943336 0.6930909

# --------------- RF train ts.trainset_OOB Center/Scale  --------------- #
set.seed(123)
system.time(knn_OOB2_CS <- train(POSITION2~., 
                                data = ts.trainSet2_OOB, 
                                method="knn", 
                                trControl=fitControl,
                                tuneLength = 5,
                                preProc = c("zv", "center", "scale")))

# user  system elapsed 
# 45.36    1.47   47.11 

knn_OOB2_CS
# k   Accuracy   Kappa    
# 5  0.6086477  0.6069848

varImp(rf_OOB2_CS)

# test against test set
knn_OOB2_CS_pred <- predict(knn_OOB2_CS, ts.testSet2_OOB)
#performace measurment
postResample(knn_OOB2_CS_pred, ts.testSet2_OOB$POSITION2)

# Accuracy     Kappa 
# 0.6304868 0.6289862   

# --------------- knn train ts.trainset_R_NS  --------------- #
set.seed(123)
system.time(knn_R_ZV <- train(POSITION2~., 
                              data = ts.testSet2_zv, 
                              method="knn", 
                              trControl=fitControl,
                              tuneLength = 5))

# user  system elapsed 
#   3.61    0.00    3.61

knn_R_ZV
# k   Accuracy   Kappa    
#  5  0.4064776  0.4034076

varImp(knn_R_ZV)

# test against test set
knn_R_ZV_pred <- predict(knn_R_ZV, ts.testSet2_zv)
#performace measurment
postResample(knn_R_ZV_pred, ts.testSet2_zv$POSITION2)

# Accuracy     Kappa 
# 0.6711891 0.6698319 

# --------------- knn train ts.trainset_R_NS Center/Scale  --------------- #
set.seed(123)
system.time(knn_R_ZV_CS <- train(POSITION2~., 
                                 data = ts.testSet2_zv, 
                                 method="knn", 
                                 trControl=fitControl,
                                 tuneLength = 5,
                                 preProc = c("zv", "center", "scale")))

# user  system elapsed 
#  5.69    0.24    5.92  

knn_R_ZV_CS
# k   Accuracy   Kappa    
# 5  0.3292526  0.3259907

varImp(knn_R_ZV_CS)

# test against test set
knn_R_ZV_CS_pred <- predict(knn_R_ZV_CS, ts.testSet2_zv)
#performace measurment
postResample(knn_R_ZV_CS_pred, ts.testSet2_zv$POSITION)

# Accuracy     Kappa 
# 0.5881883 0.5864927 

# --------------- knn train ts.trainset_R_Cor  --------------- #
set.seed(123)
system.time(knn_R_Cor2 <- train(POSITION2~., 
                               data = ts.trainSet2_R_Cor, 
                               method="knn", 
                               trControl=fitControl,
                               tuneLength = 5))

# user  system elapsed 
#  31.44    0.64   32.76 

knn_R_Cor2
#  k   Accuracy   Kappa    
#   5  0.6622629  0.6608246

varImp(rf_R_Cor2)

# test against test set
knn_R_Cor2_pred <- predict(knn_R_Cor2, ts.testSet2_R_Cor)
#performace measurment
postResample(knn_R_Cor2_pred, ts.testSet2_R_Cor$POSITION2)

# Accuracy     Kappa 
# 0.6751796 0.6738500

# --------------- knn train ts.trainset_R_Cor Center/Scale --------------- #
set.seed(123)
system.time(knn_R_Cor2_CS <- train(POSITION2~., 
                                  data = ts.trainSet2_R_Cor, 
                                  method="knn", 
                                  trControl=fitControl,
                                  tuneLength = 5,
                                  preProc = c("zv", "center", "scale")))

# user  system elapsed 
#  34.67    1.72   36.97 

knn_R_Cor2_CS
# k   Accuracy   Kappa    
# 5  0.5711717  0.5693543

varImp(knn_R_Cor2_CS)

# test against test set
knn_R_Cor2_CS_pred <- predict(knn_R_Cor2_CS, ts.testSet2_R_Cor)
#performace measurment
postResample(knn_R_Cor2_CS_pred, ts.testSet2_R_Cor$POSITION2)

# Accuracy     Kappa 
# 0.6025539 0.6009330 

# --------------- Comparing knn models --------------- #

set.seed(123)
knnModelFitResults2 <- resamples(list(knn_OOB2 = knn_OOB2, knn_OOB2_CS = knn_OOB2_CS, 
                                     knn_R_ZV = knn_R_ZV, knn_R_ZV_CS = knn_R_ZV_CS, 
                                     knn_R_Cor2 = knn_R_Cor2, knn_R_Cor2_CS = knn_R_Cor2_CS))

# output summary metrics for tuned models 
summary(knnModelFitResults2)

# Call:
#   summary.resamples(object = knnModelFitResults2)
# 
# Models: knn_OOB2, knn_OOB2_CS, knn_R_ZV, knn_R_ZV_CS, knn_R_Cor2, knn_R_Cor2_CS 
# Number of resamples: 10 
# 
# Accuracy 
# Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
# knn_OOB2      0.6633663 0.6891939 0.6941099 0.6981501 0.7107435 0.7372449    0
# knn_OOB2_CS   0.5761421 0.6022186 0.6118598 0.6086477 0.6197950 0.6289474    0
# knn_R_ZV      0.3333333 0.3747561 0.4113821 0.4064776 0.4420715 0.4603175    0
# knn_R_ZV_CS   0.2480620 0.2940283 0.3282583 0.3292526 0.3606739 0.4000000    0
# knn_R_Cor2    0.6322418 0.6553003 0.6626075 0.6622629 0.6734874 0.6868421    0
# knn_R_Cor2_CS 0.5495050 0.5600594 0.5730264 0.5711717 0.5796445 0.5943878    0
# 
# Kappa 
# Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
# knn_OOB2      0.6619621 0.6878668 0.6927847 0.6968605 0.7095065 0.7361477    0
# knn_OOB2_CS   0.5742883 0.6005471 0.6102211 0.6069848 0.6181847 0.6273810    0
# knn_R_ZV      0.3301938 0.3716231 0.4080909 0.4034076 0.4392552 0.4572750    0
# knn_R_ZV_CS   0.2449312 0.2906817 0.3252887 0.3259907 0.3574077 0.3968586    0
# knn_R_Cor2    0.6306741 0.6538511 0.6611603 0.6608246 0.6720710 0.6855158    0
# knn_R_Cor2_CS 0.5476341 0.5581599 0.5712097 0.5693543 0.5778641 0.5926700    0

#plotting model comparison
bwplot(knnModelFitResults2)

#Best model is knn_OOB2

#########################
# Compare Top Models #2 #
#########################

set.seed(123)
TOPModelFitResults2 <- resamples(list(knn_top2 = knn_OOB2, 
                                     RF_top2 = rf_OOB2_CS, 
                                     C5.0_top2 = C5.0_OOB_CS))

# output summary metrics for tuned models 
summary(TOPModelFitResults2)

# Call:
#   summary.resamples(object = TOPModelFitResults2)
# 
# Models: knn_top2, RF_top2, C5.0_top2 
# Number of resamples: 10 
# 
# Accuracy 
#             Min.      1st Qu.    Median      Mean   3rd Qu.      Max. NA's
# knn_top2  0.6633663 0.6891939 0.6941099 0.6981501 0.7107435 0.7372449    0
# RF_top2   0.7995050 0.8201810 0.8365163 0.8342260 0.8478185 0.8673219    0
# C5.0_top2 0.7208122 0.7600573 0.7655407 0.7669985 0.7738550 0.8102190    0
# 
# Kappa   
#             Min.     1st Qu.    Median      Mean   3rd Qu.      Max. NA's
# knn_top2  0.6619621 0.6878668 0.6927847 0.6968605 0.7095065 0.7361477    0
# RF_top2   0.7986438 0.8193927 0.8358083 0.8335060 0.8471552 0.8667402    0
# C5.0_top2 0.7195857 0.7590291 0.7645293 0.7659901 0.7728772 0.8094077    0

#plotting model comparison
bwplot(TOPModelFitResults2)

###################################
# Variable Importance (varImp) #2 #
###################################

plot(varImp(rf_OOB2_CS))

v_imp2 <- varImp(rf_OOB2_CS, scale = F)

plot(v_imp2, top = 20)

# ---- Conclusion ---- #

#best model based off of Accuracy and Kappa is rf_OOB_CS

saveRDS(rf_OOB2_CS, "rf Top.rds") 
saveRDS(knn_OOB2, "KNN Top.rds")
saveRDS(C5.0_OOB2_CS, "C5.0 Top.rds")

# load and name model
rf_OOB2_CS <- readRDS("rfFit2.rds")

############################
# Predict testSet/validation
############################

# predict with Top Model2 on complete training set
rfPred_top_model2 <- predict(rf_OOB2_CS, trainset2_OOB)

#performace measurment
postResample(rfPred_top_model2, trainset2_OOB$POSITION2)

# Accuracy     Kappa 
# 0.9607544 0.9605957

# print predictions
rfPred_top_model2
summary(rfPred_top_model2)

CM_top <- confusionMatrix(rfPred_top_model2, trainset2_OOB$POSITION2)
CM_top

#writing Confusion Matrix to CSV
tocsv <- data.frame(cbind(t(CM_top$overall)))
tocsv2 <- data.frame(cbind(t(CM_top$byClass)))
tocsv3 <- data.frame(cbind(t(CM_top$table)))
tocsv4 <- cbind(tocsv, tocsv2)

write.csv(tocsv3, "ConfusionMatrix_Floor_RelativePos_SpaceID.csv", row.names = T)
write.csv(tocsv4, "ConfusionMatrixMeasures_Floor_RelativePos_SpaceID.csv", row.names = T)

# visualizing CM
input.matrix <- data.matrix(CM_top)
input.matrix.normalized <- normalize(input.matrix)

confusion <- as.data.frame(as.table(input.matrix.normalized))

plot <- ggplot(confusion)
plot + geom_tile(aes(x=Var1, y=Var2, fill=Freq)) + 
  scale_x_discrete(name="Actual Class") + scale_y_discrete(name="Predicted Class") + 
  scale_fill_gradient(breaks=seq(from=-.5, to=4, by=.2)) + labs(fill="Normalized\nFrequency")

# predict with Top Model on validation set (missing data to predict)
rfPred_top_model_again <- predict(rf_OOB2_CS, testset2_OOB)

#performace measurment
postResample(rfPred_top_model_again, testset2_OOB$POSITION2)

#Accuracy / Kappa make sense as data predicted doesn't exist on the validation set
# Accuracy    Kappa 
# NA       NA

# print predictions
rfPred_top_model_again
summary(rfPred_top_model_again)



# Stop Cluster. After performing your tasks, make sure to stop your cluster. 
stopCluster(cl); registerDoSEQ()


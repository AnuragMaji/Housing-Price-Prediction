#import the training data
train <- read.csv("train.csv", stringsAsFactors = F)
Sale <- train$SalePrice
test <- read.csv("test.csv", stringsAsFactors = F)
submission <- read.csv("sample_submission.csv", stringsAsFactors = F)
y_train <- train$SalePrice
train_1 <- read.csv("train.csv", stringsAsFactors = F)

#Simple Linear Model


#binding train and test set for missing value imputation and feature engineering
train_1$SalePrice=NULL
train_1 = rbind(train_1, test)
ntrain=nrow(train)

features=names(train_1)

#Implementing missing value data imputation using mice. It is necessary for this problem since 
#we have missing values for several columns and some columns have almost all values that are missing
#Since the quantity of data is limited, we need to extract as much as possible 
library(Amelia)
library(mice)
library(ggplot2)
library(latticeExtra)
library(data.table)
library(xgboost)
library(Metrics)
library(Matrix)

#summary of the data
str(train_1)
summary(train_1)
attach(train_1)

#replacing multiple NA's to None for categorical features
table(is.na(train_1$PoolQC))
table(is.na(train_1$PoolArea)) 

#Hence most of the Pool QC info is unavailable since all the pools have an area mentioned

table(is.na(train_1$MiscFeature))
train_1$MiscFeature[is.na(train_1$MiscFeature)] <- rep('None', 1406)
table(is.na(train_1$Alley))
train_1$Alley[is.na(train_1$Alley)] <- rep('None', 1369)
table(is.na(train_1$Fence))
train_1$Fence[is.na(train_1$Fence)] <- rep('None', 1179)
table(is.na(train_1$FireplaceQu))
table(as.factor(train_1$Fireplaces), useNA = "ifany")
#Therefore, there are 690 houses with no fireplaces, which matches our NA count
train_1$FireplaceQu[is.na(train_1$FireplaceQu)] <- rep('None', 690)
table(is.na(train_1$GarageType))
table(is.na(train_1$GarageArea))
#Since all the houses have a garage area, 
#hence all the Na values in GarageType are actualy missing, more analysis needs to be done here
#regarding which values are missing and which are actually 'None'.

table(is.na(train_1$MSZoning))
#all values in MSZoning are present

#Basement features
table(is.na(train_1$BsmtExposure))
table(is.na(train_1$BsmtQual))
table(is.na(train_1$BsmtCond))
table(is.na(train_1$BsmtFinType1))


#38 values are missing or none in Basement exposure

#Changing the data types for MS Sub class, Month sold and year sold since they would be better 
#off as factor levels
train_1$MSSubClass <- as.factor(train_1$MSSubClass)
train_1$MoSold <- as.factor(train_1$MoSold)
train_1$YrSold <- as.factor(train_1$YrSold)
train_1$GarageType <- as.factor(train_1$GarageType)
train_1$GarageYrBlt <- as.factor(train_1$GarageYrBlt)
train_1$GarageFinish <- as.factor(train_1$GarageFinish)
train_1$GarageQual <- as.factor(train_1$GarageQual)
train_1$GarageCond <- as.factor(train_1$GarageCond)
train_1$BsmtExposure <- as.factor(train_1$BsmtExposure)
train_1$BsmtFinType1 <- as.factor(train_1$BsmtFinType1)
train_1$BsmtFinType2 <- as.factor(train_1$BsmtFinType2)
col <- c("BsmtQual", "BsmtCond", "MasVnrType", "Electrical", "MSZoning", 
         "Utilities", "Functional", "Exterior1st", "Exterior2nd", "KitchenQual","SaleType")
train_1[col] <- lapply(train_1[col], factor)

#Using the missmap function on Amelia package to get an overview of the data missing
missmap(train_1[,1:80],main = "Missing values in Housing Prices Dataset",
        col=c('red', 'steelblue'), y.cex=0.5, x.cex=0.8)

#Find the missing value counts using sapply
sort(sapply(train_1,function(x){sum(is.na(x))}), decreasing = T)

#From the above data, we find  PoolQC values are missing 
#most of their data. These may be lost causes.

#Forming a group of functions to be excluded and forming a group of included functions
exclude <- c('PoolQC')
include <- setdiff(names(train_1), exclude)

 #Forming a new set train_2 that include only those variables with lesser missing values
train_2 <- train_1[include]


#Now we start using the MICE package
#MICE - Multiple imputation by chained equations
#arguments = m (number of rounds of imputations, often multiple are done 
#and the results are pooled) , method used here is CART, but other methods can be used 
#and that too different methods can be used on different variables

imp.train_2 <- mice(train_2, m=1, method = 'cart', printFlag = F)

#Imputation process is done. Visualization can be done to check the quality of imputed data
#Lattice function works well with the daya type 'mids' retured by MICE
 
#Plotting Lot frontage vs Lot Area, imputed = Red, actual = Blue
xyplot(imp.train_2, LotFrontage ~ LotArea)

#Now we compare the distribution of imputed data when compared to the actual data
densityplot(imp.train_2, ~LotFrontage)
#the imputed data seems to replicate the actual data

#instead, if we had used means
imp.train_2_mean <- mice(train_2, m=1, defaultMethod = c('mean', 'cart', 'cart', 'cart'), printFlag = F)
xyplot(imp.train_2_mean, LotFrontage ~ LotArea)

#Lets look at the original and imputed garage data
table(train_2$GarageType)
table(imp.train_2$imp$GarageType)
#We observe that the although attached and detached form the majority in both,
#the ratios are different

#Similarly, looking at Garage finish
table(train_2$GarageFinish)
table(imp.train_2$imp$GarageFinish)
#Even here, the pattern seems to be the same, but the ratios are different

#Alternatives would be using oher methods of imputation or multiple rounds of imputation
#But if we are happy with the imputation results
train_3 <- complete(imp.train_2)
#Confirming no NA's
sum(sapply(train_3, function(x){sum(is.na(x))}))
str(train_3)
sort(sapply(train_3,function(x){sum(is.na(x))}), decreasing = T)

train_x=train_3[1:ntrain,]
train_x = cbind(train_x,Sale)
test_x=train_3[(ntrain+1):nrow(train_3),]

test_3 <- test[include]


#Points to note - missing data could just be a feature, like if LotFrontage is missing,
#there might actually not be a LotFrontage area for that property
#So, individual variables need to be considered before we perform imputation

train_x[] <- lapply(train_x, as.numeric)
dtrain=xgb.DMatrix(as.matrix(train_x),label= y_train)
test_x[] <- lapply(test_x, as.numeric)
dtest=xgb.DMatrix(as.matrix(test_x))

#XGBOOST parameters
xgb_params = list(
  seed = 0,
  colsample_bytree = 0.5,
  subsample = 0.8,
  eta = 0.02, 
  objective = 'reg:linear',
  max_depth = 12,
  alpha = 1,
  gamma = 2,
  min_child_weight = 1,
  base_score = 7.76
)

#XG Evaluation Function 
xg_eval_mae <- function (yhat, dtrain) {
  y = getinfo(dtrain, "label")
  err= mae(exp(y),exp(yhat) )
  return (list(metric = "error", value = err))
}

best_n_rounds=150 # try more rounds

#Train the data
gb_dt=xgb.train(xgb_params,dtrain,nrounds = as.integer(best_n_rounds))
Submission=fread("sample_submission.csv",colClasses = c("integer","numeric"))
Submission$SalePrice=predict(gb_dt,dtest)
write.csv(Submission,"xgb.csv",row.names = FALSE)


#Using Random Forest


install.packages("randomForest")
library(randomForest)
train_x$GarageYrBlt <- as.numeric(train_x$GarageYrBlt)
sort(sapply(train_x,function(x){sum(is.na(x))}), decreasing = T)
str(train_x)
train_x[sapply(train_x, is.character)] <- lapply(train_x[sapply(train_x, is.character)], as.factor)
model_RF <- randomForest(train_x$Sale ~., data = train_x, method = "anova",
                      ntree = 300,
                      mtry = 26,
                      replace = F,
                      nodesize = 1,
                      importance = T)
predict <- predict(model_RF, test_x)
Submission=fread("sample_submission.csv",colClasses = c("integer","numeric"))
Submission$SalePrice=predict(model_RF,test_x)
write.csv(Submission,"RF.csv",row.names = FALSE)

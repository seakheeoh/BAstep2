library(caTools)
library(dplyr)
library(doParallel)
library(xgboost)
library(caret)
library(fastDummies)
library(SHAPforxgboost)

#####################################################
#Data preparation for XGBoost-based prediction models
dataset <- read.csv(file = '   ', header = T)

#Splitting dataset
set.seed(2000)
split <- sample.split (dataset$BA, SplitRatio = 0.7)
train_set <- subset(dataset, split == TRUE)
test_set <- subset(dataset, split == FALSE)

#Building matrix 
train.y <- as.matrix(train_set[,1])
train.x <- as.matrix(train_set[,2:17])
test.y <- as.matrix(test_set[,1])
test.x <- as.matrix(test_set[,2:17])

#####################################################
#Initial XGBoost modelling
#Setting initial parameters
parameters <- list(eta= 0.3,
                   max_depth = 6,
                   subsample = 1,
                   colsample_bytree = 1,
                   min_child_weight = 1,
                   gamma = 0,
                   set.seed = 2000,
                   eval_metric = "auc",
                   objective = "binary:logistic",
                   booster = "gbtree") 

#Detecting cores
detectCores()

#Running initial XGBoost
model <- xgboost(data = train.x,
                 label = train.y,
                 nthread = 6,
                 nrounds = 100,
                 params = parameters,
                 print_every_n = 50,
                 early_stopping_rounds = 10)

#Initial evaluation of performances
m1_pred <- predict(model, newdata = test.x)
test_set$m1_pred <- cbind(m1_pred)
val_m1 <- val.prob(test_set$m1_pred, test_set$BA,
                   pl = FALSE) %>% round(3)
pander(val_m1)

#####################################################
#Hyperparameter optimizations, part 1
##Starting parallel processing 
cpu <- makeCluster(6)
registerDoParallel(cpu)

#Stating inputs
y <- as.factor(dataset[,1])
X <- as.matrix(dataset[,2:17])

#Stating the CV parameters
tune_control <- trainControl(method = "CV",
                             allowParallel = TRUE,
                             number = 10)
#Setting the parameters
tune_grid <- expand.grid(nrounds = seq(from  =50, to = 600, by = 50),
                         eta = c(0.1, 0.2, 0.3, 0.4),
                         max_depth = seq(2, 10, by = 2),
                         subsample = c(0.5, 0.7, 1),
                         colsample_bytree = 1,
                         min_child_weight =1,
                         gamma = 0)

#Cross validation and parameter tuning
start <- Sys.time()
xgb_tune <- train (x = X, 
                   y = y,
                   method ="xgbTree",
                   trControl = tune_control,
                   tuneGrid = tune_grid)
end <- Sys.time()

#See the best parameters
xgb_tune$bestTune
View(xgb_tune$results)

#####################################################
#Hyperparameter optimizations, part 2
cpu <- makeCluster(6)
registerDoParallel(cpu)

#Setting the parameters part 2
tune_grid2 <- expand.grid(nrounds = seq(from  =50, to = 600, by = 50),
                          eta = xgb_tune$bestTune$eta,
                          max_depth = xgb_tune$bestTune$max_depth,
                          subsample = xgb_tune$bestTune$subsample,
                          colsample_bytree = c(0.5, 0.7, 1),
                          min_child_weight = seq(1, 6, by =2),
                          gamma = c(0, 0.05, 0.1, 0.15))

#Cross validation and parameter tuning
start <- Sys.time()
xgb_tune2 <- train (x = X, 
                    y = y,
                    method ="xgbTree",
                    trControl = tune_control,
                    tuneGrid = tune_grid2)
end <- Sys.time()

#See the best parameters
xgb_tune2$bestTune
View(xgb_tune2$results)

############################################################
#Finalizing models
#Stating parameters
parameters3 <- list(eta= xgb_tune2$bestTune$eta,
                    max_depth = xgb_tune2$bestTune$max_depth,
                    subsample = xgb_tune2$bestTune$subsample,
                    colsample_bytree = xgb_tune2$bestTune$colsample_bytree,
                    min_child_weight = xgb_tune2$bestTune$min_child_weight,
                    gamma = xgb_tune2$bestTune$gamma,
                    set.seed = 1502,
                    eval_metric = "auc",
                    objective = "binary:logistic",
                    booster = "gbtree") 

#run XGBoost
model3 <- xgboost(data = train.x,
                  label = train.y,
                  sed.seed = 1502,
                  nthread = 6,
                  nrounds = xgb_tune2$bestTune$nrounds,
                  params = parameters,
                  print_every_n = 50,
                  early_stopping_rounds = 10)

#Evaluation of performances in testing set
m1_pred <- predict(model3, newdata = test.x)
test_set$m1_pred <- cbind(m1_pred)
val_m1 <- val.prob(test_set$m1_pred, test_set$BA,
                   pl = FALSE) %>% round(3)
pander(val_m1)



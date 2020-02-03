# Data Wrangling
library(data.table)
library(dplyr)
library(tidyverse)
library(skimr)

# Plotting
library(GGally)

# Feature Engineering
library(recipes)

# Resampling & Modeling
library(rsample)
library(caret)
library(glmnet)
library(rpart)
library(ipred)

# Parallel
library(foreach)
library(doParallel)

# Utility
library(here)

# Global & Theme
theme_set(theme_light())

# Theme Overrides
theme_update(axis.text.x = element_text(size = 10),
             axis.text.y = element_text(size = 10),
             plot.title = element_text(hjust = 0.5, size = 16, face = "bold", color = "darkgreen"),
             axis.title = element_text(face = "bold", size = 12, colour = "steelblue4"),
             plot.subtitle = element_text(face = "bold", size = 8, colour = "darkred"),
             legend.title = element_text(size = 12, color = "darkred", face = "bold"),
             legend.position = "right", legend.title.align=0.5,
             panel.border = element_rect(linetype = "solid", 
                                         colour = "lightgray"), 
             plot.margin = unit(c( 0.1, 0.1, 0.1, 0.1), "inches"))

# Utility

create_submission <- function(name, model, data) {
  pred <- predict(object = model, newdata = data, type = "prob")
  
  results <- data.table(PassengerId = data$PassengerId, Survived = ifelse(pred$yes > pred$no, 1, 0))
  
  file <- file.path(submission.dir, paste0(name, "_submission.csv"))
  
  data.table::fwrite(results, file)
  
  print(paste0("Created file:", file))
}

# data sets
project <- 'titanic'

data.dir <- file.path(here::here(), project, 'datasets'); submission.dir <- file.path(data.dir, "submission")

train <- data.table::fread(file.path(data.dir, "train.csv"), 
                           stringsAsFactors = T)

test <- data.table::fread(file.path(data.dir, "test.csv"),
                          stringsAsFactors = T)

# Initial Split

titanic.split <- initial_split(train, 
                               prop = .7,
                               strata = "Survived")

titanic.train <- training(titanic.split)
titanic.test <- testing(titanic.split)

# eda

train %>% skim()

ggplot(titanic.train, aes(Fare)) +
  geom_histogram(aes(fill = ..count..), bins = 30)

train[Fare > 3 * sd(Fare)]

ggplot(titanic.train, aes())

missing.age <- titanic.train[is.na(Age),]

nrow(titanic.train) / nrow(missing.age) # 5% missing age

missing.age[, .(Missing = .N, Pct = .N / nrow(missing.age)),
              by = Sex] # 70% male, 30% female

ggplot(titanic.train[Cabin != ""], aes(Cabin)) +
  geom_bar()

numeric.cols <- colnames(titanic.train)[sapply(titanic.train, is.numeric)]

ggpairs(titanic.train[, ..numeric.cols])

ggplot(titanic.train, aes(Survived, Age, group = Survived)) +
  geom_boxplot(aes(fill = Survived))

ggplot(titanic.train, aes(Embarked)) +
  geom_bar(aes(fill = ..count..))

# Data Recipe
recipe <- recipe(Survived ~ Sex + Age + Pclass + Cabin + Parch + Embarked + Fare + PassengerId, data = titanic.train) %>%
  step_meanimpute(Age, Fare) %>%
  step_dummy(all_nominal(), -all_numeric()) %>%
  step_nzv(all_predictors()) %>%
  step_bin2factor(Survived)

prep <- prep(recipe, data = titanic.train, strings_as_factors = T)

titanic.processed.train <- bake(prep, new_data = titanic.train)
titanic.processed.test <- bake(prep, new_data = titanic.test)

titanic.results <- bake(prep, new_data = test)

# Simple Logistic Regression

cv_glm <- caret::train(
      form = Survived ~ . -PassengerId,
      data = titanic.processed.train,
      method = "glm",
      family = "binomial",
      trControl = trainControl(method = "cv",
                               number = 10))

titanic.test.glm <- cbind(titanic.processed.test, 
                          predict(object = cv_glm, 
                                  newdata = titanic.processed.test, 
                                  type = "prob"))

vip::vip(cv_survived_glm, num_features = 10, geom = "point")

create_submission("ls", cv_survived_glm, titanic.results)

# Regularized

X_train <- model.matrix(Survived ~ . -PassengerId, data = titanic.processed.train)
Y_train <- titanic.processed.train$Survived

cv_glmnet <- caret::train(
  x = X_train,
  y = Y_train,
  method = "glmnet",
  preProc = c("zv", "center", "scale"),
  trControl = trainControl(method = "cv", number = 10),
  tuneLength = 10
)

cv_glmnet$bestTune

# eval train

ggplot(cv_glmnet)

train_pred <- predict(cv_glmnet, X_train)

confusionMatrix(Y_train, train_pred) # .8016

vip::vip(cv_glmnet, num_features = 20, bar = F)

# eval test
X_test <- model.matrix(Survived ~ . -PassengerId, data = titanic.processed.test)
Y_test <- titanic.processed.test$Survived

test_pred <- predict(cv_glmnet, X_test)

confusionMatrix(Y_test, test_pred)

create_submission("glmnet", cv_glmnet, titanic.results)

# Bagged Decision Tree

n.tree <- seq(1, 200, by = 2)
accuracy <- vector(mode = "numeric", length = length(n.tree))

for(i in seq_along(n.tree))
{
  set.seed(123)
  
  model <- ranger::ranger(
    formula = Survived ~ . -PassengerId,
    data = titanic.processed.train,
    num.trees = n.tree[i],
    mtry = ncol(titanic.processed.train) - 2,
    min.node.size = 1
  )
  
  accuracy[i] <- model$prediction.error
}

bagging.errors <- data.table(n.tree, accuracy)

min.error <- bagging.errors[which.min(bagging.errors$accuracy)]

ggplot(bagging.errors, aes(n.tree, accuracy)) +
  geom_line() +
  geom_vline(xintercept = min.error$n.tree, col = "darkred")

cv_survived_bag <- train(
  Survived ~ . -PassengerId,
  data = titanic.processed.train,
  trControl = trainControl(method = "cv",
                           number = 10),
  nbagg = 200,
  control = rpart.control(minsplit = 2, cp = 0)
)

cv_survived_bag$bestTune

vip::vip(cv_survived_bag, num_features = 10, geom = "point")

create_submission("baggedtree", cv_survived_bag, data = titanic.results)

# Cluster Train

cl <- makeCluster(8)

registerDoParallel(cl)

predictions <- foreach(
  icount(160),
  .packages = "rpart",
  .combine = cbind
) %dopar% {
  # bootstrap copy of training data
  index <- sample(nrow(titanic.processed.train), replace = T)
  titanic.train.boot <- titanic.processed.train[index, ]
  
  # fit tree to bootstrap copy
  bagged.tree <- rpart(
    Survived ~ . -PassengerId,
    control = rpart.control(minsplit = 2, cp = 0),
    data = titanic.train.boot
  )
  
  predict(bagged.tree, newdata = titanic.processed.test, type = "prob")
}

predictions

stopCluster(cl)

# clean-up

rm(list = ls())

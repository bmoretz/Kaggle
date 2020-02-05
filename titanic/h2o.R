library(tidyverse)
library(data.table)
library(h2o)
library(recipes)
library(here)
library(pROC)

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

# data sets
project <- 'titanic'

data.dir <- file.path(here::here(), project, "datasets"); submission.dir <- file.path(data.dir, "submission")

train <- data.table::fread(file.path(data.dir, "train.csv"), 
                           stringsAsFactors = T)

train$Survived <- as.factor(train$Survived)
train[, PassengerId := NULL]

test <- data.table::fread(file.path(data.dir, "test.csv"),
                          stringsAsFactors = T)

# Init h2o

h2o.init(nthreads=-1, max_mem_size = "32gb")

# Recipe

blueprint <- recipe(Survived ~ ., data = train) %>%
  step_other(all_nominal(), threshold = 0.005) %>%
  step_dummy(Ticket)

train_h2o <- prep(blueprint, training = train, retain = T) %>%
  juice() %>%
  as.h2o()

test_h2o <- prep(blueprint, training = train) %>%
  bake(new_data = test) %>%
  as.h2o()

# Features
excluded <- c("Name")

response <- "Survived"
predictors <- setdiff(setdiff(colnames(train_h2o), response), excluded)

n_predictors <- length(predictors)

#=================================
#  Utilities
#=================================

export_results <- function(type, h2o_model) {
  #' Generated a submission file from a h2o model.
  #'
  #' Generates predictions from model &
  #'
  #' @param data.sheet the information data sheet (excel file)  
  model_pred <- h2o.predict(h2o_model, test_h2o, type = "prob") %>% as.data.table()
  
  results <- data.table(PassengerId = test$PassengerId,
                        Survived = model_pred %>% pull(predict))
  
  submission_path <- file.path(submission.dir, paste0("h2o.", type ,".submission.csv"))
  
  data.table::fwrite(results, submission_path)
  
  print(paste0("Created: ", submission_path))
}

results_cross_validation <- function(h2o_model) {
  h2o_model@model$cross_validation_metrics_summary %>% 
    as.data.frame() %>% 
    select(-mean, -sd) %>% 
    t() %>% 
    as.data.frame() %>% 
    mutate_all(as.character) %>% 
    mutate_all(as.numeric) %>% 
    select(Accuracy = accuracy, 
           AUC = auc, 
           Precision = precision, 
           Specificity = specificity, 
           Recall = recall, 
           Logloss = logloss) %>% 
    return()
}

plot_results <- function(df_results) {
  df_results %>% 
    gather(Metrics, Values) %>% 
    ggplot(aes(Metrics, Values, fill = Metrics, color = Metrics)) +
    geom_boxplot(alpha = 0.3, show.legend = FALSE) + 
    theme(plot.margin = unit(c(1, 1, 1, 1), "cm")) +    
    scale_y_continuous(labels = scales::percent) + 
    facet_wrap(~ Metrics, scales = "free") + 
    labs(title = "Model Performance by Some Criteria Selected", y = NULL)
}

#=================================
#  Generalized Linear Model
#=================================

h2o_glm <- h2o.glm(
  x = predictors,
  y = response,
  training_frame = train_h2o,
  family = "binomial",
  lambda_search = T,
  alpha = .5,
  nfolds = 10,
  seed = 123
)

# GLM, Diag

summary(h2o_glm)

h2o_glm@model$training_metrics@metrics$AUC

h2o.varimp_plot(h2o_glm)

# Submission

export_results("glm", h2o_glm) # glm submission, 0.75119

#=================================
#  Random Forest
#=================================

as.numeric.factor <- function(x) as.numeric(levels(x))[x]

test_roc <- function(model_selected) {
  actual <- test_h2o$Survived %>% as.data.table() %>% pull(Survived) %>% as.numeric.factor()
  pred_prob <- h2o.predict(model_selected, test_h2o) %>% as.data.table() %>% pull(predict) %>% as.numeric.factor()
  
  return(roc(actual, pred_prob))
}

test_confusion <- function(model_selected) {
  actual <- test_h2o %>% as.data.table() %>% pull(Survived) %>% as.factor()
  pred <- h2o.predict(model_selected, newdata = test_h2o) %>% as.data.table() %>% pull(predict) %>% as.factor()
  confusionMatrix(pred, actual)
}

test_performance <- function(model_selected) {
  return(list(roc = test_roc(model_selected), cm = test_confusion(model_selected)))  
}

plot_performance <- function(perf) {
  
  print(perf)
  
  sen_spec_df <- tibble(TPR = perf$roc$sensitivities, FPR = 1 - perf$roc$specificities)
  
  sen_spec_df %>% 
    ggplot(aes(x = FPR, ymin = 0, ymax = TPR))+
    geom_polygon(aes(y = TPR), fill = "red", alpha = 0.3)+
    geom_path(aes(y = TPR), col = "firebrick", size = 1.2) +
    geom_abline(intercept = 0, slope = 1, color = "gray37", size = 1, linetype = "dashed") + 
    theme_bw() +
    coord_equal() +
    labs(x = "FPR (1 - Specificity)", 
         y = "TPR (Sensitivity)", 
         title = "Model Performance for RF Classifier based on Test Data", 
         subtitle = paste0("AUC Value: ", rf.auc$auc %>% round(2)))
}

#############################################
# Base Random Forest
#############################################

rf_base <- h2o.randomForest(
  x = predictors,
  y = response,
  training_frame = train_h2o,
  ntrees = n_predictors * 20,
  nfolds = 10,
  stopping_rounds = 5,
  stopping_tolerance = 0.001,
  stopping_metric = "logloss",
  balance_classes = F,
  score_each_iteration = T,
  seed = 123
)

# RF, Diag

summary(rf_base)

h2o.varimp_plot(rf_base) # vip

# get CV results
results_cross_validation(rf_base) %>%
  plot_results() +
  labs(subtitle = "Model: Random Forest (Base)")

# performance on test data
test_performance(rf_base) %>%
  plot_performance()

#=================================
#  Full Cartesian Grid Search
#=================================

# Set hyperparameter grid: 

hyper_grid.h2o <- list(ntrees = seq(50, 500, by = 50),
                       mtries = seq(3, 5, by = 1),
                       #max_depth = seq(10, 30, by = 10),
                       #min_rows = seq(1, 3, by = 1),
                       nbins = seq(20, 30, by = 10),
                       sample_rate = c(0.55, 0.632, 0.75))

# The number of models is:
sapply(hyper_grid.h2o, length) %>% prod()

# Train 6000 Random Forest Models: 
system.time(grid_cartesian <- h2o.grid(algorithm = "randomForest",
                                       grid_id = "rf_grid1",
                                       x = predictors, 
                                       y = response, 
                                       seed = 123, 
                                       nfolds = 10, 
                                       training_frame = frame.train,
                                       stopping_metric = "logloss", 
                                       hyper_params = hyper_grid.h2o,
                                       search_criteria = list(strategy = "Cartesian")))

# Collect the results and sort by our model performance metric of choice:
grid_perf <- h2o.getGrid(grid_id = "rf_grid1", 
                         sort_by = "logloss", 
                         decreasing = FALSE)

# Best model chosen by validation error: 
rf_cartesian_best <- h2o.getModel(grid_perf@model_ids[[1]])

summary(rf_cartesian_best)

h2o.varimp_plot(rf_cartesian_best) # vip

# get CV results
results_cross_validation(rf_cartesian_best) %>%
  plot_results() +
  labs(subtitle = "Model: Random Forest (Best Cartesian)")

# performance on test data
test_performance(rf_cartesian_best) %>%
  plot_performance()

#=================================
#  Random Discrete Grid Search
#=================================

# Set random grid search criteria: 
search_criteria_2 <- list(strategy = "RandomDiscrete",
                          stopping_metric = "logloss",
                          stopping_tolerance = 0.005,
                          stopping_rounds = 10,
                          max_runtime_secs = 30*60)

# Turn parameters for RF: 
system.time(random_grid <- h2o.grid(algorithm = "randomForest",
                                    grid_id = "rf_grid2",
                                    x = predictors, 
                                    y = response, 
                                    seed = 29, 
                                    nfolds = 10, 
                                    training_frame = frame.train,
                                    hyper_params = hyper_grid.h2o,
                                    search_criteria = search_criteria_2))

# Collect the results and sort by our models: 
grid_perf2 <- h2o.getGrid(grid_id = "rf_grid2", 
                          sort_by = "logloss", 
                          decreasing = FALSE)

# Best RF:
rf_random_best <- h2o.getModel(grid_perf2@model_ids[[1]])

summary(rf_random_best)

h2o.varimp_plot(rf_random_best) # vip

# get CV results
results_cross_validation(rf_random_best) %>%
  plot_results() +
  labs(subtitle = "Model: Random Forest (Best Random)")

# performance on test data
test_performance(rf_random_best) %>%
  plot_performance()

# RF, submission

rf_pred_std <- h2o.predict(h2o.rf, test.h2o, type = "prob") # std
rf_pred_cart <- h2o.predict(best_model, test.h2o, type = "prob") # tuned cartesian
rf_pred_rand <- h2o.predict(best_model2, test.h2o, type = "prob") # tuned random
  
h2o.rf.results <- data.table(PassengerId = test$PassengerId,
                              Survived = as.vector(round(rf.pred.rand$predict)))

data.table::fwrite(h2o.glm.results, file = file.path(submission.dir, "h2o.rf.submission.csv"))


#=================================
#  Auto ML
#=================================

# Use AutoML to find a list of candidate models (i.e., leaderboard)

auto_ml <- h2o.automl(
  x = predictors, 
  y = response, 
  training_frame = train_h2o, 
  nfolds = 5, 
  max_runtime_secs = 60 * 60 * 8, # Run for 8 hours
  max_models = 50,
  keep_cross_validation_predictions = TRUE, 
  sort_metric = "logloss", 
  seed = 123,
  stopping_rounds = 50, 
  stopping_metric = "logloss",
  stopping_tolerance = 0
)

# Assess the leader board; the following truncates the results to show the top 
# 25 models. You can get the top model with auto_ml@leader
auto_ml@leaderboard %>% 
  as.data.frame() %>%
  dplyr::select(model_id, auc, logloss, aucpr) %>%
  dplyr::slice(1:25)

h2o_top_auto <- auto_ml@leader

summary(h2o_top_auto)

h2o.varimp_plot(h2o_top_auto) # vip

# get CV results
results_cross_validation(h2o_top_auto) %>%
  plot_results() +
  labs(subtitle = "Model: Auto ML top(1)")

# performance on test data
test_performance(h2o_top_auto) %>%
  plot_performance()

# get submission
export_results("auto", h2o_top_auto)

### All done, shutdown H2O    
h2o.shutdown(prompt=FALSE)

rm(list = ls())

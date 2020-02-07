library(tidyverse)
library(data.table)
library(caret)
library(h2o)
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

# Init h2o
h2o.init(nthreads=-1, max_mem_size = "32gb")

# Init Project
project <- 'titanic'
source(file.path(project, "blueprint.R"))

data <- data_raw() %>%
  data_preprocessed() %>%
  data_processed()

splits <- data$train %>% 
  as.h2o() %>%
  h2o.splitFrame(. , ratios = .8, seed = 12345)

train_h2o <- splits[[1]]
test_h2o <- splits[[2]]

competition_h2o <- data$test %>%
  as.h2o()

excluded <- c("Name", "PassengerId")

response <- "Survived"
predictors <- setdiff(setdiff(colnames(train_h2o), response), excluded)

n_predictors <- length(predictors)

#=================================
#  Utilities
#=================================

export_results <- function(type, h2o_model) {
  #' Generated a submission file from a h2o model.
  #'
  #' @param type model type, used to distinguish submission types.
  #' @param h2o_model h2o model.
  model_pred <- h2o.predict(h2o_model, competition_h2o, type = "prob") %>% as.data.table()
  
  results <- data.table(PassengerId = test$PassengerId,
                        Survived = model_pred %>% pull(predict))
  
  submission.dir <- file.path(here(), project, "submissions")
  submission_path <- file.path(submission.dir, paste0("h2o.", type ,".submission.csv"))
  
  data.table::fwrite(results, submission_path)
  
  print(paste0("Created: ", submission_path))
}

results_cross_validation <- function(h2o_model) {
  #' Get cross-validation results from model.
  #'
  #' @param h2o_model h2o model.
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
  #' Get cross-validation results from model.
  #'
  #' @param h2o_model h2o model.  
  df_results %>% 
    gather(Metrics, Values) %>% 
    ggplot(aes(Metrics, Values, fill = Metrics, color = Metrics)) +
    geom_boxplot(alpha = 0.3, show.legend = FALSE) + 
    theme(plot.margin = unit(c(1, 1, 1, 1), "cm")) +    
    scale_y_continuous(labels = scales::percent) + 
    facet_wrap(~ Metrics, scales = "free") + 
    labs(title = "Model Performance by Criteria Selected", y = NULL)
}

as.numeric.factor <- function(x) as.numeric(levels(x))[x]

test_roc <- function(model_selected) {
  #' Get test frame ROC results for model.
  #'
  #' @param h2o_model h2o model.
  actual <- test_h2o$Survived %>% as.data.table() %>% pull(Survived) %>% as.numeric.factor()
  pred_prob <- h2o.predict(model_selected, test_h2o) %>% as.data.table() %>% pull(predict) %>% as.numeric.factor()
  
  return(roc(actual, pred_prob))
}

test_confusion <- function(model_selected) {
  #' Get test frame confusion matrix for model.
  #'
  #' @param h2o_model h2o model.  
  actual <- test_h2o %>% as.data.table() %>% pull(Survived) %>% as.factor()
  pred <- h2o.predict(model_selected, newdata = test_h2o) %>% as.data.table() %>% pull(predict) %>% as.factor()
  confusionMatrix(pred, actual)
}

test_performance <- function(model_selected) {
  #' Get test performance (ROC/CM)
  #'
  #' @param h2o_model h2o model.
  return(list(roc = test_roc(model_selected), cm = test_confusion(model_selected)))  
}

plot_performance <- function(perf) {
  #' Plot Reciever Operating Curve & print perf confusion matrix.
  #'
  #' @param h2o_model h2o model. 
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
         title = "Model Performance for Classifier based on Test Data", 
         subtitle = paste0("AUC Value: ", perf$roc$auc %>% round(2)))
}

test_get_auc <- function(selected_model) {
  results <- h2o.performance(selected_model, newdata = test_h2o)
  results@metrics$AUC
}

test_get_logloss <- function(selected_model) {
  results <- h2o.performance(selected_model, newdata = test_h2o)
  results@metrics$logloss
}

#=================================
#  Generalized Linear Model
#=================================

glm_hyper_params <- list(
  lambda = seq(0, 1, 0.005)
  #alpha = seq(0, .4, 0.005)
)

glm_search_criteria <- list(
  strategy = "Cartesian"
)

sapply(glm_hyper_params, length) %>% prod()

glm_grid_id <- "glm_grid_21"

system.time(glm_grid <- h2o.grid(
  algorithm = "glm",
  family = "binomial",
  grid_id = glm_grid_id,
  x = predictors,
  y = response,
  seed = 12345, 
  nfolds = 10,
  training_frame = train_h2o,
  hyper_params = glm_hyper_params,
  remove_collinear_columns = TRUE,
  fold_assignment = "Modulo",
  keep_cross_validation_predictions = TRUE,
  search_criteria = glm_search_criteria))

glm_grid_perf <- h2o.getGrid(grid_id = glm_grid_id, 
                          sort_by = "auc",
                          decreasing = T)

# Peek Models
glm_top_models <- glm_grid_perf@summary_table %>% 
  as.data.table() %>%
  head(25)

# Best GLM, ( alpha = .005 )

glm_model_id <- glm_grid_perf@model_ids[[1]]
glm_best <- h2o.getModel(glm_model_id)

# Model Diagnostics (Training Results)

summary(glm_best)

glm_best@model$training_metrics@metrics$logloss # train logloss: 0.4137314
glm_best@model$training_metrics@metrics$AUC # train AUC: 0.8776963

h2o.varimp_plot(glm_best, num_of_features = 25)

# Cross-Validated (Test Results)

results_cross_validation(glm_best) %>%
  plot_results() +
  labs(subtitle = "Model: Generalized Linear Model (Exhaustive Alpha)")

test_performance(glm_best) %>%
  plot_performance() # AUC = .834,

# Submission
export_results("glm", glm_best) # 0.76555

#=================================
#  Random Discrete Grid Search
#=================================

rf_grid_id <- "titanic_rf_grid_10"

rf_hyper_grid <- list(
  ntrees = seq(150, 350, by = 25),
  mtries = seq(1, 3, by = 1),
  max_depth = seq(5, 20, by = 5),
  min_rows = seq(1, 3, by = 1),
  nbins = seq(20, 30, by = 10),
  sample_rate = seq(.5, 1, by = .05))

# The number of models is:
sapply(rf_hyper_grid, length) %>% prod()

rf_search_criteria <- list(
  strategy = "RandomDiscrete",
  max_runtime_secs = 60 * 15)

system.time(random_grid <- h2o.grid(
  algorithm = "randomForest",
  grid_id = rf_grid_id,
  x = predictors, 
  y = response, 
  seed = 123456,
  nfolds = 10, 
  training_frame = train_h2o,
  hyper_params = rf_hyper_grid,
  search_criteria = rf_search_criteria,
  fold_assignment = "Modulo",
  keep_cross_validation_predictions = TRUE))

rf_grid_perf <- h2o.getGrid(
  grid_id = rf_grid_id, 
  sort_by = "auc", 
  decreasing = T)

# Peek Best Models
top_rf_models <- rf_grid_perf@summary_table %>%
  as.data.table() %>%
  top_n(50, auc)

# Best RF:
rf_best_id <- rf_grid_perf@model_ids[[1]]
rf_best <- h2o.getModel(rf_best_id)

summary(rf_best)

h2o.varimp_plot(rf_best)

# get CV results

results_cross_validation(rf_best) %>%
  plot_results() +
  labs(subtitle = "Model: Random Forest (Best Random)")

# performance on test data
test_performance(rf_best) %>%
  plot_performance()

# RF Submission

export_results("rf", rf_best) # 0.74162

#############################################
# Gradient Boosting Machine
#############################################

gbm_hyper_grid <- list(
  ntrees = seq(50, 500, by = 25),
  sample_rate = c(0.5, .675, 0.75),
  col_sample_rate = c(0.8, 0.9, 1),
  col_sample_rate_per_tree = c(0.75, 0.85, 1)
)

gbm_search_criteria <- list(
  strategy = "RandomDiscrete",
  stopping_metric = "auc",
  stopping_tolerance = 0.00001,
  stopping_rounds = 10,
  max_models = 50,
  max_runtime_secs = 60 * 10
)

gbm_grid_id <- "titanic_gbm_grid_3"

system.time(random_grid <- h2o.grid(
  algorithm = "gbm", 
  grid_id = gbm_grid_id, 
  x = predictors, 
  y = response,
  training_frame = train_h2o, 
  hyper_params = gbm_hyper_grid,
  search_criteria = gbm_search_criteria, 
  stopping_metric = "auc",
  stopping_rounds = 10, 
  stopping_tolerance = 0, 
  nfolds = 10, 
  fold_assignment = "Modulo", 
  keep_cross_validation_predictions = TRUE,
  seed = 12345
))

gbm_grid_perf <- h2o.getGrid(
  grid_id = gbm_grid_id,
  sort_by = "auc",
  decreasing = T
)

# Peek Models for Tuning Opportunities
gbm_grid_perf

gbm_best_id <- gbm_grid_perf@model_ids[[1]]
gbm_best <- h2o.getModel(gbm_random_best_id)

# GBM Perf

summary(gbm_best)

h2o.varimp_plot(gbm_best)

# get CV results
results_cross_validation(gbm_best) %>%
  plot_results() +
  labs(subtitle = "Model: GBM (Best Random)")

# performance on test data
test_performance(gbm_best) %>%
  plot_performance()

# GBM Submission 
export_results("gbm", gbm_best) # 0.78468

#=================================
# XG Boost
#=================================

xgb_hyper_grid <- list(
  learn_rate = c(0.01, 0.02, 0.05),
  sample_rate = seq(0.7, 1, .05),
  ntrees =  seq(50, 500, by = 25),
  col_sample_rate = seq(0.75, 1, 0.05),
  col_sample_rate_per_tree = seq(0.75, 1, 0.05),
  max_depth = seq(2, 5, by = 1), 
  min_rows = seq(2, 5, by = 1)
)

xgb_search_criteria <- list(
  strategy = "RandomDiscrete",
  stopping_metric = "auc",
  stopping_tolerance = 0.00001,
  stopping_rounds = 10,
  max_models = 25,
  max_runtime_secs = 60 * 10
)

xgb_grid_id <- "titanic_xgb_grid_3"

system.time(xgb_random_grid <- h2o.grid(
  algorithm = "xgboost", 
  grid_id = xgb_grid_id, 
  x = predictors, 
  y = response,
  training_frame = train_h2o, 
  hyper_params = xgb_hyper_grid,
  search_criteria = xgb_search_criteria, 
  stopping_metric = "auc",
  stopping_rounds = 10, 
  stopping_tolerance = 0, 
  nfolds = 10,
  fold_assignment = "Modulo", 
  keep_cross_validation_predictions = TRUE,
  seed = 12345
))

xgb_grid_perf <- h2o.getGrid(
  grid_id = xgb_grid_id,
  sort_by = "auc",
  decreasing = T
)

xgb_grid_perf

xgb_best_id <- xgb_grid_perf@model_ids[[1]]
xgb_best <- h2o.getModel(xgb_best_id)

# Model Diagnostics (Train Error)

summary(xgb_best)

h2o.varimp_plot(xgb_best)

# CV Performance (Test Error)

results_cross_validation(xgb_best) %>%
  plot_results() +
  labs(subtitle = "Model: XGB (Best Random)")

test_performance(xgb_best) %>%
  plot_performance()

export_results("xgb", xgb_best) # 0.78468

#=================================
#  Stacked
#=================================

ensemble_tree_id = "titanic_tree_ensemble_01"

best_models <- list(glm_best,
                    rf_best,
                    gbm_best,
                    xgb_best)

best_models %>%
  purrr::map_dbl(test_get_auc)

best_models %>%
  purrr::map_dbl(test_get_logloss)

# Train a stacked tree ensemble
ensemble_tree <- h2o.stackedEnsemble(
  x = predictors, 
  y = response, 
  training_frame = train_h2o,
  validation_frame = test_h2o,
  model_id = ensemble_tree_id,
  base_models = best_models,
  metalearner_algorithm = "drf"
)

h2o.performance(ensemble_tree, newdata = test_h2o)@metrics$logloss
h2o.performance(ensemble_tree, newdata = test_h2o)@metrics$AUC

# CV Performance (Test Error)

test_performance(ensemble_tree) %>%
  plot_performance()

export_results("stacked", ensemble_tree) # 0.76076

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
export_results("auto", h2o_top_auto) # .76076

### All done, shutdown H2O    
h2o.shutdown(prompt=FALSE)

rm(list = ls())

library(tidyverse)
library(data.table)
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

train_h2o <- data$train %>% 
  as.h2o()

test_h2o <- data$test %>%
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
  model_pred <- h2o.predict(h2o_model, test_h2o, type = "prob") %>% as.data.table()
  
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

#=================================
#  Generalized Linear Model
#=================================

glm_hyper_params <- list( 
  alpha = seq(0, 1, 0.01)
)

glm_search_criteria <- list(
  strategy = "Cartesian"
)

glm_grid_id <- "glm_grid_6"

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
  search_criteria = glm_search_criteria))

glm_grid_perf <- h2o.getGrid(grid_id = glm_grid_id, 
                          sort_by = "auc",
                          decreasing = FALSE)

# Best GLM, ( lambda = 0 )

glm_model_id <- glm_grid_perf@model_ids[[1]]
glm_best <- h2o.getModel(glm_model_id)

# GLM, Diag

summary(glm_best)

glm_best@model$training_metrics@metrics$logloss
glm_best@model$training_metrics@metrics$AUC

h2o.varimp_plot(glm_best, num_of_features = 20)

# get CV results
results_cross_validation(glm_best) %>%
  plot_results() +
  labs(subtitle = "Model: Generalized Linear Model (Exhaustive Lambda)")

# Submission

export_results("glm", glm_best) # glm submission, 0.75598

#=================================
#  Random Discrete Grid Search
#=================================

rf_grid_id <- "titanic_rf_grid_6"

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
  stopping_metric = "logloss",
  stopping_tolerance = 0.005,
  stopping_rounds = 10,
  max_models = 25,
  max_runtime_secs = 60 * 30)

system.time(random_grid <- h2o.grid(
  algorithm = "randomForest",
  grid_id = rf_grid_id,
  x = predictors, 
  y = response, 
  seed = 123456, 
  nfolds = 10, 
  training_frame = train_h2o,
  hyper_params = rf_hyper_grid,
  search_criteria = rf_search_criteria))

# Collect the results and sort by our models: 
rf_grid_perf <- h2o.getGrid(
  grid_id = rf_grid_id, 
  sort_by = "logloss", 
  decreasing = FALSE)

# Best RF:
rf_best <- h2o.getModel(rf_grid_perf@model_ids[[1]])

summary(rf_best)

h2o.varimp_plot(rf_best)

# get CV results
results_cross_validation(rf_best) %>%
  plot_results() +
  labs(subtitle = "Model: Random Forest (Best Random)")

# performance on test data
test_performance(rf_best) %>%
  plot_performance()

# RF, submission

export_results("rf", rf_best)

#############################################
# Gradient Boosting Machine
#############################################

# Define GBM hyperparameter grid
hyper_grid <- list(
  #learning_rate = c(0.01, 0.02, 0.05),
  #sample_rate = c(0.5, 0.75, 1),
  ntrees = c(2500, 3000, 5000),
  sample_rate = c(.9, 1),
  col_sample_rate = c(0.8, 0.9, 1),
  col_sample_rate_per_tree = c(0.75, 0.85, 1)
)

# Define random grid search criteria
search_criteria <- list(
  strategy = "RandomDiscrete",
  stopping_metric = "logloss",
  stopping_tolerance = 0.001,
  stopping_rounds = 10,
  max_runtime_secs = 60 * 10
)

gbm_grid_id <- "titanic_gbm_grid_2"

# Build random grid search 
random_grid <- h2o.grid(
  algorithm = "gbm", 
  grid_id = gbm_grid_id, 
  x = predictors, 
  y = response,
  training_frame = train_h2o, 
  hyper_params = hyper_grid,
  search_criteria = search_criteria, 
  stopping_metric = "logloss",
  stopping_rounds = 10, 
  stopping_tolerance = 0, 
  nfolds = 10, 
  fold_assignment = "Modulo", 
  keep_cross_validation_predictions = TRUE,
  seed = 123
)

gbm_grid_perf <- h2o.getGrid(
  grid_id = gbm_grid_id,
  sort_by = "logloss",
  decreasing = F
)

gbm_random_best_id <- gbm_grid_perf@model_ids[[1]]
gbm_random_best <- h2o.getModel(gbm_random_best_id)

# GBM Perf

summary(gbm_random_best)

h2o.varimp_plot(gbm_random_best) # vip

# get CV results
results_cross_validation(gbm_random_best) %>%
  plot_results() +
  labs(subtitle = "Model: GBM (Best Random)")

# performance on test data
test_performance(rf_random_best) %>%
  plot_performance()

export_results("gbm", gbm_random_best)

#=================================
# XG Boost
#=================================

xgb_hyper_grid <- list(
  # learning_rate = c(0.01, 0.02, 0.05),
  sample_rate = c(0.5, 0.75, 1),
  ntrees = c(10000),
  col_sample_rate = seq(0.75, 1, 0.05),
  col_sample_rate_per_tree = seq(0.75, 1, 0.05),
  max_depth = c(1, 2, 3, 4), 
  min_rows = c(1, 2, 3, 4)
)

xgb_search_criteria <- list(
  strategy = "RandomDiscrete",
  stopping_metric = "logloss",
  stopping_tolerance = 0.00001,
  stopping_rounds = 10,
  max_models = 25,
  max_runtime_secs = 60 * 1
)

xgb_grid_id <- "titanic_xgb_grid_1"

xgb_random_grid <- h2o.grid(
  algorithm = "xgboost", 
  grid_id = xgb_grid_id, 
  x = predictors, 
  y = response,
  training_frame = train_h2o, 
  hyper_params = xgb_hyper_grid,
  # max_depth = 3,
  # min_rows = 3,
  search_criteria = xgb_search_criteria, 
  stopping_metric = "logloss",
  stopping_rounds = 10, 
  stopping_tolerance = 0, 
  nfolds = 10,
  fold_assignment = "Modulo", 
  keep_cross_validation_predictions = TRUE,
  seed = 123
)

xgb_grid_perf <- h2o.getGrid(
  grid_id = xgb_grid_id,
  sort_by = "logloss",
  decreasing = F
)

xgb_grid_perf

xgb_random_best_id <- xgb_grid_perf@model_ids[[1]]
xgb_random_best <- h2o.getModel(xgb_random_best_id)

# XGB Perf

summary(xgb_random_best)

h2o.varimp_plot(xgb_random_best) # vip

# get CV results
results_cross_validation(xgb_random_best) %>%
  plot_results() +
  labs(subtitle = "Model: XGB (Best Random)")

# performance on test data
test_performance(rf_random_best) %>%
  plot_performance()

export_results("xgb", xgb_random_best) # 0.76555

#=================================
#  Stacked
#=================================

data.frame(
  GLM_pred = as.vector(h2o.getFrame(best_glm@model$cross_validation_holdout_predictions_frame_id$name)),
  RF_pred = as.vector(h2o.getFrame(best_rf@model$cross_validation_holdout_predictions_frame_id$name)),
  GBM_pred = as.vector(h2o.getFrame(best_gbm@model$cross_validation_holdout_predictions_frame_id$name)),
  XGB_pred = as.vector(h2o.getFrame(best_xgb@model$cross_validation_holdout_predictions_frame_id$name))
) %>% cor()

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

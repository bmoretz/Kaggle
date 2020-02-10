
#############################################
# Gradient Boosting Machine
#############################################

gbm_hyper_grid <- list(
  learn_rate = c(0.01, 0.03),
  ntrees = seq(200, 500, by = 25),
  sample_rate = c(0.5, .675, 0.75, 1),
  col_sample_rate = c(0.8, 0.9, 1),
  col_sample_rate_per_tree = c(0.85, 1),
  max_depth = seq(2, 5, by = 1), 
  min_rows = seq(2, 5, by = 1)  
)

gbm_search_criteria <- list(
  strategy = "RandomDiscrete",
  max_runtime_secs = 60 * 1
)

# number of total models
sapply(gbm_hyper_grid, length) %>% prod()

gbm_grid_id <- "titanic_gbm_grid_04"

system.time(random_grid <- h2o.grid(
  algorithm = "gbm",
  grid_id = gbm_grid_id, 
  x = predictors, 
  y = response,
  training_frame = train_h2o, 
  hyper_params = gbm_hyper_grid,
  search_criteria = gbm_search_criteria, 
  nfolds = 5, 
  fold_assignment = "Stratified",
  seed = 12345
))

gbm_grid_perf <- h2o.getGrid(
  grid_id = gbm_grid_id,
  sort_by = "auc",
  decreasing = T
)

# Peek Models for Tuning Opportunities
top_gbm_models <- gbm_grid_perf@summary_table %>%
  as.data.table() %>%
  top_n(50, auc)

rand_gbm_best_id <- gbm_grid_perf@model_ids[[1]]
rand_gbm_best <- h2o.getModel(rand_gbm_best_id)

summary(rand_gbm_best)
h2o.varimp_plot(rand_gbm_best)

results_cross_validation(rand_gbm_best) %>%
  plot_results() +
  labs(subtitle = "Model: GBM (Best)")

# performance on test data
test_performance(rand_gbm_best) %>%
  plot_performance()

#################################################################

best_gbm <- h2o.gbm(
  x = predictors,
  y = response,
  training_frame = train_h2o,
  # best params
  col_sample_rate = 1.0,
  col_sample_rate_per_tree = 1.0,
  learn_rate = 0.01,
  ntrees = 300,
  sample_rate = .75,
  # cv settings for stacked
  nfolds = 10,
  fold_assignment = "Modulo",
  keep_cross_validation_predictions = T,
  seed = 12345
)

# Best GBM

summary(best_gbm)
h2o.varimp_plot(best_gbm)

results_cross_validation(best_gbm) %>%
  plot_results() +
  labs(subtitle = "Model: GBM (Best)")

test_performance(best_gbm) %>%
  plot_performance() # 0.87

# GBM Submission 
export_results("gbm", best_gbm) # 0.77990

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
  max_runtime_secs = 60 * 30
)

xgb_grid_id <- "titanic_xgb_grid_01"

system.time(xgb_random_grid <- h2o.grid(
  algorithm = "xgboost", 
  grid_id = xgb_grid_id,
  x = predictors, 
  y = response,
  training_frame = train_h2o,
  hyper_params = xgb_hyper_grid,
  search_criteria = xgb_search_criteria,
  nfolds = 10,
  fold_assignment = "Modulo",
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

### All done, shutdown H2O    
h2o.shutdown(prompt=FALSE)

rm(list = ls())
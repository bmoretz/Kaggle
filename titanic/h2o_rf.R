library(tidyverse)
library(data.table)
library(h2o)
library(here)

h2o.init(nthreads=-1)

project <- 'titanic'
source(file.path(project, "blueprint.R"))
source(file.path(project, "h2o.R"))

set_global_theme()
load_titanic_data()

#=================================
#  Random Forest
#=================================

rf_runtime_minutes <- 5

rf_grid_id <- "titanic_rf_grid_01"

rf_hyper_grid <- list(
  ntrees = seq(150, 500, by = 25),
  mtries = seq(1, 5, by = 1),
  max_depth = seq(5, 20, by = 5),
  min_rows = seq(1, 3, by = 1),
  nbins = seq(10, 30, by = 10),
  sample_rate = seq(.5, 1, by = .05))

paste("Total Parameter Combinations: ", sapply(rf_hyper_grid, length) %>% prod())

rf_search_criteria <- list(
  strategy = "RandomDiscrete",
  max_runtime_secs = 60 * rf_runtime_minutes)

system.time(rf_grid <- h2o.grid(
  algorithm = "randomForest",
  grid_id = rf_grid_id,
  x = predictors_h2o, 
  y = response_h2o,
  training_frame = train_h2o,
  nfolds = 10,
  hyper_params = rf_hyper_grid,
  search_criteria = rf_search_criteria,
  fold_assignment = "Modulo",
  seed = 12345))

rf_grid_perf <- h2o.getGrid(
  grid_id = rf_grid_id,
  sort_by = "auc",
  decreasing = T)

# Peek Best Models

top_rf_models <- rf_grid_perf@summary_table %>%
  as.data.table() %>%
  top_n(50, auc)

rand_rf_best_id <- rf_grid_perf@model_ids[[1]]
rand_rf_best <- h2o.getModel(rand_rf_best_id)

summary(rand_rf_best)

h2o.varimp_plot(rand_rf_best)

# get CV results

results_cross_validation(rf_best) %>%
  plot_results() +
  labs(subtitle = "Model: Random Forest (Best Random)")

# performance on test data
test_performance(rand_rf_best) %>%
  plot_performance()


# RF Submission
export_results("rf", rf_best) # 0.74162


rm(list = ls())

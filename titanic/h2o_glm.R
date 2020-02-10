library(tidyverse)
library(data.table)
library(h2o)
library(here)

h2o.init(nthreads=-1, max_mem_size = "32gb")

project <- 'titanic'
source(file.path(project, "blueprint.R"))
source(file.path(project, "h2o.R"))

set_global_theme()
load_titanic_data()

#=================================
# Generalized Linear Model
#=================================

glm_run_time_minutes <- 15

glm_hyper_params <- list(
  alpha = seq(0, 1, 0.005),
  lambda = 10 ^ seq(10, -2, length.out = 100)
)

glm_search_criteria <- list(
  strategy = "RandomDiscrete",
  max_runtime_secs = 60 * glm_run_time_minutes
)

sapply(glm_hyper_params, length) %>% prod()

glm_grid_id <- "glm_grid_01"

system.time(glm_grid <- h2o.grid(
  grid_id = glm_grid_id,
  algorithm = "glm",
  family = "binomial",
  x = predictors_h2o,
  y = response_h2o,
  training_frame = train_h2o,
  validation_frame = valid_h2o,
  nfolds = 10,
  hyper_params = glm_hyper_params,
  remove_collinear_columns = TRUE,
  fold_assignment = "Modulo",
  search_criteria = glm_search_criteria,
  seed = 12345))

glm_grid_perf <- h2o.getGrid(grid_id = glm_grid_id, 
                             sort_by = "auc",
                             decreasing = T)

# Peek Models
glm_top_models <- glm_grid_perf@summary_table %>% 
  as.data.table() %>%
  head(25)

# Best GLM, ( alpha = 0, lambda = 141747.41629268 )

glm_model_id <- glm_grid_perf@model_ids[[1]]
glm_top_gs <- h2o.getModel(glm_model_id)

h2o.giniCoef(glm_top_gs)

summary(glm_top_gs)

test_performance(glm_top_gs) %>%
  plot_performance()

test_confusion(glm_top_gs) %>% 
  plot_confusion()

#=================================
# Best Generalized Linear Model
#=================================

# Keep Best Configuraion
best_glm_id = "best_glm_h2o"

best_glm <- h2o.glm(
  x = predictors_h2o,
  y = response_h2o,
  alpha = 0.005,
  lambda = 0,
  family = "binomial",
  training_frame = train_h2o,
  validation_frame = test_h2o,
  model_id = best_glm_id,
  remove_collinear_columns = T,
  fold_assignment = "Modulo",
  nfolds = 10,
  seed = 12345,
  keep_cross_validation_models = T,
  keep_cross_validation_fold_assignment = T
)

# Model Diagnostics (Training Results)

summary(best_glm)

best_glm@model$training_metrics@metrics$logloss # train logloss: 0.4137314
best_glm@model$training_metrics@metrics$AUC # train AUC: 0.8776963

h2o.varimp_plot(best_glm)

# Cross-Validated (Test Results)

results_cross_validation(best_glm) %>%
  plot_results() +
  labs(subtitle = "Model: Generalized Linear Model")

test_performance(best_glm) %>%
  plot_performance() # AUC = .834,

test_confusion(best_glm) %>%
  plot_confusion()

export_results("glm", best_glm) # 0.76555

### All done, shutdown H2O    
h2o.shutdown(prompt=FALSE)

rm(list = ls())

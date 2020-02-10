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
#  Auto ML
#=================================

auto_ml_runtime <- 60 # 60 * 60 * 8

auto_ml <- h2o.automl(
  x = predictors_h2o, 
  y = response_h2o, 
  training_frame = train_h2o, 
  nfolds = 5, 
  max_runtime_secs = auto_ml_runtime, # Run for 8 hours
  max_models = 50,
  keep_cross_validation_predictions = TRUE, 
  sort_metric = "logloss", 
  seed = 123,
  stopping_rounds = 50, 
  stopping_metric = "logloss",
  stopping_tolerance = 0
)

# top models
auto_ml@leaderboard %>% 
  as.data.frame() %>%
  dplyr::select(model_id, auc, logloss, aucpr) %>%
  dplyr::slice(1:25)

h2o_top_auto <- auto_ml@leader

summary(h2o_top_auto)

h2o.varimp_plot(h2o_top_auto)

# get CV results
results_cross_validation(h2o_top_auto) %>%
  plot_results() +
  labs(subtitle = "Model: Auto ML top(1)")

# performance on test data
test_performance(h2o_top_auto) %>%
  plot_performance()

# get submission
export_results("auto", h2o_top_auto) # .76076

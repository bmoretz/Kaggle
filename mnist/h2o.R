library(tidyverse)
library(data.table)
library(h2o)
library(here)

# Init h2o
h2o.init(nthreads=-1, max_mem_size = "32gb")

# Init Project
project <- 'mnist'
source(file.path(project, "blueprint.R"))

load_mnist_data <- function(split_ratio = .8) {
  
  data <- data_raw() %>%
    data_preprocessed() %>%
    data_processed()
  
  splits <- data$train %>% 
    as.h2o() %>%
    h2o.splitFrame(. , ratios = c(split_ratio), seed = 12345)
  
  train_h2o <<- splits[[1]]
  test_h2o <<- splits[[2]]
  
  competition_h2o <<- data$test %>%
    as.h2o()
  
  excluded <- c("Name", "PassengerId")
  
  response_h2o <<- "Survived"
  predictors_h2o <<- setdiff(setdiff(colnames(train_h2o), response_h2o), excluded)
  n_predictors_h2o <<- length(predictors_h2o)
}

#=================================
#  Utilities
#=================================

export_results <- function(type, h2o_model) {
  #' Generated a submission file from a h2o model.
  #'
  #' @param type model type, used to distinguish submission types.
  #' @param h2o_model h2o model.
  #' 
  predicted <- h2o.predict(h2o_model, competition_h2o, type = "prob") %>%
    as.data.table()
  
  submission_content <- predicted[, .(ImageId = .I, Label = predict)]
  submission_dir <- file.path(here(), project, "submissions")
  submission_path <- file.path(submission_dir, paste0("h2o.", type,".submission.csv"))
  
  data.table::fwrite(submission_content, submission_path)
  
  print(paste0("Created: ", submission_path))
}
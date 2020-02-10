library(tidyverse)
library(data.table)
library(h2o)
library(here)
library(pROC)

# Init h2o
h2o.init(nthreads=-1, max_mem_size = "32gb")

# Init Project
project <- 'titanic'
source(file.path(project, "blueprint.R"))

load_titanic_data <- function(split_ratio = .8) {

  data <- data_raw() %>%
    data_preprocessed() %>%
    data_processed()
  
  splits <- data$train %>% 
    as.h2o() %>%
    h2o.splitFrame(. , ratios = c(split_ratio - .1, 1 - split_ratio), seed = 12345)
  
  train_h2o <<- splits[[1]]
  valid_h2o <<- splits[[2]]
  test_h2o <<- splits[[3]]
  
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
  
  passenger_id <- competition_h2o %>% as.data.table() %>% pull(PassengerId)
  
  predicted <- h2o.predict(h2o_model, competition_h2o, type = "prob") %>%
    as.data.table() %>%
    add_column(passenger_id)
  
  submission_content <- predicted[, .(PassengerId = passenger_id, Survived = predict)]
  submission_dir <- file.path(here(), project, "submissions")
  submission_path <- file.path(submission_dir, paste0("h2o.", type,".submission.csv"))
  
  data.table::fwrite(submission_content, submission_path)
  
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
           Logloss = logloss)
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

test_roc <- function(model_selected) {
  #' Get test frame ROC results for model.
  #'
  #' @param h2o_model h2o model.
  actual <- test_h2o$Survived %>% as.data.table() %>% pull(Survived) %>% as.ordered()
  pred <- h2o.predict(model_selected, test_h2o) %>% as.data.table() %>% pull(predict) %>% as.ordered()
  
  return(roc(actual, pred))
}

test_confusion <- function(model_selected) {
  #' Get test frame confusion matrix for model.
  #'
  #' @param h2o_model h2o model.

  h2o.confusionMatrix(model_selected, test_h2o)
}

test_performance <- function(model_selected) {
  #' Get test performance (ROC/CM)
  #'
  #' @param h2o_model h2o model.
  return(list(roc = test_roc(model_selected), 
              cm = test_confusion(model_selected),
              metrics = test_h2o_perf(model_selected)))
}

test_h2o_perf <- function(selected_model) {
  #' Get test performance (ROC/CM)
  #'
  #' @param h2o_model h2o model.
  h2o.performance(selected_model, newdata = test_h2o)
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

plot_confusion <- function(cm) {
  #' Plot confusion matrix.
  #'
  #' @param h2o_model h2o model. 
  
  print(cm)
  
  cm <- as.data.frame(cm)
  
  trans <- .65
  
  ggplot() +
    geom_bar(aes(x = 0, y = cm[1, 1]), 
             stat = "identity", fill = "cornflowerblue", alpha = trans) +
    geom_text(aes(x = 0, y = cm[1, 1] - 1, 
                  label = cm[1, 1]), col = "white") +
    geom_bar(aes(x = 0, y = cm[1, 2]), 
             stat = "identity", fill = "darkred", alpha = trans) +
    geom_text(aes(x = 0, y = cm[1, 2] - 1, 
                  label = cm[1, 2]), col = "white") +
    geom_bar(aes(x = 1, y = cm[2, 2]), 
             stat = "identity", fill = "cornflowerblue", alpha = trans) +
    geom_text(aes(x = 1, y = cm[2, 2] - 1, 
                  label = cm[2, 2]), col = "white") +
    geom_bar(aes(x = 1, y = cm[2, 1]), 
             stat = "identity", fill = "darkred", alpha = trans) +
    geom_text(aes(x = 1, y = cm[2, 1] - 1, label = cm[2, 1]), col = "white") +
    labs(title = "Confusion Matrix", x = "Survived [T, F]", y = "T / F") +
    theme(
          axis.text.x=element_blank(),
          axis.ticks.x=element_blank(),
          axis.ticks.y=element_blank(),
          axis.text.y=element_blank())
}


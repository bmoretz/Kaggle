library(data.table)
library(h2o)

library(ggplot2)

# Utility
library(here)

# Metrics
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

train$Survived <- factor(train$Survived, levels = c(0, 1), labels = c(0, 1))

test <- data.table::fread(file.path(data.dir, "test.csv"),
                          stringsAsFactors = T)

# Init h2o

h2o.init(nthreads=-1, max_mem_size = "16gb")

train.h2o <- as.h2o(train)
test.h2o <- as.h2o(test)

# Train / Valid / Test

splits <- h2o.splitFrame(train.h2o, 
                         c(.7, .2),
                         seed = 1234)

frame.train <- h2o.assign(splits[[1]], "train.survive")
frame.valid <- h2o.assign(splits[[2]], "valid.survive")
frame.test <- h2o.assign(splits[[3]], "test.survive")

rm(splits)

# 
excluded <- c("PassengerId", "Name", "SibSp", "Ticket")

response <- "Survived"
predictors <- setdiff(setdiff(colnames(train), response), excluded)

n.predictors <- length(predictors)

# Generalized Linear Model

h2o.glm <- h2o.glm(
  x = predictors,
  y = response,
  training_frame = train.h2o,
  family = "binomial",
  lambda_search = T,
  alpha = .5,
  nfolds = 10,
  seed = 123
)

# GLM, Diag

summary(h2o.glm)

h2o.glm@model$training_metrics@metrics$AUC

h2o.varimp_plot(h2o.glm)

# glm submission

glm.pred <- h2o.predict(h2o.glm, test.h2o, type = "response")

h2o.glm.results <- data.table(PassengerId = test$PassengerId,
                              Survived = as.vector(glm.pred$predict))

data.table::fwrite(h2o.glm.results, file = file.path(submission.dir, "h2o.glm.submission.csv"))

# Random Forest  

h2o.rf <- h2o.randomForest(
  x = predictors,
  y = response,
  training_frame = frame.train,
  ntrees = n.predictors * 20,
  nfolds = 10,
  stopping_rounds = 5,
  stopping_tolerance = 0.001,
  stopping_metric = "AUC",
  balance_classes = F,
  score_each_iteration = T,
  fold_assignment = "Stratified",
  seed = 123
)

# RF, Diag

summary(h2o.rf)

h2o.varimp_plot(h2o.rf)

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

results_cross_validation(h2o.rf) -> rf_default

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

plot_results(rf_default) +
  labs(subtitle = "Model: Random Forest (h2o package)")

# Model performance based on test data: 
pred_class <- h2o.predict(h2o.rf, frame.test) %>% as.data.frame() %>% pull(predict)
confusionMatrix(pred_class, factor(as.vector(frame.test$Survived), levels = c(0, 1), labels = c(0, 1)))

# AUC

as.numeric.factor <- function(x) { as.numeric(levels(x))[x] }

auc_for_test <- function(model_selected) {
  actual <- frame.test$Survived %>% as.data.table() %>% pull(Survived) %>% as.numeric.factor()
  pred_prob <- h2o.predict(model_selected, frame.test) %>% as.data.table() %>% pull(predict) %>% as.numeric.factor()
  return(roc(actual, pred_prob))
}

rf.auc <- auc_for_test(h2o.rf)
rf.auc$auc

# Graph ROC and AUC: 

sen_spec_df <- tibble(TPR = rf.auc$sensitivities, FPR = 1 - rf.auc$specificities)

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

# Function for calculating CM: 

my_cm_com_rf <- function(thre) {
  du_bao_prob <- h2o.predict(h2o.rf, frame.test) %>% as.data.frame() %>% pull(predict) %>% as.factor()
  cm <- confusionMatrix(du_bao_prob, frame.test %>% as.data.table() %>% pull(Survived) %>% as.factor())
  return(cm)
}

# Set a range of threshold for classification: 
my_threshold <- c(0.10, 0.15, 0.35, 0.5)
results_list_rf <- lapply(my_threshold, my_cm_com_rf)

# Function for presenting prediction power by class:  

vis_detection_rate_rf <- function(x) {
  
  results_list_rf[[x]]$table %>% as.data.frame() -> m
  rate <- round(100*m$Freq[1] / sum(m$Freq[c(1, 2)]), 2)
  acc <- round(100*sum(m$Freq[c(1, 4)]) / sum(m$Freq), 2)
  acc <- paste0(acc, "%")
  
  m %>% 
    ggplot(aes(Reference, Freq, fill = Prediction)) +
    geom_col(position = "fill") + 
    scale_fill_manual(values = c("#e41a1c", "#377eb8"), name = "") + 
    theme(panel.grid.minor.y = element_blank()) + 
    theme(panel.grid.minor.x = element_blank()) + 
    scale_y_continuous(labels = scales::percent) + 
    labs(x = NULL, y = NULL, 
         title = paste0("Survivors when Threshold = ", my_threshold[x]), 
         subtitle = paste0("Detecting Rate for Survivors: ", rate, "%", ", ", "Accuracy: ", acc))
}

gridExtra::grid.arrange(vis_detection_rate_rf(1), 
                        vis_detection_rate_rf(2), 
                        vis_detection_rate_rf(3), 
                        vis_detection_rate_rf(4))

# RF, Tune

#=================================
#  Full Cartesian Grid Search
#=================================

# Set hyperparameter grid: 

hyper_grid.h2o <- list(ntrees = seq(50, 500, by = 50),
                       mtries = seq(3, 5, by = 1),
                       # max_depth = seq(10, 30, by = 10),
                       # min_rows = seq(1, 3, by = 1),
                       # nbins = seq(20, 30, by = 10),
                       sample_rate = c(0.55, 0.632, 0.75))

# The number of models is 90: 
sapply(hyper_grid.h2o, length) %>% prod()

# Train 6000 Random Forest Models: 
system.time(grid_cartesian <- h2o.grid(algorithm = "randomForest",
                                       grid_id = "rf_grid1",
                                       x = predictors, 
                                       y = response, 
                                       seed = 123, 
                                       nfolds = 10, 
                                       training_frame = frame.train,
                                       stopping_metric = "AUC", 
                                       hyper_params = hyper_grid.h2o,
                                       search_criteria = list(strategy = "Cartesian")))

# Collect the results and sort by our model performance metric of choice: 
grid_perf <- h2o.getGrid(grid_id = "rf_grid1", 
                         sort_by = "auc", 
                         decreasing = FALSE)

# Best model chosen by validation error: 
best_model <- h2o.getModel(grid_perf@model_ids[[1]])

# Use best model for making predictions: 
confusionMatrix(h2o.predict(best_model, frame.test) %>% as.data.table() %>% pull(predict), 
                frame.test %>% as.data.table() %>% pull(Survived) %>% as.factor())

# RF, Diag

summary(best_model)

h2o.varimp_plot(best_model)

# RF, submission

rf.pred <- h2o.predict(h2o.rf, test.h2o, type = "prob") # std
rf.pred <- h2o.predict(best_model, test.h2o, type = "prob") # tuned

h2o.rf.results <- data.table(PassengerId = test$PassengerId,
                              Survived = as.vector(round(rf.pred$predict)))

data.table::fwrite(h2o.glm.results, file = file.path(submission.dir, "h2o.rf.submission.csv"))

### All done, shutdown H2O    
h2o.shutdown(prompt=FALSE)

rm(list = ls())

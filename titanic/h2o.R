library(data.table)
library(ggplot2)
library(h2o)

# Utility
library(here)

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

data.dir <- file.path(here::here(), project, 'datasets'); submission.dir <- file.path(data.dir, "submission")

train <- data.table::fread(file.path(data.dir, "train.csv"), 
                           stringsAsFactors = T)

test <- data.table::fread(file.path(data.dir, "test.csv"),
                          stringsAsFactors = T)

# Init h2o

h2o.init(max_mem_size = "32gb")


train.h2o <- as.h2o(train)
test.h2o <- as.h2o(test)

excluded <- c("PassengerId", "Name", "SibSp", "Ticket")

response <- "Survived"
predictors <- setdiff(setdiff(colnames(train), response), excluded)

n.predictors <- length(predictors)

# General Linear Model

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

summary(h2o.glm)

h2o.glm@model$training_metrics@metrics$AUC

glm.pred <- predict(h2o.glm, test.h2o, type = "response")

h2o.glm.results <- data.table(PassengerId = test$PassengerId,
                              Survived = as.vector(glm.pred$predict))

data.table::fwrite(h2o.glm.results, file = file.path(submission.dir, "h2o.glm.results.csv"))

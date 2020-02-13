library(data.table)
library(tidyverse)
library(recipes)
library(stringr)
library(GGally)
library(skimr)

#=================================
#  Utility
#=================================

set_global_theme <- function()
{
  # Base Theme
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
  
  print('Global Theme Set.')
}

data_raw <- function() {
  
  data.dir <- file.path(here::here(), project, "datasets")
  
  train_raw <- data.table::fread(file.path(data.dir, "train.csv"), 
                                 stringsAsFactors = T)
  
  test_raw <- data.table::fread(file.path(data.dir, "test.csv"),
                                stringsAsFactors = T)
  
  list(train = train_raw, test = test_raw)
}

data_preprocessed <- function(raw_data)
{
  data <- raw_data
  
  data$train$label <- as.factor(data$train$label)
  
  cols <- names(train[, !"label"])

  data$train[, (cols) := lapply(.SD, function(x) x / 255), .SDcols = cols]
  data$test[, (cols) := lapply(.SD, function(x) x / 255), .SDcols = cols]
  
  list(train = data$train, test = data$test)
}
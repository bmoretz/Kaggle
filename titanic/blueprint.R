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

#=================================
#  Raw Data
#=================================

data_raw <- function() {
  
  data.dir <- file.path(here::here(), project, "data")
  
  train_raw <- data.table::fread(file.path(data.dir, "train.csv"), 
                                 stringsAsFactors = T)
  
  test_raw <- data.table::fread(file.path(data.dir, "test.csv"),
                                stringsAsFactors = T)
  
  list(train = train_raw, test = test_raw)
}

#=================================
#  Feature Engineering (EDA Ready)
#=================================

update_ticket_info <- function(data) {
  #' Extract ticket details into seperate columns.
  #'
  #' @param data data set for mutation.
  #' @return Mutated data set with Ticket (number), Origin, Arrive (best guess).
  tickets <- data.table(Ticket = data$Ticket)
  
  suppressWarnings({
    # warnings by design
    tickets[, numeric := as.numeric(as.character(Ticket))]
    
    ticket_info <- tickets %>%
      mutate(numbers = str_extract(Ticket, "(\\s)+(.)+[0-9]+")) %>%
      mutate(prefix = str_replace(Ticket, numbers, "")) %>%
      mutate(prefix_group = str_replace_all(prefix, "\\.", "")) %>%
      separate(prefix_group, c("Origin", "Arrive")) %>%
      mutate(TicketNumber = coalesce(as.numeric(numbers), numeric)) %>%
      mutate(Origin = as.factor(Origin),
             Arrive = as.factor(Arrive)) %>%
      as.data.table()
  })
  
  ticket_info[is.na(Origin)]$Origin <- "None"
  ticket_info[is.na(Arrive)]$Arrive <- "None"
  ticket_info[is.na(TicketNumber)]$TicketNumber <- 0
  
  cbind(data %>% select(-Ticket), ticket_info[, .(Origin, Arrive, TicketNumber)])
}

update_cabin_info <- function(data) {
  #' Extract cabin details into seperate columns.
  #'
  #' @param data data set for mutation.
  #' @return Mutated data set with Cabin Area.0  
  cabin_info <- data %>%
    mutate(CabinArea = str_extract(Cabin, "[A-Z]+"),
           CabinNumber = as.numeric(str_extract(Cabin, "[0-9]+"))) %>%
    as.data.table()
  
  cabin_info[is.na(CabinArea)]$CabinArea <- "None"
  cabin_info[is.na(CabinNumber)]$CabinNumber <- 0
  
  cbind(data %>% select(-Cabin), cabin_info[, .(CabinArea, CabinNumber)])
}

data_preprocessed <- function(raw)
{
  train <- raw$train; test <- raw$test

  # Response as 2 level factor
  train$Survived <- as.factor(train$Survived)
  
  stopifnot(levels(train$Survived) == c("0", "1"))
  
  train_processed <- train %>% 
    update_ticket_info() %>%
    update_cabin_info()
  
  test_processed <- test %>%
    update_ticket_info() %>%
    update_cabin_info()
  
  list(train = train_processed, test = test_processed)
}

#=================================
#  Blueprint (ML Ready)
#=================================

# processed <- data_raw() %>% data_preprocessed()
              
data_processed <- function(processed)
{
  train <- processed$train; test <- processed$test;
  
  blueprint <- recipe(Survived ~ ., data = train) %>%
    step_other(all_nominal(), threshold = 0.005) %>%
    step_dummy(Embarked, Origin, Arrive, CabinArea) %>%
    step_meanimpute(Age, Fare) %>%
    step_scale(Fare, Age, CabinNumber, TicketNumber) %>%
    step_center(Fare, Age, CabinNumber, TicketNumber) %>%
    check_missing(all_predictors())
  
  train_prepped <- prep(blueprint, training = train, retain = T) %>%
    juice()
  
  test_prepped <- prep(blueprint, training = train) %>%
    bake(new_data = test)
  
  list(train = train_prepped, test = test_prepped)
}
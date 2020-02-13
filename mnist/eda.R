library(tidyverse)

project <- 'mnist'
source(file.path(project, "blueprint.R"))

set_global_theme()

data <- data_raw() %>%
  data_preprocessed()

mnist <- data$train

#=================================
#  Utilities
#=================================

view_image <- function(index) {
  pixels <- matrix(as.numeric(mnist[index, !"label"]), 
                   nrow = 28, ncol = 28)
  image(pixels, col = grey(255:0/255))
  title(main = paste("index: ", index, "-", mnist[index]$label))
}

view_grid <- function(offset = 1) {
  opar <- par()
  par(mfrow = c(4, 4), mar = c(3.1, 2.1, 2.1, 2.1))
  sapply(seq(offset, length.out = 16), view_image)
  par(opar)
}

view_by_label <- function(num) {
  images <- mnist[, .(label, index = .I)][label == num]
  par(mfrow = c(4, 4), mar = c(3.1, 2.1, 2.1, 2.1))
  sapply(head(images$index, 16), view_image)
  par(opar)    
}

view_distinct_digits <- function() {
  opar <- par()
  par(mfrow = c(4, 3), mar = c(3.1, 2.1, 2.1,2.1))
  images <- mnist[, .(label, index = .I)][, head(.SD, 1), by = label]
  setorder(images, label)
  sapply(images$index, view_image)
  par(opar)
}

#=================================
#  EDA
#=================================

# avg all images
avg <- matrix( apply(mnist[, !"label"], 2, mean), nrow = 28)
image(avg, col = grey(255:0/255))

# images sd
sd <- matrix( apply(mnist[, !"label"], 2, sd), nrow = 28)
image(sd, col = grey(255:0/255))

# peek top 16 images
view_grid()

view_distinct_digits()

view_by_label(0)
view_by_label(1)
view_by_label(2)


library(tidyverse)

project <- 'titanic'
source(file.path(project, "blueprint.R"))

set_global_theme()

data <- data_raw() %>%
  data_preprocessed()

titanic <- data$train;

#=================================
#  Exploratory
#=================================

numeric.cols <- colnames(titanic)[sapply(titanic, is.numeric)]
ggpairs(titanic[, ..numeric.cols])

titanic %>% skim()

titanic[, .(Pct = .N/nrow(titanic)), Survived]

ggplot(titanic, aes(Pclass, Survived, color = Sex)) +
  stat_ecdf()

# Is cabin area statistically significant?

ggplot(titanic[!is.na(CabinArea)], aes(CabinArea)) +
  geom_bar(aes(fill = CabinArea))

ggplot(train, aes(CabinArea)) +
  geom_bar(aes(fill = CabinArea))

cabin_stats <- titanic[!is.na(CabinArea), .(CabinArea, Survived)]

by_cabin_area <- cabin_stats[, 
                             .(Survived = sum(Survived == 1), Died = sum(Survived == 0)),
                             by = CabinArea]

melt(by_cabin_area, id.vars = "CabinArea") %>%
  ggplot(aes(CabinArea, value)) +
  geom_bar(aes(fill = CabinArea), stat = "identity") +
  facet_wrap( ~ variable, ncol = 1) + 
  labs(y = "# of People")

cabin_tbl <- table(cabin_stats$Survived, cabin_stats$CabinArea)

cabin_tbl %>% chisq.test() # p-value 0.1722

cabin_tbl %>% prop.table() %>% round(2)

# Is ticket type statistically significant?

ggplot(titanic[!is.na(Origin)], aes(Origin)) +
  geom_bar(aes(fill = Origin))

ggplot(titanic[!is.na(Origin)], aes(Origin)) +
  geom_bar(aes(fill = Origin)) +
  facet_wrap(~ Survived, nrow = 2)

ggplot(titanic[!is.na(Arrive)], aes(Arrive)) +
  geom_bar(aes(fill = Origin)) +
  facet_wrap(~ Survived, nrow = 2)

ticket_stats <- titanic[, .(Survived, Origin, Arrive)]

origin_tbl <- table(ticket_stats$Survived, ticket_stats$Origin)

origin_tbl %>% chisq.test() # p-value < 0.001

origin_tbl %>% prop.table() %>% round(2)

# Cabin Number

ggplot(titanic, aes(CabinNumber)) +
  geom_histogram() +
  facet_wrap(~Survived, nrow = 2)

ggplot(titanic, aes(Survived, CabinNumber)) +
  geom_boxplot(aes(fill = Survived))

# Fare

ggplot(titanic, aes(Sex, Fare, fill = Sex)) +
  geom_boxplot()

ggplot(titanic, aes(Fare, Survived)) +
  stat_ecdf() +
  facet_wrap(~Sex, nrow = 2)

ggplot(titanic, aes(Fare)) +
  geom_histogram(aes(fill = ..count..), bins = 30)

titanic[Fare > 3 * sd(Fare)]

# Age

missing.age <- titanic[is.na(Age),]

nrow(missing.age) / nrow(titanic) # 20% missing age

missing.age[, .(Missing = .N, Pct = .N / nrow(missing.age)),
            by = Sex] # 70% male, 30% female

# Survived
ggplot(titanic, aes(Survived, Age, group = Survived)) +
  geom_boxplot(aes(fill = Survived))

ggplot(titanic, aes(Embarked)) +
  geom_bar(aes(fill = ..count..))
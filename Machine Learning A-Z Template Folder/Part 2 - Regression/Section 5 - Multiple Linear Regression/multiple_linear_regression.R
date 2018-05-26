dataset = read.csv('50_Startups.csv')
# dataset = dataset[, 2:3]

# Encoding categorical data
dataset$State = factor(dataset$State,
                         levels = c('New York', 'Florida', 'California'),
                         labels = c(1, 2, 3))

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Profit, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

#Fitting multiple linear regression to training set, much easier to do in R
regressor = lm(formula = Profit ~ .,
               data = training_set)

#Predicting test set results
Y_pred = predict(regressor, newdata = test_set)

#Building optimal model using Backward Elimination
regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State,
               data = training_set)
summary(regressor)

regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend,
               data = training_set)
summary(regressor)

regressor = lm(formula = Profit ~ R.D.Spend + Marketing.Spend,
               data = training_set)
summary(regressor)

Y_pred = predict(regressor, newdata = test_set)

regressor = lm(formula = Profit ~ R.D.Spend,
               data = training_set)
summary(regressor)

#Y_pred = predict(regressor, newdata = test_set)
  
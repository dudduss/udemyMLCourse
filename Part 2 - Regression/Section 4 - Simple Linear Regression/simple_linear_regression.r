dataset = read.csv('Salary_Data.csv')
# dataset = dataset[, 2:3]

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Salary, SplitRatio = 2/3)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

#Fitting simple linear regression to training set
regressor = lm(formula = Salary ~ YearsExperience, 
               data = training_set)

#Predicting
y_pred = predict(regressor, newdata = test_set)


#Plotting

# install.packages('ggplot2')
ggplot() + 
  geom_point(aes(x = training_set$YearsExperience, y = training_set$Salary),
                      color = 'red') + 
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)),
                      color = 'blue') +
  ggtitle('Salary vs Experience') +
  xlab('Experience') +
  ylab('Salary')

ggplot() +
  geom_point(aes(x = test_set$YearsExperience, y = test_set$Salary),
             color = 'red') + 
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)),
            color = 'blue') +
  ggtitle('Salary vs Experience') +
  xlab('Experience') +
  ylab('Salary') 



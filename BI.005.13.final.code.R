rm(list=ls())
options(scipen = 999)

#setwd("/Users/christinanguyen/Desktop/MIS 6356/Final Project")

#Read dataset
stroke.df <- read.csv("stroke.csv")

summary(stroke.df) #No NAs
stroke.df <- stroke.df[,-1] #Remove id column

######################## Oversample the minority class ######################### 
table(stroke.df$stroke) #Unbalanced stroke distribution 

install.packages("ROSE")
library(ROSE)
oversampled_data <- ovun.sample(stroke ~ ., data = stroke.df, method = "over", N = 7000, seed = 123)

# The oversampled data is stored in the data slot of the resulting object
oversampled_data <- oversampled_data$data

# Check the new distribution of the target variable
table(oversampled_data$stroke)

############################ Imputation of NA values ###########################
summary(oversampled_data$bmi) #bmi is character type
# Convert bmi to numeric
oversampled_data$bmi <- as.numeric(oversampled_data$bmi)
# Replace NA with mean
mean_bmi <- mean(oversampled_data$bmi, na.rm = TRUE)
oversampled_data$bmi[is.na(oversampled_data$bmi)] <- mean_bmi

############# Convert categorical to dummy variables for modeling ##############
library(dplyr)

# Create dummy variables
oversampled_data$ever_married <- ifelse(oversampled_data$ever_married == "Yes", 1, 0)
oversampled_data $Residence_type <- ifelse(oversampled_data$Residence_type == "Urban", 1, 0)
oversampled_data $gender <- ifelse(oversampled_data$gender == "Male", 1, 0)

categorical_vars <- c("work_type", "smoking_status")
encoded.df <- cbind(oversampled_data, model.matrix(~ . - 1, data = oversampled_data[, categorical_vars]))
head(encoded.df)

# Drop original categorical and redundant variables and rename columns to remove hyphen and space
encoded.df <- encoded.df %>%
  select(-work_type, -smoking_status) %>%
  rename(work_typeSelf_employed = contains("work_typeSelf-employed"), 
         smoking_statusnever_smoked = contains("smoking_statusnever smoked"))
head(encoded.df)

########################## Exploratory Data Analysis ###########################
install.packages("corrplot")
library(corrplot)

# Move stroke to last column (better viz of corr plot)
stroke_index <- which(colnames(encoded.df) == "stroke")
encoded.df <- encoded.df[, c(1:(stroke_index - 1), (stroke_index + 1):ncol(encoded.df), stroke_index)]

# Calculate the correlation matrix
cor_matrix <- cor(encoded.df)

# Set up the correlation plot
corrplot(cor_matrix, method = "color", type = "lower", tl.col = "black", tl.srt = 45, tl.cex = 0.5)

# Create a list of continuous variables
continuous_vars <- c("age", "avg_glucose_level", "bmi")

# Perform two-sample t-tests for each continuous variable
t_test_results <- lapply(continuous_vars, function(var) {
  t_test <- t.test(encoded.df[[var]][encoded.df$stroke == 1], encoded.df[[var]]
                   [encoded.df$stroke == 0])
  rounded_p_value <- round(t_test$p.value, digits = 9)
  return(data.frame(Variable = var, Statistic = t_test$statistic, 
                    PValue = rounded_p_value, TestType = "T-Test"))
})

# Create a list of categorical variables
categorical_vars <- c("hypertension", "heart_disease", "gender","ever_married", 
                      "work_typeGovt_job", "work_typeNever_worked","work_typePrivate",
                      "work_typeSelf_employed", "Residence_type", "smoking_statusnever_smoked",
                      "smoking_statussmokes")

# Perform chi-squared tests for each categorical variable
chi_squared_results <- lapply(categorical_vars, function(var) {
  contingency_table <- table(encoded.df$stroke, encoded.df[[var]])
  chi_squared_result <- chisq.test(contingency_table)
  rounded_p_value <- round(chi_squared_result$p.value, digits = 9)
  return(data.frame(Variable = var, Statistic = chi_squared_result$statistic, 
                    PValue = rounded_p_value, TestType = "Chi-Squared"))
})

# Combine the results into one data frame
all_results <- do.call(rbind, c(t_test_results, chi_squared_results))

# Display the combined results
print(all_results) #genderFemale, work_typeGovt_job, smoking_statussmokes p-values > 0.05

############################### Regression tree ################################
library(rpart)
library(rpart.plot)
library(caret)

# Partition
set.seed(1)  
train.index <- sample(c(1:dim(encoded.df)[1]), dim(encoded.df)[1]*0.6)  
train_rt.df <- encoded.df[train.index, ]
valid_rt.df <- encoded.df[-train.index, ]

# Classification tree
default.ct <- rpart(stroke ~ ., data = train_rt.df ,method = "class")

# Plot tree, uses GINI INDEX
#prp(default.ct, type = 1, extra = 2, under = TRUE, split.font = 1, varlen = 15)

prp(default.ct, type = 1, extra = 1, under = TRUE, split.font = 1, varlen = 15,
    branch = 1,
    box.col = ifelse(!is.na(default.ct$frame$yval), default.ct$frame$yval, NA),
    split.cex = 0.8, split.suffix = "?")

# Count number of leaves
length(default.ct$frame$var[default.ct$frame$var == "<leaf>"])

# Classify records in the training data.
# Set argument type = "class" in predict() to generate predicted class membership.
default.ct.point.pred.train <- predict(default.ct,train_rt.df,type = "class")
# Generate confusion matrix for training data
confusionMatrix(default.ct.point.pred.train, as.factor(train_rt.df$stroke))

# Repeat code for validation set
default.ct.point.pred.valid <- predict(default.ct,valid_rt.df,type = "class")
confusionMatrix(default.ct.point.pred.valid, as.factor(valid_rt.df$stroke))

install.packages("pROC")
library(pROC)

# Plot ROC curve for the validation set
roc_valid_rt <- roc(valid_rt.df$stroke, as.numeric(default.ct.point.pred.valid))
plot(roc_valid_rt, main = "RT ROC Curve (Validation Set)", col = "red", lwd = 2)
auc(roc_valid_rt)

################################ Random Forest #################################
library(randomForest)

rf <- randomForest(as.factor(stroke) ~ ., data = train_rt.df, ntree = 500, mtry = 4, 
                   nodesize = 5, importance = TRUE)  

# Variable importance plot
varImpPlot(rf, type = 1)

# Generate confusion matrix for training data
rf.train <- predict(rf, train_rt.df)
confusionMatrix(rf.train, as.factor(train_rt.df$stroke))

# Repeat code for validation data
rf.pred <- predict(rf, valid_rt.df)
confusionMatrix(rf.pred, as.factor(valid_rt.df$stroke))

# Plot ROC curve for the validation set
roc_valid_rf <- roc(valid_rt.df$stroke, as.numeric(rf.pred))
plot(roc_valid_rf, main = "RF ROC Curve (Validation Set)", col = "red", lwd = 2)
auc(roc_valid_rf)

################################ Neural Network ################################
install.packages('neuralnet',dependencies=T)
library(neuralnet)

# Normalize the data so that the range is between 0 and 1
normalized.df <- encoded.df
cols_to_scale <- colnames(encoded.df) != "stroke" # Exclude stroke
normalized.df[, cols_to_scale] <- scale(encoded.df[, cols_to_scale])

# Partition
set.seed(1)  
train.index <- sample(c(1:dim(normalized.df)[1]), dim(normalized.df)[1]*0.6)  
train_nn.df <- normalized.df[train.index, ]
valid_nn.df <- normalized.df[-train.index, ]

nn <- neuralnet(stroke ~., data = train_nn.df, linear.output = F, hidden = 5, learningrate=1.5)

plot(nn, rep="best")

# Generate confusion matrix for training data
nn.pred.train <- predict(nn, train_nn.df, type = "response")
nn.pred.classes.train <- ifelse(nn.pred.train > 0.5, 1, 0)
confusionMatrix(as.factor(nn.pred.classes.train), as.factor(train_nn.df$stroke))

# Repeat code for validation set
nn.pred.valid <- predict(nn, valid_nn.df, type = "response")
nn.pred.classes.valid <- ifelse(nn.pred.valid > 0.5, 1, 0)
confusionMatrix(as.factor(nn.pred.classes.valid), as.factor(valid_nn.df$stroke))

#install.packages("pROC")
library(pROC)

roc_valid_nn <- roc(valid_nn.df$stroke, as.numeric(nn.pred.valid))
plot(roc_valid_nn, main = "NN ROC Curve (Validation Set)", col = "red", lwd = 2)
auc(roc_valid_nn)

########################## Logistic Regression ################################
set.seed(2)
train.index <- sample(c(1:dim(encoded.df)[1]), dim(encoded.df)[1]*0.6)  

train.df <- encoded.df[train.index, ]
valid.df <- encoded.df[-train.index, ]

# Create logistic regression model
model <- glm(stroke ~ ., data = train.df, family = "binomial")
summary(model)

# use predict() with type = "response" to compute predicted probabilities.
logit.reg.pred <- predict(model, valid.df, type = "response")

# first 5 actual and predicted records
data.frame(actual = valid.df$stroke[1:5], predicted = logit.reg.pred[1:5])

#Generate confusion matrix
logit.reg.pred.classes <- ifelse(logit.reg.pred > 0.5, 1, 0)
confusionMatrix(as.factor(logit.reg.pred.classes), as.factor(valid.df$stroke))

# Plot ROC curve for the validation set
roc_valid_log <- roc(valid.df$stroke, as.numeric(logit.reg.pred.classes))
plot(roc_valid_log, main = "Logistic ROC Curve (Validation Set)", col = "red", lwd = 2)
auc(roc_valid_log)

# Compute the correlation matrix
numeric_data <- oversampled_data %>%
  select_if(is.numeric)
correlation_matrix <- cor(numeric_data, use = "complete.obs")

# Print the correlation matrix
print(correlation_matrix)


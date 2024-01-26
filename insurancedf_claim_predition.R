#load libraries
library(tidyverse)
library(caret)
library(ROCR)
library(ROSE)


#set working directory (adjust this for your own computer)
setwd("C:/Users/Lucy Wu/Documents")

#read dataset into R
insurancedf <- read.csv("insurance.csv")
View(insurancedf)

#Convert categorical variables to factors with levels and labels
insurancedf$CLAIM<-factor(optivadf$CLAIM,levels = c(0,1),labels = c("No","Yes"))
insurancedf$KIDSDRIV<-factor(optivadf$KIDSDRIV,levels = c(0,1),labels = c("No","Yes"))
insurancedf$HOMEKIDS<-factor(optivadf$HOMEKIDS,levels = c(0,1),labels = c("No","Yes"))
insurancedf$HOMEOWN<-factor(optivadf$HOMEOWN,levels = c(0,1),labels = c("No","Yes"))
insurancedf$MSTATUS<-factor(optivadf$MSTATUS,levels = c(0,1),labels = c("No","Yes"))
insurancedf$GENDER<-factor(optivadf$GENDER,levels = c(0,1),labels = c("No","Yes"))
insurancedf$EDUCATION<-factor(optivadf$EDUCATION,levels = c(0,1),labels = c("High School only","College or beyond"))
insurancedf$CAR_USE<-factor(optivadf$CAR_USE,levels = c(0,1),labels = c("Private","Commercial"))
insurancedf$CLM_BEF<-factor(optivadf$CLM_BEF,levels = c(0,1),labels = c("No","Yes"))
insurancedf$RED_CAR<-factor(optivadf$RED_CAR,levels = c(0,1),labels = c("No","Yes"))
insurancedf$REVOKED<-factor(optivadf$REVOKED,levels = c(0,1),labels = c("No","Yes"))
insurancedf$MVR_PTS<-factor(optivadf$MVR_PTS,levels = c(0,1),labels = c("No","Yes"))
insurancedf$URBANICITY<-factor(optivadf$URBANICITY,levels = c(0,1),labels = c("Rural","Urban"))

#check for missing data
sum(is.na(insurancedf))

#generate summary statistics for all variables in dataframe
summary(insurancedf)


#set seed so the random sample is reproducible
set.seed(42)


#Partition the Optiva dataset into a training, validation and test set
Samples<-sample(seq(1,3),size=nrow(insurancedf),replace=TRUE,prob=c(0.6,0.2,0.2))
Train<-insurancedf[Samples==1,]
Validate<-insurancedf[Samples==2,]
Test<-insurancedf[Samples==3,]

#View descriptive statistics for each dataframe
summary(Train)
summary(Validate)
summary(Test)




#Step 3:
#We don’t have a severe class imbalance in the insurance dataset, 
#so we’re going to start with fitting a model to the training set. 
#Conduct a logistic regression analysis using the training data frame 
#with CLAIM  as the outcome variable 
#and all the other variables in the dataset as predictor variables. 

options(scipen=999)

lr_1 <- glm(CLAIM ~., data = Train, 
              family = binomial(link = "logit"))

# model summary
summary(lr_1)

#exponentiate the regression coefficients from the logistic regression model 
#using the oversample dataframe
exp(coef(lr_1))


#Step 4

#Using the model you fitted in Step (3) and 
#the validation data frame you created in Step (2), 
#create a confusion matrix to assess the accuracy of the logistic regression model.




lrprobsU <- predict(lr_1, newdata = Validate, type = "response")


lrclassU <- as.factor(ifelse(lrprobsU > 0.5, "Yes","No"))


confusionMatrix(lrclassU, Validate$CLAIM, positive = "Yes" )



#Step 5

#Again using the model you fitted in Step (3) and the validation data frame, 
#create an ROC curve plot and calculate the AUC.


predROCU <- prediction(lrprobsU, Validate$CLAIM)

#create a performance object to use for the ROC Curve
perfROCU <- performance(predROCU,"tpr", "fpr")

#plot the ROC Curve
plot(perfROCU)
abline(a=0, b= 1)


# compute AUC 
performance(predROCU, measure="auc")@y.values[[1]]




#Step 6



#imbalance to see if it improves our model accuracy. 
#Using the training set you generated in Step (2), 
#create a new training subset using the oversampling method. 

#Create an oversampled training subset
xsdf<-Train[c(-2)]
View(xsdf)

set.seed(42)
oversample<-upSample(x=xsdf, y=Train$CLAIM, yname = "CLAIM")

table(oversample$CLAIM)


#Step 7

#Conduct a logistic regression analysis using the new oversampled training subset
#with CLAIM  as the outcome variable and 
#all the other variables in the dataset as predictor variables. 


lrOver <- glm(CLAIM ~ . , data = oversample, 
              family = binomial(link = "logit"))

# model summary
summary(lrOver)


#Step 8

#Using the model you fitted in Step (7) and 
#the validation data frame you created in Step (2), 
#create a confusion matrix to assess the accuracy of the logistic regression model.


# obtain probability of defaulting for each observation in validation set
lrprobsO <- predict(lrOver, newdata = Validate, type = "response")

#Attach probability scores to Validate dataframe
Validate <- cbind(Validate, Probabilities=lrprobsO)

# obtain predicted class for each observation in validation set using threshold of 0.5
lrclassO <- as.factor(ifelse(lrprobsO > 0.5, "Yes","No"))

#Attach predicted class to Validate dataframe
Validate <- cbind(Validate, PredClass=lrclassO)

#Create a confusion matrix using "Yes" as the positive class 
confusionMatrix(lrclassO, Validate$CLAIM, positive = "Yes" )



#Step 9

#Again using the model you fitted in Step (7) and 
#the validation data frame, 
#create an ROC curve plot and calculate the AUC.


#Plot ROC Curve for model from oversampled training set

#create a prediction object to use for the ROC Curve
predROC <- prediction(lrprobsO, Validate$CLAIM)

#create a performance object to use for the ROC Curve
perfROC <- performance(predROC,"tpr", "fpr")

#plot the ROC Curve
plot(perfROC)
abline(a=0, b= 1)

# compute AUC 
performance(predROC, measure="auc")@y.values[[1]]




#Step10

#so we will use the logistic regression model fitted 
#to the oversampled training subset. 
#Using the model generated in Step (7) and 
#the test set you created in Step (2), 
#create a confusion matrix to assess the accuracy of 
#the logistic regression model on the test data frame.



# obtain probability of defaulting for each observation in test set
lrprobstest <- predict(lrOver, newdata = Test, type = "response")

# obtain predicted class for each observation in test set using threshold of 0.5
lrclasstest <- as.factor(ifelse(lrprobstest > 0.5, "Yes","No"))

#Create a confusion matrix using "Yes" as the positive class 
confusionMatrix(lrclasstest, Test$CLAIM, positive = "Yes" )

#Plot ROC Curve for model from oversampled training set using Test set

#create a prediction object to use for the ROC Curve
predROCtest <- prediction(lrprobstest, Test$CLAIM)

#create a performance object to use for the ROC Curve
perfROCtest <- performance(predROCtest,"tpr", "fpr")

#plot the ROC Curve
plot(perfROCtest)
abline(a=0, b= 1)

# compute AUC 
performance(predROCtest, measure="auc")@y.values[[1]]





#read new dataset into R
new_customers <- read.csv("insurance_predictions.csv")
View(new_customers)

#Convert categorical variables to factors with levels and labels
new_customers$KIDSDRIV<-factor(new_customers$KIDSDRIV,levels = c(0,1),labels = c("No","Yes"))
new_customers$HOMEKIDS<-factor(new_customers$HOMEKIDS,levels = c(0,1),labels = c("No","Yes"))
new_customers$HOMEOWN<-factor(new_customers$HOMEOWN,levels = c(0,1),labels = c("No","Yes"))
new_customers$MSTATUS<-factor(new_customers$MSTATUS,levels = c(0,1),labels = c("No","Yes"))
new_customers$GENDER<-factor(new_customers$GENDER,levels = c(0,1),labels = c("No","Yes"))
new_customers$EDUCATION<-factor(new_customers$EDUCATION,levels = c(0,1),labels = c("High School only","College or beyond"))

new_customers$CAR_USE<-factor(new_customers$CAR_USE,levels = c(0,1),labels = c("Private","Commercial"))
new_customers$RED_CAR<-factor(new_customers$RED_CAR,levels = c(0,1),labels = c("No","Yes"))
new_customers$CLM_BEF<-factor(new_customers$CLM_BEF,levels = c(0,1),labels = c("No","Yes"))
new_customers$REVOKED<-factor(new_customers$REVOKED,levels = c(0,1),labels = c("No","Yes"))
new_customers$MVR_PTS<-factor(new_customers$MVR_PTS,levels = c(0,1),labels = c("No","Yes"))
new_customers$URBANICITY<-factor(new_customers$URBANICITY,levels = c(0,1),labels = c("Rural","Urban"))

# make predictions for new data (for which loan default is unknown)
lrprobsnew <- predict(lrOver, newdata = new_customers , type = "response")

#Attach probability scores to new_customers dataframe 
new_customers <- cbind(new_customers, Probabilities=lrprobsnew)
View(new_customers)


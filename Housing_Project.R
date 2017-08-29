install.packages("cluster");
install.packages("mclust");
install.packages("ggplot2");
install.packages("Boruta")
install.packages("randomForest")
install.packages("SDMTools")
library(SDMTools)
library(randomForest)
library(Boruta)
library(ggplot2)
require(mclust)
library(cluster)
library(fpc)
#Boruta Feature Selection/Importance Package
#-----------------------------------------------------------------------------------------------------------------------------------



setwd("D:/MS DAEN/2nd Semester/AIT-582/Project")
K_Means_Data <- read.csv("train-C1Con.csv", header = T, stringsAsFactors = F)
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}
DF=data.frame(K_Means_Data)
K_Means_Data <- as.data.frame(lapply(DF[4:12], normalize))
K_Means_Data
traindata=K_Means_Data

str(traindata)
names(traindata) <- gsub("_", "", names(traindata))
summary(traindata)
traindata <- traindata[complete.cases(traindata),]
traindata[traindata == ""] <- NA
convert <- c(1:9)
traindata[,convert] <- data.frame(apply(traindata[convert], 2, as.factor))
set.seed(12345)
boruta.train <- Boruta(EstimatedRent~., data = traindata, doTrace = 2)
print(boruta.train)
plot(boruta.train, xlab = "", xaxt = "n")
lz<-lapply(1:ncol(boruta.train$ImpHistory),function(i) boruta.train$ImpHistory[is.finite(boruta.train$ImpHistory[,i]),i])
names(lz) <- colnames(boruta.train$ImpHistory)
Labels <- sort(sapply(lz,median))
axis(side = 1,las=2,labels = names(Labels), at = 1:ncol(boruta.train$ImpHistory), cex.axis = 0.7)
final.boruta <- TentativeRoughFix(boruta.train)
print(final.boruta)
boruta.df <- attStats(final.boruta)
class(boruta.df)
print(boruta.df)








#Simple Regression for houses
#---------------------------------------------------------------------------------------------------------------------------------------------------------------



setwd("D:/MS DAEN/2nd Semester/AIT-582/Project")
K_Means_Data <- read.csv("train-C1H.csv")
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}
DF=data.frame(K_Means_Data)
K_Means_Data <- as.data.frame(lapply(DF[4:12], normalize))
TempDnomData <- DF[4:12]
K_Means_Data
TempDnomData
myvars = c(3:6)
tempData=K_Means_Data[myvars]
tempData
Price_Year_Area_Cluster <- Mclust(tempData, G = NULL, modelNames = NULL, prior = NULL, control = emControl(), initialization = NULL, warn = FALSE)
summary(Price_Year_Area_Cluster)
library(fpc)
plotcluster(tempData, Price_Year_Area_Cluster$classification)
#write.csv(K_Means_Data[Price_Year_Area_Cluster$cluster==1,], file = "foo.csv")
K_Means_Data[Price_Year_Area_Cluster$classification==1,]

mydata <- K_Means_Data[Price_Year_Area_Cluster$classification==2,]
row<-nrow(K_Means_Data[Price_Year_Area_Cluster$classification==2,])
row
mydata
set.seed(12345)
0.6*row
trainindex <- sample(row, 0.6*row, replace=FALSE)
training <- mydata[trainindex,]
validation <- mydata[-trainindex,]
trainfit<-lm(Estimated_Rent ~ ZipCode + Bedrooms + Bathrooms + Estimated_Price + AreaSpace + YearBuilt + Mortgage +  Price_Per_SQFT, data=training)
summary(trainfit)
fitsummary = summary(trainfit)
fitsummary$r.squared
fitsummary$adj.r.squared
BIC(trainfit)
PredBase<-predict(trainfit, validation, se.fit=TRUE)
Predicted_validation_Result <- PredBase$fit
mean((validation$Estimated_Rent-Predicted_validation_Result)^2)




#Backward Regression for houses
#---------------------------------------------------------------------------------------------------------------------------------------------------------------


mydata <- K_Means_Data[Price_Year_Area_Cluster$classification==2,]
row<-nrow(K_Means_Data[Price_Year_Area_Cluster$classification==2,])
row
set.seed(12345)
0.6*row
trainindex <- sample(row, 0.6*row, replace=FALSE)
training <- mydata[trainindex,]
validation <- mydata[-trainindex,]
trainfit<-lm(Estimated_Rent ~ ZipCode + Bedrooms + Bathrooms + Estimated_Price + AreaSpace + YearBuilt +  Price_Per_SQFT, data=training)
backward<-step(trainfit, direction='backward')
backward
coefficients(backward)
summary(backward)
fitsummary = summary(backward)
fitsummary$r.squared
fitsummary$adj.r.squared
BIC(backward)
PredBase<-predict(backward, validation, se.fit=TRUE)  
Predicted_validation_Result <- PredBase$fit
mean((validation$Estimated_Rent-Predicted_validation_Result)^2)




#Simple Regression for townhouse 
#---------------------------------------------------------------------------------------------------------------------------------------------------------------




setwd("D:/MS DAEN/2nd Semester/AIT-582/Project")
K_Means_Data <- read.csv("train-C1T.csv")


normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}
DF=data.frame(K_Means_Data)
K_Means_Data <- as.data.frame(lapply(DF[4:12], normalize))
K_Means_Data
myvars = c(3:6)
#myvars = c(3)
tempData=K_Means_Data[myvars]
tempData


Price_Year_Area_Cluster <-  Mclust(tempData, G = NULL, modelNames = NULL, prior = NULL, control = emControl(), initialization = NULL, warn = FALSE)
summary(Price_Year_Area_Cluster)
library(fpc)
plotcluster(tempData, Price_Year_Area_Cluster$classification)
#write.csv(K_Means_Data[Price_Year_Area_Cluster$classification==7,], file = "Feat4Clust7.csv")

mydata<- K_Means_Data[Price_Year_Area_Cluster$classification==4,]
mydata
row<-nrow(mydata)
row
set.seed(12345)
0.6*row
trainindex <- sample(row, 0.6*row, replace=FALSE)
training <- mydata[trainindex,]
validation <- mydata[-trainindex,]
trainfit<-lm(Estimated_Rent ~ ZipCode + Bathrooms + Estimated_Price + AreaSpace + Mortgage +  Price_Per_SQFT, data=training)
summary(trainfit)
fitsummary = summary(trainfit)
fitsummary$r.squared
fitsummary$adj.r.squared
BIC(trainfit)
PredBase<-predict(trainfit, validation, se.fit=TRUE)
#PredBase
#write.csv(PredBase, file = "fooHPred.csv")
Predicted_validation_Result <- PredBase$fit
mean((validation$Estimated_Rent-Predicted_validation_Result)^2)




#Backward Regression for townhouse
#---------------------------------------------------------------------------------------------------------------------------------------------------------------


mydata<- K_Means_Data[Price_Year_Area_Cluster$classification==4,]
row<-nrow(mydata)
row
set.seed(12345)
0.6*row
trainindex <- sample(row, 0.6*row, replace=FALSE)
training <- mydata[trainindex,]
validation <- mydata[-trainindex,]
trainfit<-lm(Estimated_Rent ~ ZipCode + Bedrooms + Bathrooms + Estimated_Price + AreaSpace + YearBuilt + Mortgage +  Price_Per_SQFT, data=training)
backward<-step(trainfit, direction='backward')
backward
coefficients(backward)
summary(backward)
fitsummary = summary(backward)
fitsummary$r.squared
fitsummary$adj.r.squared
BIC(backward)
PredBase<-predict(backward, validation, se.fit=TRUE)  
Predicted_validation_Result <- PredBase$fit
mean((validation$Estimated_Rent-Predicted_validation_Result)^2)



#Simple Regression for Condos
#---------------------------------------------------------------------------------------------------------------------------------------------------------------



setwd("D:/MS DAEN/2nd Semester/AIT-582/Project")
K_Means_Data <- read.csv("train-C1Con.csv")
min(K_Means_Data[4])
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}
DF=data.frame(K_Means_Data)
K_Means_Data <- as.data.frame(lapply(DF[4:12], normalize))
K_Means_Data
myvars = c(3:6)
tempData=K_Means_Data[myvars]
tempData
Price_Year_Area_Cluster <- Mclust(tempData, G = NULL, modelNames = NULL, prior = NULL, control = emControl(), initialization = NULL, warn = FALSE)
summary(Price_Year_Area_Cluster)
mydata<- K_Means_Data[Price_Year_Area_Cluster$classification==5,]
mydata
row<-nrow(mydata)
row
set.seed(12345)
0.6*row
trainindex <- sample(row, 0.6*row, replace=FALSE)
training <- mydata[trainindex,]
validation <- mydata[-trainindex,]
trainfit<-lm(Estimated_Rent ~ ZipCode + Bedrooms + Estimated_Price + AreaSpace + YearBuilt + Mortgage +  Price_Per_SQFT, data=training)
summary(trainfit)
fitsummary = summary(trainfit)
fitsummary$r.squared
fitsummary$adj.r.squared
BIC(trainfit)
PredBase<-predict(trainfit, validation, se.fit=TRUE)
Predicted_validation_Result <- PredBase$fit
mean((validation$Estimated_Rent-Predicted_validation_Result)^2)






#Backward Regression for Condos
#---------------------------------------------------------------------------------------------------------------------------------------------------------------


K_Means_Data
mydata<- K_Means_Data[Price_Year_Area_Cluster$classification==5,]
mydata
row<-nrow(mydata)
row
set.seed(12345)
0.6*row
trainindex <- sample(row, 0.6*row, replace=FALSE)
training <- mydata[trainindex,]
validation <- mydata[-trainindex,]
trainfit<-lm(Estimated_Rent ~ ZipCode + Bedrooms + Bathrooms + Estimated_Price + AreaSpace + YearBuilt + Mortgage +  Price_Per_SQFT, data=training)
backward<-step(trainfit, direction='backward')
backward
coefficients(backward)
summary(backward)
fitsummary = summary(backward)
fitsummary$r.squared
fitsummary$adj.r.squared
BIC(backward)
PredBase<-predict(backward, validation, se.fit=TRUE)  
Predicted_validation_Result <- PredBase$fit
mean((validation$Estimated_Rent-Predicted_validation_Result)^2)




library(caTools)
library(dplyr)
library(ggplot2)
library(caret)
library(class)
library(corrplot)
install.packages ("gmodels")
install.packages("MASS")
library (gmodels)
library (MASS)
library(dplyr)

animals<-read.csv(file.choose())
View(animals)
anyNA(animals)
str(animals)

animals1<-animals[,-1]
animals1$type<-as.factor(animals1$type)
str(animals1)
animals1[,17]
anyNA(animals)

set.seed(101)

sample <- sample.split(animals1$type,SplitRatio = 0.70)

train <- subset(animals1,sample==TRUE)

test <- subset(animals1,sample==FALSE)


predicted.type <- knn(train[1:16],test[1:16],train$type,k=11)
#Error in prediction
error <- mean(predicted.type!=test$type)
#Confusion Matrix
print(san <- confusionMatrix(data = predicted.type, test$type))

CrossTable(x = test$type, y = predicted.type,
           prop.chisq=FALSE)



predicted.type <- NULL
error.rate <- NULL

for (i in 1:17) {
  predicted.type <- knn(train[1:9],test[1:9],train$type,k=i)
  error.rate[i] <- mean(predicted.type!=test$type)
  
}

knn.error <- as.data.frame(cbind(k=1:17,error.type =error.rate))


ggplot(knn.error,aes(k,error.type))+ 
  geom_point()+ 
  geom_line() + 
  scale_x_continuous(breaks=1:17)+ 
  theme_bw() +
  xlab("Value of K") +
  ylab('Error')

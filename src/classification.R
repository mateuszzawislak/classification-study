# MOW project
#
# classification study
#
# author: Mateusz Zawiœlak
#

library(RWeka)

splitData <- function(data, percent) {
  # calculate columns count
  columnsCount <- floor(nrow(data) * percent / 100)
  # get data random indexes
  randomIndexes <- sample(seq(1,nrow(data)),columnsCount, replace=FALSE)
  
  # select data
  trainset <- data[randomIndexes,]
  testset <- data[-randomIndexes,]
  
  list(test = testset, training = trainset)
}

getColumnIndex <- function(data, columnName) {
  match(columnName, colnames(data))
}

getColumnName <- function(data, index) {
  colnames(data)[index]
}

randomWalk <- function() {
  
}

kNN <- function(splittedData, className) {
  formul = as.formula(paste(className,"~."))
  colIndex <- getColumnIndex(data = splittedData$training, columnName = className)
  
  classifier <- IBk(formul, data = splittedData$training)
  prediction <- predict(classifier, splittedData$test[,-colIndex])
  
  correct <- sum(splittedData$test[,colIndex] == prediction)
  all <- nrow(splittedData$test)
  
  print(correct/all)
}

# const
trainPercent <- 80

# test
irisData <- read.csv(file="D:/Github/classification-study/data/iris.data", head=FALSE, sep=",")
splittedData <- splitData(data = irisData, percent = trainPercent)

kNN(splittedData, "V5")

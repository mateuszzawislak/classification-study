# MOW project
#
# classification study
#
# author: Mateusz Zawiœlak
#

library(RWeka)
library(e1071)

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

selectRandom = function(neighbors){
  rand = sample(1:ncol(neighbors),1)
  neighbors[,rand]
}

randomMask <- function(colNum, classCol, selNum) {
  vector = 1:colNum
  randomColumns = sample(vector[-classCol], selNum, replace=F)
  
  sort(randomColumns)
}

applyMaskOnData <- function(mask, splittedData, classIndex) {
  mask = append(mask, classIndex)
  maskedData = list(training = splittedData$training[,mask], test = splittedData$test[,mask])
  
  maskedData
}

getNeighbors <- function(colNum, classCol, mask) {
  allColumns = seq(from = 1, to = colNum)
  leftColumns = allColumns[-mask]
  leftColumns = leftColumns[leftColumns!=classCol]
  
  neighbors = array(0, c(length(mask), length(leftColumns)*length(mask)))
  
  if(length(leftColumns) > 0) {
    for(i in 1:length(leftColumns)) {
      for(j in 1:length(mask)) {
        neighbor = mask
        neighbor[j] = leftColumns[i]
        
        neighbors[,(i-1)*length(mask)+j] = sort(neighbor)
      }
    }
  }
  
  neighbors
}

# selection methods
randomWalk <- function(splittedData, selNum, className, param, method) {
  trainData = splittedData$training
  classIndex = getColumnIndex(data = trainData, className)
  mask = randomMask(colNum = ncol(trainData), classCol = classIndex, selNum = selNum)
  
  bestRate = rateClassifierWithMask(method, mask, splittedData, classIndex, className)
  for(i in 1:param) {
    neighbors = getNeighbors(ncol(trainData), classIndex, mask)
    if(length(neighbors) > 0) {
      mask = selectRandom(neighbors)
      
      accuracyRate = rateClassifierWithMask(method, mask, splittedData, classIndex, className)
      
      if (accuracyRate > bestRate){
        bestRate = accuracyRate
      }
    }
  }
  
  bestRate
}

hillClimbing <- function(splittedData, selNum, className, param, method) {
  trainData = splittedData$training
  classIndex = getColumnIndex(data = trainData, className)
  mask = randomMask(colNum = ncol(trainData), classCol = classIndex, selNum = selNum)
  
  bestRate = rateClassifierWithMask(method, mask, splittedData, classIndex, className)
  for(i in 1:param) {
    neighbors = getNeighbors(ncol(trainData), classIndex, mask)
    if(length(neighbors) > 0) {
      randMask = selectRandom(neighbors)
      
      accuracyRate = rateClassifierWithMask(method, randMask, splittedData, classIndex, className)
      
      if(accuracyRate > bestRate){
        bestRate = accuracyRate
        mask = randMask
      }
    }
  }
  
  bestRate
}

# classification algorithms
kNN <- function(colIndex, splittedData) {
  className = getColumnName(data = splittedData$training, index = colIndex)
  formul = as.formula(paste(className,"~."))
  
  IBk(formul, data = splittedData$training)
}

SVM <- function(colIndex, splittedData) {
  className = getColumnName(data = splittedData$training, index = colIndex)
  formul = as.formula(paste(className,"~."))
  
  svm(formul, data = splittedData$training)
}

NB <- function(colIndex, splittedData) {
  naiveBayes(splittedData$training[,-colIndex], splittedData$training[,colIndex])
}

# study functions
rateClassifier <- function(method, splittedData, className) {
  colIndex <- getColumnIndex(data = splittedData$training, columnName = className)
  
  classifier <- method(colIndex, splittedData = splittedData)
  prediction <- predict(classifier, splittedData$test)
  
  correct <- sum(splittedData$test[,colIndex] == prediction)
  all <- nrow(splittedData$test)
  
  correct/all
}

rateClassifierWithMask <- function(method, mask, splittedData, classIndex, className) {
  maskedData = applyMaskOnData(mask, splittedData, classIndex)
  rateClassifier(method = method, splittedData = maskedData, className = className)
}

# function studying selection method on indicated data and classifiaction method
study <- function(data, selectMethod, param, method) {
  className = getColumnName(data$data, data$classIndex)
  splittedData <- splitData(data = data$data, percent = trainPercent)
  trainAttrNum = 1:(ncol(data$data)-1);
  
  accuracyRate = c()
  for(i in trainAttrNum) {
    qualityResults = double(length = selectionRepeatCount)
    for(j in 1:selectionRepeatCount) {
      quality = selectMethod(splittedData, i, className, param, method)
      qualityResults[j] = as.numeric(quality)
    }
    
    accuracyRate = append(accuracyRate, sum(qualityResults)/selectionRepeatCount)
  }
  
  print(accuracyRate)
  plot(trainAttrNum,accuracyRate, type = "p", col = "red", xlab = "number of training attributes", ylab = "accuracy rate")
}

studySelectionMethod <- function(data, method, param) {
  print("Support Vector Machines")
  study(data, method, param, SVM)
  
  print("Naive Bayes")
  study(data, method, param, NB)
  
  print("k-Nearest Neighbors")
  study(data, method, param, kNN)
}

# const
trainPercent <- 80
selectionRepeatCount <- 10
  
# main
main <- function() {
  uciData <- read.csv(file="D:/Github/classification-study/data/wine.data", head=FALSE, sep=",")
  data = list("data" = uciData, "classIndex" = 1)
  
  print("Random Walk")
  studySelectionMethod(data, randomWalk, param = c(5))
  
  print("Hill climbing")
  studySelectionMethod(data, hillClimbing, param = c(5))
}

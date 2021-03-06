# MOW project
#
# Classification Study
#
# author: Mateusz Zawi�lak
#

library(RWeka)
library(e1071)

SplitData <- function(data, percent) {
  # Args:
  #   data: Data which should be devided
  #   precent: Precent of training data set
  #
  # Returns:
  #   Devided data into two parts: training set and test set 
  
  # calculate columns count
  columns.count <- floor(nrow(data) * percent / 100)
  # get data random indexes
  random.indexes <- sample(seq(1, nrow(data)), columns.count, replace=FALSE)
  
  # select data
  train.set <- data[random.indexes, ]
  test.set <- data[-random.indexes, ]
  
  list(test = test.set, training = train.set)
}

GetColumnIndex <- function(data, column.name) {
  # Args:
  #   data: Data
  #   column.name: Column name
  #
  # Returns:
  #   Column index for the give column name
  
  match(column.name, colnames(data))
}

GetColumnName <- function(data, index) {
  # Args:
  #   data: Data
  #   index: Column index
  #
  # Returns:
  #   Column name for the give column index
  
  colnames(data)[index]
}

SelectRandom = function(neighbors) {
  # Args:
  #   neighbors: Set of neighbors
  #
  # Returns:
  #   Random neighbor from the given set of neighbors
  
  rand = sample(1:ncol(neighbors), 1)
  neighbors[, rand]
}

RandomMask <- function(col.num, class.col, sel.num) {
  # Args:
  #   col.num: Number of data columns
  #   class.col: Class column name
  #   sel.num: Size of the mask
  #
  # Returns:
  #   Generates random mask
  
  vector <- 1:col.num
  random.columns <- sample(vector[-class.col], sel.num, replace=F)
  
  sort(random.columns)
}

ApplyMaskOnData <- function(mask, splitted.data, class.index) {
  # Args:
  #   mask: Data mask
  #   splitted.data: Data
  #   class.index: Class attribute column index
  #
  # Returns:
  #   Filtered data
  
  mask <- append(mask, class.index)
  masked.data <- list(training = splitted.data$training[, mask], test = splitted.data$test[, mask])
  
  masked.data
}

GetNeighbors <- function(col.num, class.col, mask) {
  # Args:
  #   col.num: Number of data attributes
  #   class.col: Class attribute column name
  #   mask: Data mask
  #
  # Returns:
  #   Mask's neighbors
  
  all.columns <- seq(from = 1, to = col.num)
  left.columns <- all.columns[-mask]
  left.columns <- left.columns[left.columns != class.col]
  
  neighbors <- array(0, c(length(mask), length(left.columns)*length(mask)))
  
  if (length(left.columns) > 0) {
    for(i in 1:length(left.columns)) {
      for(j in 1:length(mask)) {
        neighbor <- mask
        neighbor[j] <- left.columns[i]
        
        neighbors[, (i-1)*length(mask)+j] <- sort(neighbor)
      }
    }
  }
  
  neighbors
}

# selection methods

RandomWalk <- function(splitted.data, sel.num, class.name, param, method) {
  # Random walk as an attribute selection method.
  #
  # Args:
  #   splitted.data: Data devided into two parts: training and test
  #   sel.num: Number of attributes which should be selected
  #   class.name: Class attribute column name
  #   param: Number of iterations
  #   method: Classification algorithm
  #
  # Returns:
  #   The best model's quality
  
  train.data <- splitted.data$training
  class.index <- GetColumnIndex(data = train.data, class.name)
  mask <- RandomMask(col.num = ncol(train.data), class.col = class.index, sel.num = sel.num)
  
  best.rate <- RateClassifierWithMask(method, mask, splitted.data, class.index, class.name)
  for(i in 1:param) {
    neighbors <- GetNeighbors(ncol(train.data), class.index, mask)
    if (length(neighbors) > 0) {
      mask <- SelectRandom(neighbors)
      
      accuracy.rate <- RateClassifierWithMask(method, mask, splitted.data, class.index, class.name)
      
      if (accuracy.rate > best.rate){
        best.rate <- accuracy.rate
      }
    }
  }
  
  best.rate
}

HillClimbing <- function(splitted.data, sel.num, class.name, param, method) {
  # Hill climbing as an attribute selection method.
  #
  # Args:
  #   splitted.data: Data devided into two parts: training and test
  #   sel.num: Number of attributes which should be selected
  #   class.name: Class attribute column name
  #   param: Number of iterations
  #   method: Classification algorithm
  #
  # Returns:
  #   The best model's quality
  
  train.data <- splitted.data$training
  class.index <- GetColumnIndex(data = train.data, class.name)
  mask <- RandomMask(col.num = ncol(train.data), class.col = class.index, sel.num = sel.num)
  
  best.rate = RateClassifierWithMask(method, mask, splitted.data, class.index, class.name)
  for(i in 1:param) {
    neighbors <- GetNeighbors(ncol(train.data), class.index, mask)
    if (length(neighbors) > 0) {
      rand.mask <- SelectRandom(neighbors)
      
      accuracy.rate <- RateClassifierWithMask(method, rand.mask, splitted.data, class.index, class.name)
      
      if (accuracy.rate > best.rate){
        best.rate <- accuracy.rate
        mask <- rand.mask
      }
    }
  }
  
  best.rate
}


SimulatedAnnealing <- function(splitted.data, sel.num, class.name, params, method) {
  # Simulated Annealing as an attribute selection method.
  # Algorithm parameters:
  #   L: The length of the period
  #   repetition.number: Number of iterations
  #   init.temp: Initial temperature
  #   temp.change: Temperature change size
  #   min.temp: Minimal temperature
  #
  # Args:
  #   splitted.data: Data devided into two parts: training and test
  #   sel.num: Number of attributes which should be selected
  #   class.name: Class attribute column name
  #   param: Algorithm parameters
  #   method: Classification algorithm
  #
  # Returns:
  #   The best model's quality
  
  train.data <- splitted.data$training
  class.index <- GetColumnIndex(data = train.data, class.name)
  
  # generate init result
  mask <- RandomMask(col.num = ncol(train.data), class.col = class.index, sel.num = sel.num)
  best.rate <- RateClassifierWithMask(method, mask, splitted.data, class.index, class.name)
  curr.rate <- best.rate
  
  # simulated annealing params
  temp <- params$init.temp
  repetition <- 0
  
  for(iter in 1:(params$repetition.number)) {
    for(i in 1:(params$L)) {
      neighbors <- GetNeighbors(ncol(train.data), class.index, mask)
      if (length(neighbors) > 0) {
        neighbor <- SelectRandom(neighbors)
        
        neighbor.rate <- RateClassifierWithMask(method, neighbor, splitted.data, class.index, class.name)
        delta <- neighbor.rate - curr.rate
        if (delta < 0) {
          mask <- neighbor
          curr.rate <- neighbor.rate
          
          if (neighbor.rate > best.rate) {
            best.rate <- neighbor.rate
          }
        } else {
          x <- runif (1, 0.0, 1.0)
          if (x < exp((0-delta)/temp)) {
            mask <- neighbor
            curr.rate <- neighbor.rate
          }
        }
      }
    }
    
    temp <- temp * params$temp.change
    
    if (temp <= params$min.temp) {
      break
    }
  }
  
  best.rate
}

# classification algorithms

kNN <- function(col.index, splitted.data) {
  # K nearest neighbours algorithm
  #
  # Args:
  #   col.index: Class attribute column index
  #   splitted.data: Data devided into two parts: training and test
  #
  # Returns:
  #   Classification model
  
  class.name <- GetColumnName(data = splitted.data$training, index = col.index)
  formul <- as.formula(paste(class.name,"~."))
  
  IBk(formul, data = splitted.data$training)
}

SVM <- function(col.index, splitted.data) {
  # Support vector machine algorithm
  #
  # Args:
  #   col.index: Class attribute column index
  #   splitted.data: Data devided into two parts: training and test
  #
  # Returns:
  #   Classification model
  
  class.name <- GetColumnName(data = splitted.data$training, index = col.index)
  formul <- as.formula(paste(class.name,"~."))
  
  svm(formul, data = splitted.data$training)
}

NB <- function(col.index, splitted.data) {
  # Naive Bayes classifier algorithm
  #
  # Args:
  #   col.index: Class attribute column index
  #   splitted.data: Data devided into two parts: training and test
  #
  # Returns:
  #   Classification model
  
  naiveBayes(splitted.data$training[, -col.index], splitted.data$training[, col.index])
}

# Study functions

RateClassifier <- function(method, splitted.data, class.name) {
  # Evaluates classification model
  #
  # Args:
  #   method: Classification algorithm
  #   splitted.data: Data devided into two parts: training and test
  #   class.name: class attribute column name
  #
  # Returns:
  #   Classification model quality
  
  col.index <- GetColumnIndex(data = splitted.data$training, column.name = class.name)
  
  classifier <- method(col.index, splitted.data = splitted.data)
  prediction <- predict(classifier, splitted.data$test)
  
  correct <- sum(splitted.data$test[, col.index] == prediction)
  all <- nrow(splitted.data$test)
  
  correct/all
}

RateClassifierWithMask <- function(method, mask, splitted.data, class.index, class.name) {
  # Evaluates classification algorithm on masked data
  #
  # Returns:
  #   Classification model quality
  
  maskedData <- ApplyMaskOnData(mask, splitted.data, class.index)
  RateClassifier(method = method, splitted.data = maskedData, class.name = class.name)
}

Study <- function(data, select.method, param, method) {
  # Function studies selection method on indicated data and classifiaction method.
  # Method draws results' plot.
  #
  # Args:
  #   data: Data
  #   select.method: Attributes selection method
  #   param: Attributes selection method's parameters
  #   method: Classification algorithm
  
  class.name <- GetColumnName(data$data, data$class.index)
  splitted.data <- SplitData(data = data$data, percent = kTrainPercent)
  trainAttrNum <- 1:(ncol(data$data)-1);
  
  accuracy.rate <- c()
  for(i in trainAttrNum) {
    qualityResults <- double(length = kSelectionRepeatCount)
    for(j in 1:kSelectionRepeatCount) {
      quality <- select.method(splitted.data, i, class.name, param, method)
      qualityResults[j] <- as.numeric(quality)
    }
    
    accuracy.rate <- append(accuracy.rate, sum(qualityResults)/kSelectionRepeatCount)
  }
  
  print(accuracy.rate)
  plot(trainAttrNum,accuracy.rate, type = "p", col = "red", xlab = "number of training attributes", ylab = "accuracy rate")
}

StudySelectionMethod <- function(data, method, param) {
  # Studies attributes selection method for selected classification algorithms.
  #
  # Args:
  #   data: Data
  #   method: Attributes selection algorithm
  #   param: Attributes selection method's parameters
  
  print("Support Vector Machines")
  Study(data, method, param, SVM)
  
  print("Naive Bayes")
  Study(data, method, param, NB)
  
  print("k-Nearest Neighbors")
  Study(data, method, param, kNN)
}

# constant values
kTrainPercent <- 80
kSelectionRepeatCount <- 10
  
# main
main <- function() {
  uci.data <- read.csv(file="D:/Github/classification-Study/data/wine.data", head=FALSE, sep=",")
  data <- list("data" = uci.data, "class.index" = 1)
  
  print("Random Walk")
  StudySelectionMethod(data, RandomWalk, param = c(5))
  
  print("Hill climbing")
  StudySelectionMethod(data, HillClimbing, param = c(5))
  
  print("Simulated Annealing")
  StudySelectionMethod(data, SimulatedAnnealing, param = list("L" = 10, "min.temp" = 30, "repetition.number" = 100, "init.temp" = 100, "temp.change" = 0.90))
}

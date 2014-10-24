Classification Study
====================

Study of the effect of attributes' selection on the quality of the classification models created with algorithms available in `R language`.

Subject interpretation
---------------------
The aim of the project is to investigate the effect of attribute selection type on the quality of classification models by using the algorithms available in `R language`.

Classification algorithms available in R require the indication of a set of attributes on which classification has to find a way of mapping the data into a set of predefined classes. This project will examine the impact of the selection of these attributes on the quality of the classification model.

There will be a matrix of data (read from `.csv` file) and a parameter specifying the number of attributes that will have to be selected by the studied selection algorithms. After selecting attributes the program will build the classification model based on data-training using one of the classification algorithms available in R language. Then algorithm will test the accuracy of the model using other available data and compare the results with the results obtained using other search strategy.

Quality measure
-------------------

The quality of the selection method will be evaluated based on the quality of the created classification model. This quality is determined by the ratio of *accuracy* - the percentage of test examples correctly classified by the model.

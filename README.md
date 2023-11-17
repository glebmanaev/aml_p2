# AML p2: Abovebaseline

## RNN: GRUs and LSTMs

Both end up producing the same results, always predicting the majority class. List of the things that I tried:
* Weighted loss function
* Discarding the class 3
* Oversampling the minority classes to have the same number of samples as the majority class
* 
* Increasing the number of parameters
* Decreasing the number of parameters
* Dropout
* Conbinations of the above

### Nothing worked!

Looks like it can't fit to the data.

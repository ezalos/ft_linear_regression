# ft_linear_regression

This project implement Linear Regression on multivariate data



![](assets/graph_demo.gif)



## Getting Started

You need `python3.7`.

* ```pip3 install requirements.txt --user```
  * Each package can also be installed with the following commands:
    * ```pip3 install numpy --user```
	* ```pip3 install pandas --user```
	* ```pip3 install matplotlib --user```

## Train on dataset:
* Simple example:
  * ```python3 train.py -v -s data/spacecraft_data.csv```
    * ```-v``` : Allows visual representation of the on going regression
    * ```-s``` : Will apply minmax to the data
* usage:
  * ```usage: python3 train.py -h```

## Predict dataset from a saved model:
* Simple example:
  * ```python3 predict.py -d data/spacecraft_data.csv -l pickles/spacecraft.pkl -v```
    * ```-v``` : Allows visual representation of the prediction
	* ```-d``` : Will use the specified dataset for prediction
    * ```-l``` : Will load the specified trained model for prediction
* usage:
  * ```usage: python3 predict.py -h```

## Acknowledgments

Thank you for using my project !

# substorm-detection

First download and install conda.

Create the conda environment by opening up a terminal in the /substorm-detection directory, then run 

```
conda env create -f environment.yml
```

In pycharm, select the environment by going to File>Settings>Project>Project Interpreter>Project Interpreter: 
substorm-detection

## TODO:
- visualization
    - make "semantic dictionary"
        - see which neurons are being activated in an example
        - visualize those neurons with activation maximization
        - other stuff from distill
    - graph cnn won't have to deal with missing data so visualization might be easier
- regression
    - ~~see what the model is choosing?~~
        - totally random
        - is this thrown off by outliers? would L1 loss help instead?
        - tried L1 Loss, didn't really help, pretty much no predictive power whatsoever, predictions in
        approximately the right range
    - why is it going so fast? compare with binary classification
        - make binary classification dataset
    - double check dataset
    - multiclass classification
    - initialize with binary classification weights
        - make binary classification dataset
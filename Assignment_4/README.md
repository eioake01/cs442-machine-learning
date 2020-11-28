# Assignment 4

For this assignment, we were asked to implement the RBF algorithm and design a model were we should predict the biological activity of antifilarial antimycin analogues based on their structure.

## Usage

You should make sure you have installed: python3,numpy and matplotlib.

Next, in the same folder have the python file (Assignment_4.py), the training file (training.txt), the test file (test.txt), the centres file (centersVector.txt), the learning rates file (learningRates.txt), the sigmas file (sgimas.txt) and the parameters file (parameters.txt).

After ensuring all of the above, you can run the programme as such:
```bash
$ python3 Assignment_4.py
```

## Output
As an output you will get 2 files: results.txt and weights.txt.

The file results.txt shows the training and testing error of the training data per epoch.

The file weights.txt shows the last value of the model's coefficients after the last epoch.

A visualization of the results.txt file will be generated as well.
# Assignment 3

The assignment called for creating a SOM using the Kohonen algorithm that can cluster the letters of the Roman Alphabet based on 16 numerical features.

## Usage

You should make sure you have installed: python3,numpy and matplotlib.

Next, in the same folder have the python file (Assignment_3.py), the parameters file (parameters.txt) and the data file (letter-recognition.txt)

After ensuring all of the above, you can run the programme as such:
```bash
$ python3 Assignment_3.py
```

## Output
As an output you will get 2 files: results.txt and clustering.txt

The file results.txt shows the training and testing error of the training data per epoch.

The file clustering.txt shows for each node of the grid, the label that it supports after training.

Also, the graphical representation of each file will be displayed.
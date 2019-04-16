# Neural Gas

Alghorithm for neural gas. It is used to find optimal data representations based on feature vectors.
There are two example data sets that you can use to test the alghorithm.

## Getting Started

Clone the repository or download zip version and copy "Neural_Gas.py" to the folder with your python script.
To run the code you have to import it first:

```
import * from Neural_Gas
```

Then import your a set of data from .csv file:

```
data = pd.read_csv("irisData.csv", header=0)
```

Initialize NeuralGas class with specified parameters:

```
neural_gas = Neural Gas(data, no_of_neurons=15, iterations=3000, display_animation=True, display_current_iteration=False,
              skip_frames_count=15)
```

Finally run the alghorithm:

```
neural_gas.run()
```

## List of parameters

* data - your set of data to analyze
* no_of_neurons - number of neurons you want to use to analyze the data (default 25)
* iterations - number of epochs in which neurons are going to train (default 1000)
* display_animation - boolean value - specify whether you want to watch neurons on plot during learning (default True)
* display_current_iteration - boolean value - print current epoch to the console (default True)
* skip_frames_count - if you want to reduce wait time during learning use it not to draw every epoch on plot (default 1)
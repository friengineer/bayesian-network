# Asia Respiratory Conditions Bayesian Network
A neural network for respiratory conditions in Asia to calculate the probability of events happening given some evidence.

A list of valid parameters for evidence and queries is below with descriptions.
- asia
- bron: bronchitis
- dysp: dyspnea
- either
- lung: lung cancer
- smoke: smoker
- tub: tuberculosis
- xray: lung x-ray contains abnormality

Execute the program by running `python asia.py <evidence arguments> <query arguments> <find exact solution> [<find approximate solution using Gibbs sampling> <number of samples> <calculate cross-entropy>]`.

`[]` denote arguments that are grouped together.

An example command is shown below, more examples can be found in [test_asia.py](test_asia.py).
```shell
$ python asia.py --evidence xray=1 dysp=0 --query tub lung bron --exact --gibbs -N 200000 --ent
```

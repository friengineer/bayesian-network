from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianModel
from pgmpy.inference import VariableElimination
from GibbsSamplingWithEvidence import GibbsSampling
import numpy as np
import argparse
import re
import math

# Parses the inputs from test_asia.py

parser = argparse.ArgumentParser(description='Asia Bayesian network')
parser.add_argument('--evidence', nargs='+', dest='eVars')
parser.add_argument('--query', nargs='+', dest='qVars')
parser.add_argument('-N', action='store', dest='N')
parser.add_argument('--exact', action="store_true", default=False)
parser.add_argument('--gibbs', action="store_true", default=False)
parser.add_argument('--ent', action="store_true", default=False)

args = parser.parse_args()
print(args)
print('\n-----------------------------------------------------------')

evidence={}
for item in args.eVars:
    evidence[re.split('=|:', item)[0]]=int(re.split('=|:',item)[1])

print('evidence:', evidence)

query = args.qVars
print('query:', args.qVars)

if args.N is not None:
    N = int(args.N)


# Using TabularCPD, define CPDs
cpd_asia = TabularCPD(variable='asia', variable_card=2, values=[[0.99], [0.01]])
cpd_tub = TabularCPD(variable='tub', variable_card=2, values=[[0.99, 0.95], [0.01, 0.05]], evidence=['asia'], evidence_card=[2])
cpd_smoke = TabularCPD(variable='smoke', variable_card=2, values=[[0.5], [0.5]])
cpd_lung = TabularCPD(variable='lung', variable_card=2, values=[[0.99, 0.9], [0.01, 0.1]], evidence=['smoke'], evidence_card=[2])
cpd_bron = TabularCPD(variable='bron', variable_card=2, values=[[0.7, 0.4], [0.3, 0.6]], evidence=['smoke'], evidence_card=[2])
cpd_either = TabularCPD(variable='either', variable_card=2, values=[[0.999, 0.001, 0.001, 0.001], [0.001, 0.999, 0.999, 0.999]], evidence=['lung', 'tub'], evidence_card=[2, 2])
cpd_xray = TabularCPD(variable='xray', variable_card=2, values=[[0.95, 0.02], [0.05, 0.98]], evidence=['either'], evidence_card=[2])
cpd_dysp = TabularCPD(variable='dysp', variable_card=2, values=[[0.9, 0.2, 0.3, 0.1], [0.1, 0.8, 0.7, 0.9]], evidence=['either', 'bron'], evidence_card=[2, 2])

# Define edges of the asia model
asia_model = BayesianModel([('asia', 'tub'),
                            ('smoke','lung'),
                            ('smoke','bron'),
                            ('tub','either'),
                            ('lung', 'either'),
                            ('bron', 'dysp'),
                            ('either', 'dysp'),
                            ('either', 'xray'),
                            ])

# Associate the parameters with the model structure.
asia_model.add_cpds(cpd_asia, cpd_tub, cpd_smoke, cpd_lung, cpd_bron, cpd_either, cpd_xray, cpd_dysp)

# Find exact solution if args.exact is True:
if args.exact:
    asia_infer = VariableElimination(asia_model)
    question = asia_infer.query(query, evidence=evidence)
    print("\n")

    for variable in query:
        print("Exact posterior probabilities for", variable, ":")
        print(variable, "_0 is false,", variable, "_1 is true")
        print(question[variable], "\n")

# Find approximate solution and cross entropy if args.gibbs is True:
if args.gibbs:
    gibbs_sampler = GibbsSampling(asia_model)
    samples = gibbs_sampler.sample(size=N, evidence=evidence)
    probabilities = []
    counter = 0

    for variable in query:
        occurences = 0

        # Counts the number of times the given query variable is true
        for i in range(0, N):
            occurences += samples.at[i, variable]

        # Divides the number of true occurences by the number of samples to give the approximate posterior probability
        probabilities.append(occurences / N)

        print("Approximate posterior probability for", variable, "_0:", 1 - probabilities[counter])
        print("Approximate posterior probability for", variable, "_1:", probabilities[counter])

        counter += 1

    # Calculates the total cross-entropy
    if args.ent:
        cross_entropy = 0
        counter = 0

        for variable in query:
            exact_cpd = question[variable]
            cross_entropy += ((1 - probabilities[counter]) * math.log10(exact_cpd.values[0])) + (probabilities[counter] * math.log10(exact_cpd.values[1]))
            counter += 1

        print("\nTotal cross-entropy:", -(cross_entropy))

print('\n-----------------------------------------------------------')

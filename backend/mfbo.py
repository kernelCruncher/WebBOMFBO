import pandas as pd
import numpy as np

import torch

from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP, SingleTaskGP
from botorch.posteriors.gpytorch import scalarize_posterior
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.models.cost import AffineFidelityCostModel
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.acquisition.knowledge_gradient import qMultiFidelityKnowledgeGradient
from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
from botorch.acquisition.utils import project_to_target_fidelity
from botorch.acquisition.max_value_entropy_search import qMultiFidelityMaxValueEntropy
from botorch.acquisition import PosteriorMean 
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.optim.optimize import optimize_acqf
import numpy as np
from scipy.spatial.distance import cdist
from botorch import fit_gpytorch_mll
torch.set_printoptions(precision=12, sci_mode=False)
import copy
import math
import matplotlib.pyplot as plt
import random
import time
import pickle
import os

import warnings
warnings.filterwarnings("ignore")


def runMes(model, Xrpr, previous_evaluations=None, train_x_past=None):
    fidelities = np.unique(Xrpr[:, -1])
    Xrpr = torch.tensor(Xrpr)
    bounds = torch.tensor([[0.0] * (Xrpr.shape[1] - 1), [1.0] * (Xrpr.shape[1]-1)])
    candidate_set_no_hf = bounds[0] + np.multiply(bounds[1] - bounds[0], torch.rand(10000,  Xrpr.shape[1] -1))
    candidate_set = torch.tensor(np.concatenate((candidate_set_no_hf, np.array([[random.choice(fidelities) for x in range(10000)]]).T), axis=1))
    target_fidelities = {6: 1.0}
            
    cost_model = AffineFidelityCostModel(fidelity_weights={6: 1.0}, fixed_cost=1.0)
    cost_aware_utility = InverseCostWeightedUtility(cost_model=cost_model)

    acquisition = qMultiFidelityMaxValueEntropy(
            model=model,
            cost_aware_utility=cost_aware_utility,
            project=lambda x: project_to_target_fidelity(X=x, target_fidelities=target_fidelities),
            candidate_set=candidate_set,
        )
    acquisitionScores =  acquisition.forward(Xrpr.reshape(-1,1, Xrpr.shape[1]))
    return acquisitionScores

def runKG(model, Xrpr, previous_evaluations=None, train_x_past=None):
    Xrpr = torch.tensor(Xrpr)
    bounds = torch.tensor([[0.0] * Xrpr.shape[1], [1.0] * Xrpr.shape[1]])
    target_fidelities = {1: 1.0}
            
    cost_model = AffineFidelityCostModel(fidelity_weights={1: 1.0}, fixed_cost=1.0)
    cost_aware_utility = InverseCostWeightedUtility(cost_model=cost_model)

    curr_val_acqf = FixedFeatureAcquisitionFunction(
        acq_function=PosteriorMean(model),
        d=Xrpr.shape[1],
        columns=[Xrpr.shape[1]-1],
        values=[1],
    )                
    _, current_value = optimize_acqf(
        acq_function=curr_val_acqf,
        bounds=bounds[:,:-1],
        q=1,
        num_restarts= 2,
        raw_samples=4
    )
    acquisition = qMultiFidelityKnowledgeGradient(
            model=model,
            cost_aware_utility=cost_aware_utility,
            project=lambda x: project_to_target_fidelity(X=x, target_fidelities=target_fidelities),
            current_value=current_value,
            num_fantasies= 5
        )
    acquisitionScores =  acquisition.evaluate(Xrpr.reshape(-1,1, Xrpr.shape[1]), bounds=bounds).detach()
    return acquisitionScores

def runEI(model, Xrpr, previous_evaluations, train_x_past=None):
    Xrpr = torch.tensor(Xrpr)
    acquisition = ExpectedImprovement(
            model=model,
            best_f= max(previous_evaluations)
        )
    
    acquisitionScores =  acquisition.forward(Xrpr.reshape(-1,1, Xrpr.shape[1]) ).detach()
    return acquisitionScores

def runTVR(model, Xrpr, previous_evaluations=None, train_x_past=None):
    Xrpr_hf = Xrpr[np.where(Xrpr[:, -1]==1)]
    # indices = np.where(train_x_past[:, 1] == 1)

    acquisition_scores = runEI(model, Xrpr_hf, previous_evaluations)
    max_hf_ind = acquisition_scores.argmax()

    index_in_xrpr = Xrpr.tolist().index(Xrpr_hf[max_hf_ind].tolist())
    Xrpr = torch.tensor(Xrpr)

    posterior = model.posterior(Xrpr)

    pcov = posterior.distribution.covariance_matrix
    p_var = posterior.variance
    hf_max_cov = pcov[index_in_xrpr]
    hf_max_var = hf_max_cov[index_in_xrpr]
    cost = Xrpr[:, -1]
    
    return hf_max_cov ** 2 / (p_var.reshape(-1) * hf_max_var * cost)   
    
# This approach transforms the fidelity column to be as described in the paper, i.e. [1] -> [0] and [0.1] -> 1
# We do this transformation repeatedly as we wish to keep the data as it is, since the output from these searches 
# are required to stick to a specific format so that the graphing functionality knows how to deal with it.
def runTVR_mod(model, Xrpr, previous_evaluations=None, train_x_past=None):
    X_rpr_transf = copy.deepcopy(Xrpr)

    for row in range(len(X_rpr_transf)):
        X_rpr_transf[row][-1] = 1 if X_rpr_transf[row][-1] != 1 else 0

    #Get hf data-points. 
    Xrpr_hf = X_rpr_transf[np.where(X_rpr_transf[:, -1]==0)]

    acquisition_scores = runEI(model, Xrpr_hf, previous_evaluations)
    max_hf_ind = acquisition_scores.argmax()

    index_in_xrpr = X_rpr_transf.tolist().index(Xrpr_hf[max_hf_ind].tolist())
    Xrpr_transf = torch.tensor(X_rpr_transf)

    posterior = model.posterior(Xrpr_transf)

    pcov = posterior.distribution.covariance_matrix
    p_var = posterior.variance
    hf_max_cov = pcov[index_in_xrpr]
    hf_max_var = hf_max_cov[index_in_xrpr]
    cost = Xrpr[:, -1]
    return  hf_max_cov ** 2 / (p_var.reshape(-1) * hf_max_var * torch.tensor(cost))   

def runCustom(model, Xrpr, previous_evaluations=None, train_x_past=None):
    mes = runMes(model, Xrpr, previous_evaluations, train_x_past)
    normalized_mes= mes / torch.sqrt(torch.sum(mes**2))
    tvr = runTVR(model, Xrpr, previous_evaluations, train_x_past)
    normalized_tvr = tvr / torch.sqrt(torch.sum(tvr**2))
    return normalized_mes + normalized_tvr

def optimiseAcquisitionFunction(sortedAcqusitionScores, domain, trainingData, index_store):
    # X_detached = trainingData.detach().numpy()
    # def checkFunction(candidate, set):
    #     for x in set:
    #         if np.array_equal(candidate[:-1], x):
    #             return True
    #     return False
    def checkIndexNotAlreadyEvaluated(candidate, set):
        return candidate in set
    
    for i in range(domain.shape[0]):
        if not checkIndexNotAlreadyEvaluated(sortedAcqusitionScores[i].item(), index_store):
            index_store.append(sortedAcqusitionScores[i].item())
            return domain[sortedAcqusitionScores[i], 0:-2], domain[sortedAcqusitionScores[i], -2], domain[sortedAcqusitionScores[i], -1]
            # , sortedAcqusitionScores[i]

def run_entire_cycle(train_x_full, 
                     train_obj, 
                     domain, 
                     fidelity_history, 
                     index_store, 
                     func,
                     sf=False, 
                     no_of_iterations=100000, 
                     allocated_budget=100000
                     ):
    train_x_full = copy.deepcopy(train_x_full)
    train_obj = copy.deepcopy(train_obj)
    fidelity_history = copy.deepcopy(fidelity_history)
    index_store = copy.deepcopy(index_store)
    
    domain_X_only = domain[:, 0:-1]
    budget_sum = sum(fidelity_history)
    iteration_counter = 0
    while budget_sum  <= allocated_budget - 1 and iteration_counter < no_of_iterations: 
        # The - 1 important in the budget (as well as the equal) as the check happens at the start and we only really care about high-fidelity points.
        # Consider a budget of 40, and when we hit the sum at 39. We would want the subsequent step to be the last
        # as at most we can add 1. If we instead only add 0.5, you could argue that stopping at 39.5 is premature
        # and we could go another step since it's possible to get another low-fidelity point, but this does not interest us.
        # It's the high-fidelity points we care about and then that would exceed the budget.
        model = SingleTaskGP(train_x_full, train_obj) if sf else SingleTaskMultiFidelityGP(train_x_full, train_obj, data_fidelity=6 )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)  
        acquisitionScores = func(model=model, Xrpr=domain_X_only, previous_evaluations = train_obj, train_x_past=train_x_full )
        sorted_acqusition_scores = acquisitionScores.argsort(descending=True)
        top_candidate, fidelity, evaluation = optimiseAcquisitionFunction(sorted_acqusition_scores, domain, train_x_full, index_store)
        fidelity_history.append(fidelity)
        np.append(top_candidate, fidelity)
        train_x_full = torch.cat([train_x_full, torch.tensor(np.append(top_candidate, fidelity)).unsqueeze(0)])
        train_obj = torch.cat([train_obj, torch.tensor([evaluation]).unsqueeze(-1)])
        iteration_counter+=1
        budget_sum += fidelity
        
    cumulative_cost = [fidelity_history[0]]
    for i in range(len(fidelity_history) - 1):
        cumulative_cost.append(cumulative_cost[-1] + fidelity_history[i+1])
    return train_x_full, train_obj, cumulative_cost, index_store
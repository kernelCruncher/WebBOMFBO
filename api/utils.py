import numpy as np
import torch

from botorch import test_functions
import numpy as np
import matplotlib.pyplot as plt
import random
import time
import pickle
import os

import warnings
warnings.filterwarnings("ignore")


def setUpSampleSpace(spaceSize=200, var = 1, lf_cost=0.1):
      lf_cost = torch.tensor([lf_cost])
      problem = test_functions.AugmentedHartmann(negate=True)
      Xpr_before = np.random.rand(spaceSize, 6)
      Xpr = [np.append(i, [1.0]) for i in Xpr_before ]
      Xpr_tensor = torch.tensor(Xpr)
      output = problem(X=Xpr_tensor).unsqueeze(-1)
      X_total_hf = torch.cat((Xpr_tensor, output), dim=1)

      domain = []
      for index, hf in enumerate(X_total_hf):
            domain.append(hf)
            value = torch.cat((lf_cost, (hf[-1] + random.gauss(0, var)).reshape(1)), dim=0)
            domain.append(torch.cat((Xpr_tensor[index][0:6], value), dim=0))
      
      domain = torch.stack(domain)
      timestr = time.strftime("%Y%m%d-%H%M%S")
      fileName = 'HartmannSampleSpaces/'+timestr + '.csv'
      os.makedirs('HartmannSampleSpaces/', exist_ok=True)
      np.savetxt(fileName, domain, delimiter=',')

      return fileName

def setUpInitialData(sampleSpaceName, initialSize=10, predefined_indices = None, sf=False, file=True):
      # The file argument is telling us whether we expect the sampleSpaceName to be a file or the actual domain is already in memory.
      # The predefined_indices argument us used in the batch case across multiple search-algorithms where we want 
      #  each element in the batch to have the same intitial set up so that we can compare the averages fairly.
      sampleSpace = np.loadtxt(sampleSpaceName, delimiter=',') if file else sampleSpaceName
      if predefined_indices is None:
            bad_range = True
            top_size = len(sampleSpace) //20
            hf_points = sampleSpace[np.where(sampleSpace[:, -2]==1)]
            top_5_percent = hf_points[hf_points[:, -1].argsort()[::-1]][0:top_size, 0]

            while bad_range:
                  bad_range = False
                  sampleSpace_hf = sampleSpace[np.where(sampleSpace[:, -2]==1)]
                  size = len(sampleSpace_hf)
                  index_store = random.sample(range(size), initialSize)
                  #This gets the high fidelity and low fidelity points in pairs if we're doing MF.
                  sampleSpace, index_store = (sampleSpace_hf, index_store) if sf else (sampleSpace, [2 * x  for x in  index_store] + [1 + 2 * x for x in index_store])
                  fidelity_history = sampleSpace[index_store, -2]
                  train_X = sampleSpace[index_store, :-1]
                  train_obj = sampleSpace[index_store, -1:]

                  #Do not have an intitial sample that includes the top 5% of points                  
                  for row in train_X:
                       if row[0] in top_5_percent:
                            bad_range=True
                            break
                  
            return torch.tensor(train_X), torch.tensor(train_obj), sampleSpace, index_store, fidelity_history.flatten().tolist()
      else:
            fidelity_history = sampleSpace[predefined_indices, -2]
            train_X = sampleSpace[predefined_indices, :-1]
            train_obj = sampleSpace[predefined_indices, -1:]
            return torch.tensor(train_X), torch.tensor(train_obj), sampleSpace, predefined_indices, fidelity_history.flatten().tolist()

def generate_batch_indices(sampleSpaceName, initialSize=5, batch_size=5):
      batch_index_store = []
      for batch_no in range(batch_size):
          _, _, _, index_store,_ = setUpInitialData(sampleSpaceName, initialSize, file=False)
          batch_index_store.append(index_store)
      return batch_index_store
     
     
# Required when we want to ensure that the sf has the same hf points in its intitial sampel as the mf case.
def convertMFDatatoSFData(sampleSpace, indexStore):
      sampleSpace_hf = sampleSpace[np.where(sampleSpace[:, -2]==1)]
      index_store = [x // 2 for x in indexStore if x % 2 == 0]
      return torch.tensor(sampleSpace_hf[index_store, : -1]), torch.tensor(sampleSpace_hf[index_store, -1:]), sampleSpace_hf, index_store, sampleSpace[index_store, 1].flatten().tolist()
    

def save_dictionary(dictionary, batch=False, root='HartmannSearchDictionaries'):
      os.makedirs(root, exist_ok=True)
      timestr = time.strftime("%Y%m%d-%H%M%S")
      fileName = root + '/' + 'Batch_' + timestr if batch else root + '/' + timestr
      with open(fileName, 'wb') as handle:
         pickle.dump(dictionary, handle)
      return fileName

def load_dictionary(file):
    with open(file, 'rb') as inp:
      output = pickle.load(inp)
      return output

def save_image(fig, root='Images/'):
      os.makedirs(root, exist_ok=True)
      timestr = time.strftime("%Y%m%d-%H%M%S")
      fig.savefig(f'{root}/{timestr}')
            
def create_correlation_dict(no_points, corr_parameters):
    range_100 = np.random.rand(no_points, 6)
    problem = test_functions.AugmentedHartmann(negate=True)
    high_fid = problem(torch.cat((torch.tensor(range_100), torch.ones(no_points).unsqueeze(-1)), dim=1))
    corr_dict = {'base': range_100, '1': high_fid}
    gaussian_noise = np.array([random.gauss(0, 50) for x in range(0, no_points) ])
    for n in corr_parameters:
        low_fid = np.add(high_fid, 1/n * gaussian_noise)
        correlation = np.corrcoef(high_fid, low_fid)[0,1]
        corr_dict[str(correlation)] = low_fid
    return corr_dict

def compute_correlation(domain):
      hf_points = np.where(domain[:, -2] == 1)
      lf_points = np.where(domain[:, -2] != 1)
      return np.corrcoef(domain[hf_points, -1], domain[lf_points, -1])[0,1]


import numpy as np
import torch
import torch.nn as nn
import torch.distributed as distributed
from torch.nn.functional import binary_cross_entropy_with_logits
from FastMBAR import FastMBAR
import warnings

class DistributedCLBase(nn.Module):
    """
    High-level class for doing contrastive learning with linear potential u0 and u1.
    The class should be inherited by the class with definitions of u0 and u1.
    """
    def __init__(self, n_mols, local_mol_ids, dtype=torch.float32):
        """
        Initialize. 
        Run this initialization in the child class.
        In the child class, the __init__ function may be rewritten, but it should call super().__init__(n_mols, local_mol_ids, dtype).
        Also in the child class, remember to set parameters as nn.Parameter to track the gradients at the correspoding device.
        
        Parameters
        ----------
        n_mols : int
            The total number of molecules.
        
        local_mol_ids : 1d array-like
            The molecule IDs that are locally stored and handled at the current rank.
            The molecule IDs should be integers within 0, 1, ..., n_mols - 1.
        
        dtype : torch.dtype
            The data type of the model parameters.
        
        """
        super().__init__()
        self.training_input = []
        for i in range(n_mols):
            self.training_input.append(None)
        self.local_mol_ids = sorted(set([int(x) for x in local_mol_ids])) # convert to list of integers, remove duplicates
        self.delta_fs = nn.Parameter(torch.zeros(n_mols, dtype=dtype)) # initialize delta_fs as zero
    
    def check_local_mol_ids(self):
        """
        Check if each mol_id is covered by one and only one rank.
        """
        binary_mol_ids = torch.zeros(self.n_mols, dtype=self.dtype, device=self.device)
        binary_mol_ids[self.local_mol_ids] = 1.0
        distributed.all_reduce(binary_mol_ids, op=distributed.ReduceOp.SUM)
        binary_mol_ids = binary_mol_ids.detach().cpu().numpy()
        for i in range(self.n_mols):
            if binary_mol_ids[i] == 0:
                warnings.warn(f'Molecule {i} is not covered by any rank!')
            elif binary_mol_ids[i] > 1:
                warnings.warn(f'Molecule {i} is covered by more than one rank!')
        return binary_mol_ids
    
    def set_training_input(self, mol_id):
        """
        Set training input.
        Set self.training_input[mol_id] as dicionary including variables for training.
        self.training_input[mol_id] should have keys: 'labels' and other keys for computing u0 and u1.
        Note the labels should be a 1d tensor including zeros and ones.
        """
        if mol_id in self.local_mol_ids:
            # define in the child class
            pass
    
    def u0(self, mol_id):
        # template
        if self.training_input[mol_id] is None:
            warnings.warn(f'Training input for molecule {mol_id} is None! Please set the training input for this molecule.')
            return None
        else:
            # define in the child class
            pass
    
    def u1(self, mol_id):
        # template
        if self.training_input[mol_id] is None:
            warnings.warn(f'Training input for molecule {mol_id} is None! Please set the training input for this molecule.')
            return None
        else:
            # define in the child class
            pass
    
    @property
    def n_mols(self):
        _n_mols = len(self.training_input)
        assert _n_mols == len(self.delta_fs)
        return _n_mols
    
    @property
    def device(self):
        _device = self.delta_fs.device
        return _device
    
    @property
    def dtype(self):
        _dtype = self.delta_fs.dtype
        return _dtype
    
    def compute_delta_fs(self, cuda=True, verbose=False, method='L-BFGS-B', update=True):
        delta_fs = torch.zeros(self.n_mols, dtype=self.dtype, device=self.device)
        for i in self.local_mol_ids:
            each = self.training_input[i]
            if each is None:
                warnings.warn(f'Training input for molecule {i} is None! Please set the training input for this molecule.')
                continue
            labels = each['labels']
            n1 = int(torch.sum(labels).item())
            n0 = int(labels.shape[0] - n1)
            A = np.stack([self.u0(i).detach().cpu().numpy().copy(), 
                          self.u1(i).detach().cpu().numpy().copy()], axis=0)
            fastmbar = FastMBAR(A, np.array([n0, n1]), cuda=cuda, bootstrap=False, verbose=verbose, method=method)
            delta_fs[i] = fastmbar.DeltaF[0, 1]
        # collect delta_Fs from all ranks
        distributed.all_reduce(delta_fs, op=distributed.ReduceOp.SUM)
        if update:
            self.delta_fs = nn.Parameter(delta_fs.to(dtype=self.dtype, device=self.device))
        return delta_fs
    
    def forward(self, reduction='mean', mol_weights=None):
        """
        Compute the loss function. 
        Note the returned loss is the loss of samples on the current rank (i.e. a local loss).
        """
        local_loss = torch.tensor(0.0, dtype=self.dtype, device=self.device)
        for i in self.local_mol_ids:
            each = self.training_input[i]
            if each is None:
                warnings.warn(f'Training input for molecule {i} is None! Please set the training input for this molecule.')
                continue
            labels = each['labels']
            n1 = torch.sum(labels).item()
            n0 = labels.shape[0] - n1
            nu = torch.tensor(n0 / n1, dtype=self.dtype, device=self.device)
            logit = -torch.log(nu) + self.u0(i) - self.u1(i) + self.delta_fs[i]
            if mol_weights is None:
                mol_weight_i = 1.0
            else:
                mol_weight_i = float(mol_weights[i])
            local_loss += binary_cross_entropy_with_logits(logit, labels, reduction=reduction) * mol_weight_i
        return local_loss
                  
            
        

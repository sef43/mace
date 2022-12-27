import torch
from typing import Tuple
from e3nn.util import jit
from mace.tools import utils, to_one_hot, atomic_numbers_to_indices
from ase.units import kJ, mol
from torch_nl import compute_neighborlist_n2




def simple_nl(positions: torch.Tensor, cutoff: float) -> Tuple[torch.Tensor, torch.Tensor]:
    
    N=positions.shape[0]
    
    distmat = torch.cdist(positions,positions)

    linked = distmat < cutoff
    
    M = torch.sum(linked) - N 
    
    nl0 = torch.zeros(M,dtype=torch.long)
    nl1 = torch.zeros(M,dtype=torch.long)
    
    k=0
    for i in range(N):
        for j in range(N):
            if i != j and linked[i,j]:
                nl0[k]=i
                nl1[k]=j

                k+=1

    shifts = torch.zeros(M,3)
    
    return torch.stack((nl0,nl1)), shifts



class MACE_openmm_NNP(torch.nn.Module):
    def __init__(self, model_path, positions, atomic_numbers, device="cpu"):
        super().__init__()

        self.device = torch.device(device)
        self.default_dtype=torch.float64
        torch.set_default_dtype(self.default_dtype)

        self.register_buffer("ev_to_kj_mol", torch.tensor(mol / kJ, device=self.device))

        dat = self._compile_model(model_path)

        self.mace_dict = self._create_mace_dict(dat, positions, atomic_numbers)

        self.model = dat["model"]
        self.r_max = dat["r_max"]


    def forward(self, positions):
        positions = 10.0*positions #nm -> angstoms

        self.mace_dict = self._update_mace_dict(positions)

        res = self.model(self.mace_dict, compute_force=False)
        interaction_energy = res["energy"]


        conversion_factor = self.ev_to_kj_mol
        if interaction_energy is None:
            interaction_energy = torch.tensor(0.0, device=self.device)

        # eV -> kJ/mol
        return interaction_energy * conversion_factor


    def _create_mace_dict(self, dat, positions, atomic_numbers):
        mace_dict = {}
        N=positions.shape[0]
        M=N*N-N

        mace_dict["ptr"]=torch.tensor([0,N],dtype=torch.long, device=self.device)
        mace_dict["cell"]=torch.zeros((3,3),dtype=torch.get_default_dtype(), device=self.device)
        mace_dict["batch"]=torch.zeros(N, dtype=torch.long, device=self.device)

        z_table=dat["z_table"]

        # one hot encoding of atomic number
        mace_dict["node_attrs"]=to_one_hot(
                torch.tensor(atomic_numbers_to_indices(atomic_numbers, z_table=z_table), dtype=torch.long, device=self.device).unsqueeze(-1),
                num_classes=len(z_table),
            )

        mace_dict["shifts"]=torch.zeros((M,3),dtype=torch.get_default_dtype(), device=self.device)
        mace_dict["unit_shifts"]=torch.zeros((M,3),dtype=torch.get_default_dtype(), device=self.device)
        mace_dict["positions"]=positions.to(self.device)
        mace_dict["edge_index"]=None

        return mace_dict
   

    def _update_mace_dict(self,positions):

        mapping, batch_mapping, shifts_idx = compute_neighborlist_n2(
           cutoff=self.r_max,
           pos=positions.to(dtype=self.default_dtype),
           cell=self.mace_dict["cell"],
           pbc=torch.tensor([False, False, False],device=self.device),
           batch=self.mace_dict["batch"])
        
        
        # mapping, shifts_idx = simple_nl(positions, self.r_max)
        # mapping=mapping.to(self.device)
        # shifts_idx=shifts_idx.to(self.device)

        # assume no PBC in this implementation

        # Eliminate self-edges that don't cross periodic boundaries
        # true_self_edge = mapping[0] == mapping[1]
        # true_self_edge &= torch.all(shifts_idx == 0, dim=1)
        # keep_edge = ~true_self_edge
        # Note: after eliminating self-edges, it can be that no edges remain in this system
        # sender = mapping[0][keep_edge]
        # receiver = mapping[1][keep_edge]
        # shifts_idx = shifts_idx[keep_edge]

        sender = mapping[0]
        receiver = mapping[1]
        edge_index = torch.stack((sender, receiver))

        self.mace_dict["positions"] = positions
        self.mace_dict["edge_index"] = edge_index
        self.mace_dict["shifts"] = shifts_idx

        return self.mace_dict

    def _compile_model(self, model_path):
        model = torch.load(model_path)
        res = {}
        res["model"] = jit.compile(model).to(self.device)
        res["z_table"] = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
        res["r_max"] = model.r_max
        return res




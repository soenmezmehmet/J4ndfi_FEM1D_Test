import math
from typing import Tuple
from dataclasses import dataclass
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch

from fem_1d.core import Fem1D, MaterialProperties, BoundaryConditions


torch.set_default_dtype(torch.float64)
nodes = torch.linspace(0, 70, 11).view(-1, 1)
conn_list = torch.tensor([[1, 3, 2], [3, 5, 4], [5, 7, 6], [7, 9, 8], [9, 11, 10]])
conn_list = conn_list[:, [0, 2, 1]]  # Reorder to [left, mid, right]

E = 2.1e11 * torch.ones(conn_list.size(0), 1)
# NOTE: Three cables with the same cross-sectional area
area = 3 * 89.9e-6 * torch.ones(conn_list.size(0), 1)
# NOTE: Density is dervied from weight per unit length
RHO = 0.861 / 89.9e-6
BODY_FORCE = RHO * 9.81
#BODY_FORCE = 0
mat = MaterialProperties(E=E, area=area, b=BODY_FORCE)

f_sur = torch.zeros(11)
f_sur[-1] = (300 + 630) * 9.81
#f_sur[-1] = 0
u_d = torch.tensor([0.])
drlt_dofs = torch.tensor([1])
boundary_conditions = BoundaryConditions(u_d=u_d, drlt_dofs=drlt_dofs, f_sur=f_sur)

fem = Fem1D(nodes, conn_list, mat, boundary_conditions, nqp=2)
fem.preprocess()
fem.solve()
fem.postprocess()
fem.report()
fem.plot()
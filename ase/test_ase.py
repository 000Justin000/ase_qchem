from ase import Atoms
from ase.visualize import view
from ase.optimize import BFGS
from ase.calculators.nwchem import NWChem
from ase.constraints import FixInternals
from ase.io import write

h2o = Atoms('HOH', positions=[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
h2o.calc = NWChem(xc='B3LYP')
c = FixInternals(bonds=[[1.0, [0,1]]])
h2o.set_constraint(c)
opt = BFGS(h2o)
opt.run(fmax=0.02)

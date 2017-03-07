from ase import Atoms
from ase.visualize import view
from ase.optimize import BFGS
from ase.calculators.emt import EMT
from ase.calculators.lammpsrun import LAMMPS
from ase.calculators.nwchem import NWChem
from ase.constraints import FixInternals, Hookean
from ase.io import read, write

#-----------------------------------------
# set atomic radius
#-----------------------------------------
radius = {'H':0.25, 'C':0.70, 'O':0.60}
#-----------------------------------------

#-----------------------------------------
# read the molecule
#-----------------------------------------
mol = read("1001.xyz")
#-----------------------------------------

#-----------------------------------------
# read the molecule, center in the cell
#-----------------------------------------
mol.cell = [20.0, 20.0, 20.0]
mol.center()
#-----------------------------------------

#-----------------------------------------
# set the constraint
#-----------------------------------------
natom = mol.get_number_of_atoms()
#-----------------------------------------
mol_atoms = mol.get_chemical_symbols()
#-----------------------------------------
fix_bonds = []
#-----------------------------------------
for i in range(0,natom):
    for j in range(i+1,natom):
        ratio = mol.get_distance(i,j) / (radius[mol_atoms[i]] + radius[mol_atoms[j]])
        if (ratio < 1.25):
            fix_bonds.append( [mol.get_distance(i,j), [i,j]] )
#-----------------------------------------
constraints = FixInternals(bonds = fix_bonds)
#-----------------------------------------
mol.set_constraint(constraints)
#-----------------------------------------

#-----------------------------------------
# classical MD with EMT
#-----------------------------------------
mol.calc = EMT()
opt = BFGS(mol, trajectory="1001.traj")
opt.run(fmax=0.1)
#-----------------------------------------

#-----------------------------------------
# classical MD with LAMMPS
#-----------------------------------------
# mol.calc = LAMMPS(tmp_dir="./lammps_tmp", )
# opt = BFGS(mol, trajectory="1001.traj")
# opt.run(fmax=0.1)
#-----------------------------------------

#-----------------------------------------
# quantum NWChem
#-----------------------------------------
# mol.calc = NWChem(xc='B3LYP', basis='STO-3G')
# opt = BFGS(mol)
# opt.run(fmax=0.1)
#-----------------------------------------

#-----------------------------------------
# show final molecule
#-----------------------------------------
# view(mol, viewer='Avogadro')
#-----------------------------------------

#--------------------------------------------------
import os, math
import openbabel, pybel, ase
#--------------------------------------------------
from ase import Atoms
from ase.visualize import view
from ase.optimize import BFGS
from ase.optimize import QuasiNewton
from ase.calculators.lj import LennardJones
from ase.calculators.emt import EMT
from ase.calculators.lammpsrun import LAMMPS
from ase.calculators.nwchem import NWChem
from ase.constraints import FixInternals, Hookean
#--------------------------------------------------

#--------------------------------------------------
def pyb2ase(pybmol):
    pybmol.write("pdb", "tmp.pdb", overwrite=True)
    asemol = ase.io.read("tmp.pdb")
    os.remove("tmp.pdb")
    #--------------
    return asemol

#--------------------------------------------------
def geomopt(pybmol, forcefield):
    FF = pybel._forcefields[forcefield]
    FF.Setup(pybmol.OBMol)
    EE = FF.Energy()
    dE = EE
    while(abs(dE / EE) > 1.0e-8):
        pybmol.localopt(forcefield=forcefield, steps=1000)
        dE = FF.Energy() - EE
        EE = FF.Energy()
    #--------------
    return pybmol

#--------------------------------------------------
def pybview(pybmol):
    pybmol.write("pdb", "tmp.pdb", overwrite=True)
    os.system("avogadro tmp.pdb")
    os.remove("tmp.pdb")

#--------------------------------------------------
def getRMSD(pybmol1, pybmol2):
    alg = openbabel.OBAlign(pybmol1.OBMol, pybmol2.OBMol)
    alg.Align()
    return alg.GetRMSD()


#################################################
#           main function start here            #
#################################################
#------------------------------------------------
#         read the molecule with pybel          #
#------------------------------------------------
pybmol = pybel.readfile("pdb", "1001.pdb").next()
#------------------------------------------------

#------------------------------------------------
#    geometry optimization MM using openbabel   #
#------------------------------------------------
pybmol = geomopt(pybmol, 'mmff94s')
#------------------------------------------------


#------------------------------------------------
mins = []
#------------------------------------------------
nrot = 36
#------------------------------------------------
rb1 = [7,6,14,19]
rb2 = [6,7,12,18]
#------------------------------------------------
for i in range(0, nrot):
#------------------------------------------------
    for j in range(0, nrot):
    #--------------------------------------------
        print i, j
        #--------------------------------------------
        molrot = pybmol.clone
        molrot.OBMol.SetTorsion(rb1[0],rb1[1],rb1[2],rb1[3], (2*math.pi/nrot)*i)
        molrot.OBMol.SetTorsion(rb2[0],rb2[1],rb2[2],rb2[3], (2*math.pi/nrot)*j)
        molrot = geomopt(molrot, 'mmff94s')
        #--------------------------------------------
        unique = True
        #--------------------------------------------
        for exmol in mins:
            if (getRMSD(exmol, molrot) < 0.05):
                unique = False
        if (unique == True):    
            mins.append(molrot)
        #--------------------------------------------

#--------------------------------------------
f = open("RMSD.dat", "w")
#--------------------------------------------
for i in range(0, len(mins)):
    for j in range(0, len(mins)):
        f.write("{:7.2f}".format(getRMSD(mins[i], mins[j])))
    f.write("\n")
#--------------------------------------------
f.close()
#--------------------------------------------


#------------------------------------------------
#          convert pymol to asemol              #
#------------------------------------------------
asemol = pyb2ase(pybmol)
#------------------------------------------------

#-----------------------------------------
# quantum NWChem
#-----------------------------------------
# asemol.calc = NWChem(xc='B3LYP', basis='STO-3G')
# opt = QuasiNewton(asemol, trajectory='1001.traj')
# opt.run(fmax=1.0)
#-----------------------------------------

#-----------------------------------------
# show final molecule
#-----------------------------------------
# view(asemol, viewer='Avogadro')
#-----------------------------------------

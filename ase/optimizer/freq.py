#--------------------------------------------------
import sys, os, math, numpy
import openbabel, pybel, ase
#--------------------------------------------------
from mpi4py import MPI
#--------------------------------------------------
from ase import Atoms
from ase.visualize import view
from ase.optimize import BFGS
from ase.optimize import QuasiNewton
from ase.calculators.lj import LennardJones
from ase.calculators.emt import EMT
from ase.calculators.lammpsrun import LAMMPS
from ase.calculators.qchem import QChem
from ase.constraints import FixInternals, Hookean
#--------------------------------------------------

#--------------------------------------------------
def pyb2ase(pybmol, pid):
    pybmol.write("pdb", "tmp"+"{:04d}".format(pid)+".pdb", overwrite=True)
    asemol = ase.io.read("tmp"+"{:04d}".format(pid)+".pdb")
    os.remove("tmp"+"{:04d}".format(pid)+".pdb")
    #--------------
    return asemol

#--------------------------------------------------
def geomOptMM(pybmol, MMFF, tol):
    FF=pybel._forcefields[MMFF]
    FF.Setup(pybmol.OBMol)
    EE = FF.Energy()
    dE = EE
    while(abs(dE / EE) > tol):
        pybmol.localopt(forcefield=MMFF, steps=1000)
        dE = FF.Energy() - EE
        EE = FF.Energy()
    return pybmol

#--------------------------------------------------
def pybview(pybmol, pid):
    pybmol.write("pdb", "tmp"+"{:04d}".format(pid)+".pdb", overwrite=True)
    os.system("avogadro tmp"+"{:04d}".format(pid)+".pdb")
    os.remove("tmp"+"{:04d}".format(pid)+".pdb")

#--------------------------------------------------
def getRMSD(pybmol1, pybmol2):
    alg = openbabel.OBAlign(pybmol1.OBMol, pybmol2.OBMol)
    alg.Align()
    return alg.GetRMSD()

#--------------------------------------------------
def getCoords(pybmol):
    coords = []
    for atom in pybmol:
        coords.append( list(atom.coords) )
    return coords

#--------------------------------------------------
def getPybmol(pybmol, coords):
    molr = pybmol.clone
    for atom, coord in zip(molr, coords):
        atom.OBAtom.SetVector(coord[0], coord[1], coord[2])
    return molr


#################################################
#           main function start here            #
#################################################


#------------------------------------------------
#                 customize area                #
#------------------------------------------------
jobname = str(sys.argv[1])
#------------------------------------------------
QMFUNC     = 'B3LYP'
DISPERSION = 'd3'
QMBASIS    = '6-31G*'
TASK       = 'frequency'
#------------------------------------------------
ertol = 1.0e-10
#------------------------------------------------

#------------------------------------------------
#           initialize mpi parameters           #
#------------------------------------------------
nproc = MPI.COMM_WORLD.Get_size()
iproc = MPI.COMM_WORLD.Get_rank()
#------------------------------------------------

#------------------------------------------------
opt_dir_name = "qchem_opt_"+jobname+"_"+QMFUNC+"_"+DISPERSION+"_"+QMBASIS
#------------------------------------------------

#-----------------------------------------
# frequency analysis with qchem
#-----------------------------------------
configId = list(range(0, 100))
configId_loc = configId[iproc::nproc]
#-----------------------------------------
for i in configId_loc:
    molfile = opt_dir_name+"/config_"+"{:04d}".format(i)+".pdb"
    if (os.path.exists(molfile)):
        pybmol = next(pybel.readfile("pdb", molfile))
        asemol = pyb2ase(pybmol, iproc)
        calc = QChem(xc=QMFUNC, 
                     disp=DISPERSION,
                     basis=QMBASIS,
                     task=TASK,
                     symmetry=False,
                     thresh=12,
                     scf_convergence=8,
                     maxfile=128,
                     mem_static=400,
                     mem_total=30000,
                     label="tmp_qchem"+"{:04d}".format(iproc)+"/qchem"+"{:04d}".format(i))
        calc.run(asemol)
#-----------------------------------------

#-----------------------------------------
MPI.COMM_WORLD.Barrier()
#-----------------------------------------

#-----------------------------------------
# show final molecule
#-----------------------------------------
# view(asemol, viewer='Avogadro')
#-----------------------------------------

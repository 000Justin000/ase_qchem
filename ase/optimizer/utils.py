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
def geomOptMM(pybmol, tcs, MMFF, tol):
    #----------------------------------------
    constraints = openbabel.OBFFConstraints()
    #----------------------------------------
    if tcs is not None:
        for tc in tcs:
            constraints.AddTorsionConstraint(tc[0][0],tc[0][1],tc[0][2],tc[0][3], tc[1]*(360/(2*math.pi)))
    #----------------------------------------
    FF = pybel._forcefields[MMFF]
    FF.Setup(pybmol.OBMol, constraints)
    FF.SetConstraints(constraints)
    #----------------------------------------
    EE = FF.Energy()
    dE = EE
    #----------------------------------------
    while(abs(dE / EE) > tol):
        #------------------------------------
        FF.ConjugateGradients(1000)
        dE = FF.Energy() - EE
        EE = FF.Energy()
    #--------------
    FF.GetCoordinates(pybmol.OBMol)
    #--------------
    return pybmol

# #--------------------------------------------------
# def geomOptMM(pybmol, MMFF, tol):
#     FF=pybel._forcefields[MMFF]
#     FF.Setup(pybmol.OBMol)
#     EE = FF.Energy()
#     dE = EE
#     while(abs(dE / EE) > tol):
#         pybmol.localopt(forcefield=MMFF, steps=1000)
#         dE = FF.Energy() - EE
#         EE = FF.Energy()
#     return pybmol

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

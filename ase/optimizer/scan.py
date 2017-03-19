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
        FF.ConjugateGradient(1000)
        dE = FF.Energy() - EE
        EE = FF.Energy()
    #--------------
    FF.GetCoordinates(pybmol.OBMol)
    #--------------
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
rb1 = [int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])]     # dihedral idx
rb2 = [int(sys.argv[6]), int(sys.argv[7]), int(sys.argv[8]), int(sys.argv[9])]     # dihedral idx
#------------------------------------------------
MMFF    = "mmff94s"
QMFUNC  = 'B3LYP'
QMBASIS = 'STO-3G'
#------------------------------------------------
MMtol = 1.0e-8
QMtol = 4.5e-1
ertol = 1.0e-10
#------------------------------------------------
nrot = int(sys.argv[10])             # number of angular discret
#------------------------------------------------

#------------------------------------------------
#           initialize mpi parameters           #
#------------------------------------------------
nproc = MPI.COMM_WORLD.Get_size()
iproc = MPI.COMM_WORLD.Get_rank()
#------------------------------------------------

#------------------------------------------------
#         read the molecule with pybel          #
#------------------------------------------------
pybmol = pybel.readfile("pdb", jobname+".pdb").next()
#------------------------------------------------

#------------------------------------------------
#  constrained optimization MM using openbabel  #
#------------------------------------------------
diangle = numpy.linspace(0.0, 2*math.pi, nrot, endpoint=False)
nblock = int(math.ceil(float(nrot)/nproc) + ertol)
diangle_loc = diangle[iproc*nblock:min((iproc+1)*nblock, nrot)]
#------------------------------------------------
dir_name = "qchem_scan_"+jobname+"_"+QMBASIS
#------------------------------------------------
if not os.path.isdir(dir_name):
    os.makedirs(dir_name)
#------------------------------------------------

#------------------------------------------------
#    constrained optimization QM using qchem    #
#------------------------------------------------
energies_loc = []
#------------------------------------------------
for i in range(0, len(diangle_loc)):
    #--------------------------------------------
    angle1 = pybmol.OBMol.GetTorsion(rb1[0],rb1[1],rb1[2],rb1[3])/360*(2*math.pi) + diangle_loc[i]
    angle2 = pybmol.OBMol.GetTorsion(rb2[0],rb2[1],rb2[2],rb2[3])/360*(2*math.pi)
    #--------------------------------------------
    molr = pybmol.clone
    molr.OBMol.SetTorsion(rb1[0],rb1[1],rb1[2],rb1[3], angle1)
    molr.OBMol.SetTorsion(rb2[0],rb2[1],rb2[2],rb2[3], angle2)
    molr = geomOptMM(molr, [[rb1, angle1],[rb2, angle2]], MMFF, MMtol)
    #--------------------------------------------
    asemol = pyb2ase(molr, iproc)
    #--------------------------------------------
    calc = QChem(xc=QMFUNC, 
                 disp='d3',
                 basis=QMBASIS,
                 task='optimization',
                 symmetry=False,
                 tcs=[[rb1, angle1],[rb2,angle2]],
                 thresh=12,
                 scf_convergence=8,
                 maxfile=128,
                 mem_static=400,
                 mem_total=4000,
                 label="tmp_qchem"+"{:04d}".format(iproc) + "/diangle_"+"{:5.3f}".format(diangle_loc[i]))
    asemol, E = calc.run(asemol)
    #----------------------------------------
    if asemol is None:
        print "Error:"+"diangle"+"{:5.3f}".format(diangle_loc[i])
    #----------------------------------------
    ase.io.write(dir_name+"/diangle_" + "{:5.3f}".format(diangle_loc[i]) + ".pdb", asemol)
    #--------------------------------------------
    energies_loc.append(("{:5.3f}".format(diangle_loc[i]), E))
    #--------------------------------------------

#--------------------------------------------
energies = MPI.COMM_WORLD.allgather(energies_loc)
#--------------------------------------------
if (iproc == 0):
#------------------------------------------------
    f = open(dir_name+"/diangle_energies", "w")
    #--------------------------------------------
    for i in range(0, len(energies)):
        for j in range(0, len(energies[i])):
            f.write("diangle_%s:    %15.7f\n" % (energies[i][j][0], energies[i][j][1]))
    #--------------------------------------------
    f.close()
    #--------------------------------------------

#------------------------------------------------
MPI.COMM_WORLD.Barrier()
#------------------------------------------------

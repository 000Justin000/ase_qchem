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
MMFF       = "mmff94s"
QMFUNC     = 'B3LYP'
DISPERSION = 'd3'
QMBASIS    = '6-31G*'
TASK       = 'optimization'
#------------------------------------------------
MMtol = 1.0e-8
QMtol = 4.5e-3
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
diangle = numpy.linspace(0.0, 2*math.pi, nrot, endpoint=False)
diangle_loc = diangle[iproc::nproc]
#------------------------------------------------
dir_name = "qchem_scan_"+jobname+"_"+QMFUNC+"_"+DISPERSION+"_"+QMBASIS
#------------------------------------------------
if not os.path.isdir(dir_name):
    try:
        os.makedirs(dir_name)
    except Exception:
        pass
#------------------------------------------------

#------------------------------------------------
#    constrained optimization QM using qchem    #
#------------------------------------------------
energies_loc = []
#------------------------------------------------
for rot_diangle1 in diangle_loc:
    for rot_diangle2 in diangle:
        #----------------------------------------
        angle1 = pybmol.OBMol.GetTorsion(rb1[0],rb1[1],rb1[2],rb1[3])/360*(2*math.pi) + rot_diangle1
        angle2 = pybmol.OBMol.GetTorsion(rb2[0],rb2[1],rb2[2],rb2[3])/360*(2*math.pi) + rot_diangle2
        #----------------------------------------
        molr = pybmol.clone
        molr.OBMol.SetTorsion(rb1[0],rb1[1],rb1[2],rb1[3], angle1)
        molr.OBMol.SetTorsion(rb2[0],rb2[1],rb2[2],rb2[3], angle2)
        molr = geomOptMM(molr, [[rb1, angle1],[rb2, angle2]], MMFF, MMtol)
        #----------------------------------------
        asemol = pyb2ase(molr, iproc)
        #----------------------------------------
        prefix = "theta1_"+"{:5.3f}".format(rot_diangle1)+"_theta2_"+"{:5.3f}".format(rot_diangle2)
        #----------------------------------------
        calc = QChem(xc=QMFUNC, 
                     disp=DISPERSION,
                     basis=QMBASIS,
                     task=TASK,
                     symmetry=False,
                     tcs=[[rb1, angle1],[rb2,angle2]],
                     thresh=12,
                     scf_convergence=8,
                     maxfile=128,
                     mem_static=400,
                     mem_total=4000,
                     label="tmp_qchem"+"{:04d}".format(iproc)+"/" + prefix)
        asemol, E = calc.run(asemol)
        #----------------------------------------
        if ((asemol is not None) and (E is not None)):
            energies_loc.append((rot_diangle1, rot_diangle2, E))
            ase.io.write(dir_name+"/" + prefix +".pdb", asemol)
            print "theta1: %5.3f,  theta2: %5.3f,  energy: %15.7f\n" % (rot_diangle1, rot_diangle2, E)
        else:
            print "theta1: %5.3f,  theta2: %5.3f,  optimization failed\n" % (rot_diangle1, rot_diangle2)
        #----------------------------------------

#--------------------------------------------
energies = MPI.COMM_WORLD.allgather(energies_loc)
#--------------------------------------------
if (iproc == 0):
#------------------------------------------------
    f = open(dir_name+"/energies", "w")
    #--------------------------------------------
    for i in range(0, len(energies)):
        for j in range(0, len(energies[i])):
            f.write("%5.3f  %5.3f  %15.7f\n" % (energies[i][j][0], energies[i][j][1], energies[i][j][2]))
    #--------------------------------------------
    f.close()
    #--------------------------------------------

#------------------------------------------------
MPI.COMM_WORLD.Barrier()
#------------------------------------------------

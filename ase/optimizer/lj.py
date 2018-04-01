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
from utils import *
#--------------------------------------------------



#################################################
#           main function start here            #
#################################################


#------------------------------------------------
#                 customize area                #
#------------------------------------------------
jobname = str(sys.argv[1])
#------------------------------------------------
ref = [int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])]     # dihedral idx
#------------------------------------------------
MMFF       = "mmff94s"
QMFUNC     = 'B3LYP'
DISPERSION = 'd3_op'
QMBASIS    = '6-311++G**'
# QMBASIS    = 'STO-3G'
TASK       = 'single_point'
#------------------------------------------------
MMtol = 1.0e-8
QMtol = 4.5e-3
ertol = 1.0e-10
#------------------------------------------------
distances = [3.0, 3.5, 4.0, 4.2, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0, 5.1, 5.2, 5.3, 5.4, 5.6, 5.7, 5.8, 5.9, 6.0, 6.5, 7.0, 8.0, 10.0]
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
pybmol = next(pybel.readfile("xyz", "opt/xyz/"+jobname+".xyz"))
#------------------------------------------------

#------------------------------------------------
distances_loc = distances[iproc::nproc]
#------------------------------------------------

#------------------------------------------------
dir_name = "qchem_"+jobname+"_"+QMFUNC+"_"+DISPERSION+"_"+QMBASIS
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
for distance in distances_loc:
    #----------------------------------------
    pybmol_disp = pybmol.clone
    #----------------------------------------
    # displacement along given direction
    #----------------------------------------
    coords = numpy.zeros((3,4))
    for i in range(0,4):
        coords[0,i] = pybmol.OBMol.GetAtom(ref[i]).GetVector().GetX()
        coords[1,i] = pybmol.OBMol.GetAtom(ref[i]).GetVector().GetY()
        coords[2,i] = pybmol.OBMol.GetAtom(ref[i]).GetVector().GetZ()
    #----------------------------------------
    coords = coords - numpy.mean(coords, axis=1).reshape((3,1))
    u,s,v = numpy.linalg.svd(coords)
    #----------------------------------------
    vec = u[:,-1] * distance
    pybmol_disp.OBMol.Translate(openbabel.vector3(vec[0], vec[1], vec[2]))
    #----------------------------------------
    asemol = pyb2ase(pybmol, iproc)
    asemol_disp = pyb2ase(pybmol_disp, iproc)
    #----------------------------------------
    prefix = jobname + "_distance_" + "{:05.3f}".format(distance)
    #----------------------------------------
    calc = QChem(xc=QMFUNC, 
                 disp=DISPERSION,
                 basis=QMBASIS,
                 task=TASK,
                 symmetry=False,
                 tcs=None,
                 opt_maxcycle=100,
                 opt_tol_grad=300,
                 opt_tol_disp=1,
                 opt_tol_e=100,
                 thresh=12,
                 scf_convergence=8,
                 maxfile=128,
                 mem_static=400,
                 mem_total=4000,
                 label="tmp_qchem"+"{:04d}".format(iproc)+"/" + prefix)
    #----------------------------------------
    E_10 = calc.run((asemol, asemol_disp), "_10")
    E_01 = calc.run((asemol_disp, asemol), "_01")
    E_11 = calc.run([asemol, asemol_disp], "_11")
    #----------------------------------------
    if ((E_10 is not None) and (E_01 is not None) and (E_11 is not None)):
        energies_loc.append((distance, E_11 - E_10 - E_01))
        print("distance: %05.3f, energy: %15.7f kJ/mol" % (distance, E_11 - E_10 - E_01))
        sys.stdout.flush()
    else:
        print("distance: %05.3f, interaction calculation failed" % (distance))
        sys.stdout.flush()
    #----------------------------------------

#------------------------------------------------
energies = MPI.COMM_WORLD.allgather(energies_loc)
energies = sum(energies, []) # flatten 2d array to 1d
energies.sort()
#------------------------------------------------
if (iproc == 0):
#------------------------------------------------
    f = open(dir_name+"/binding_energies", "w")
    #--------------------------------------------
    for i in range(0, len(energies)):
        f.write("%05.3f:  %15.7f\n" % (energies[i][0], energies[i][1]*2625.498844780508))
    #--------------------------------------------
    f.close()
    #--------------------------------------------

#------------------------------------------------
MPI.COMM_WORLD.Barrier()
#------------------------------------------------

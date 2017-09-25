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
rb1 = [int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])]     # dihedral idx
rb2 = [int(sys.argv[6]), int(sys.argv[7]), int(sys.argv[8]), int(sys.argv[9])]     # dihedral idx
#------------------------------------------------
QMFUNC     = 'wB97M-V'
DISPERSION = 'None'
QMBASIS    = 'aug-cc-pVTZ'
TASK       = 'single_point'
#------------------------------------------------
QMtol = 4.5e-3
ertol = 1.0e-10
#------------------------------------------------
nrot = int(sys.argv[10])             # number of angular discret
#------------------------------------------------
geom_path = str(sys.argv[11])
#------------------------------------------------

#------------------------------------------------
#           initialize mpi parameters           #
#------------------------------------------------
nproc = MPI.COMM_WORLD.Get_size()
iproc = MPI.COMM_WORLD.Get_rank()
#------------------------------------------------

#------------------------------------------------
if (jobname   == "path_1001"):
    rotlist = numpy.array([[360,  0], [310, 40], [250,70], [220,120], [190,160], [170,190],  [100,250], [ 30,290]])
elif (jobname == "path_1101"):
    rotlist = numpy.array([[360,  0], [310, 70], [250, 90], [230,150], [170,200], [ 80,240], [ 80,290], [ 70,300]])
elif (jobname == "path_1111"):
    rotlist = numpy.array([[360,  0], [320, 60], [270,100], [200,130], [160,160], [170,200], [120,280], [ 40,300]])
elif (jobname == "path_10n01"):
    rotlist = numpy.array([[360,  0], [300, 50], [240, 90], [210,140], [180,180], [150,200], [ 90,240], [ 50,300]])
elif (jobname == "path_11n01"):
    rotlist = numpy.array([[360,  0], [300, 70], [240,130], [200,170], [180,190], [150,210], [110,250], [ 50,320]])
elif (jobname == "path_11n11"):
    rotlist = numpy.array([[360,  0], [310, 40], [260, 90], [220,140], [180,180], [140,220], [ 90,260], [ 40,310]])
#------------------------------------------------
rotlist = rotlist - numpy.floor(rotlist/360)*360
if (iproc == 0):
    print(rotlist)
    sys.stdout.flush()
rotlist = rotlist / 360.0 * (2*math.pi)
#------------------------------------------------
diangles = []
#------------------------------------------------
for i in range(0, len(rotlist)):
#------------------------------------------------
    diangles.append([rotlist[i,0], rotlist[i,1]])
#------------------------------------------------
                
#------------------------------------------------
diangles_loc = diangles[iproc::nproc]
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

#-------------------------------------------------------
#    single point energy calculation QM using qchem    #
#-------------------------------------------------------
energies_loc = []
#------------------------------------------------
for diangle in diangles_loc:
    #----------------------------------------
    prefix = "theta1_"+"{:5.3f}".format(diangle[0])+"_theta2_"+"{:5.3f}".format(diangle[1])
    #----------------------------------------
    pybmol = next(pybel.readfile("pdb", geom_path + "/" + prefix + ".pdb"))
    #----------------------------------------
    asemol = pyb2ase(pybmol, iproc)
    #----------------------------------------
    calc = QChem(xc=QMFUNC, 
                 disp=DISPERSION,
                 basis=QMBASIS,
                 task=TASK,
                 symmetry=False,
                 thresh=12,
                 scf_convergence=8,
                 maxfile=128,
                 mem_static=400,
                 mem_total=4000,
                 label="tmp_qchem"+"{:04d}".format(iproc)+"/" + prefix)
    E = calc.run(asemol)
    #----------------------------------------
    if (E is not None):
        energies_loc.append((diangle[0], diangle[1], E))
        print("theta1: %5.3f,  theta2: %5.3f,  energy: %15.7f" % (diangle[0], diangle[1], E))
        sys.stdout.flush()
    else:
        print("theta1: %5.3f,  theta2: %5.3f,  single_point failed" % (diangle[0], diangle[1]))
        sys.stdout.flush()
    #----------------------------------------

#------------------------------------------------
energies = MPI.COMM_WORLD.allgather(energies_loc)
energies = sum(energies, []) # flatten 2d array to 1d
energies.sort()
#------------------------------------------------
if (iproc == 0):
#------------------------------------------------
    f = open(dir_name+"/energies", "w")
    #--------------------------------------------
    for i in range(0, len(energies)):
        f.write("%3.0f  %3.0f  %15.7f\n" % (energies[i][0]/(2*math.pi)*360, energies[i][1]/(2*math.pi)*360, energies[i][2]))
    #--------------------------------------------
    f.close()
    #--------------------------------------------

#------------------------------------------------
MPI.COMM_WORLD.Barrier()
#------------------------------------------------

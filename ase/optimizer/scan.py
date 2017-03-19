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
MMFF    = "mmff94s"
QMFUNC  = 'B3LYP'
QMBASIS = '6-31G*'
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
#    geometry optimization MM using openbabel   #
#------------------------------------------------
pybmol = geomOptMM(pybmol, None, MMFF, MMtol)
#------------------------------------------------

#------------------------------------------------
#  constrained optimization MM using openbabel  #
#------------------------------------------------
mins_loc = []
#------------------------------------------------
diangle = numpy.linspace(0.0, 2*math.pi, nrot, endpoint=False)
nblock = int(math.ceil(float(nrot)/nproc) + ertol)
diangle_loc = diangle[iproc*nblock:min((iproc+1)*nblock, nrot)]
#------------------------------------------------
for angle_i in diangle_loc:
#------------------------------------------------
    st_mins_loc = []
    #--------------------------------------------
    for angle_j in diangle:
    #--------------------------------------------
        molr = pybmol.clone
        molr.OBMol.SetTorsion(rb1[0],rb1[1],rb1[2],rb1[3], angle_i)
        molr.OBMol.SetTorsion(rb2[0],rb2[1],rb2[2],rb2[3], angle_j)
        molr = geomOptMM(molr, [[rb1, angle_i]], MMFF, MMtol)
        #--------------------------------------------
        unique = True
        #--------------------------------------------
        for exmol in st_mins_loc:
            if (getRMSD(exmol, molr) < 0.05):
                unique = False
        if (unique == True):    
            st_mins_loc.append(molr)
    #------------------------------------------------
    mins_loc.append(st_mins_loc)
    #------------------------------------------------

#------------------------------------------------
#    constrained optimization QM using qchem    #
#------------------------------------------------
for i in range(0, len(diangle_loc)):
    #--------------------------------------------
    st_energies_loc = []
    #--------------------------------------------
    dir_name = "qchem_scan_"+jobname+"_"+QMBASIS+"/diangle_"+"{:5.3f}".format(diangle_loc[i])
    #--------------------------------------------
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    #--------------------------------------------
    for j in range(0, len(mins_loc[i])):
        #----------------------------------------
        asemol = pyb2ase(mins_loc[i][j], iproc)
        calc = QChem(xc=QMFUNC, 
                     disp='d3',
                     basis=QMBASIS,
                     task='optimization',
                     symmetry=False,
                     tcs=[[rb1, diangle_loc[i]]],
                     thresh=12,
                     scf_convergence=8,
                     maxfile=128,
                     mem_static=400,
                     mem_total=4000,
                     label="tmp_qchem"+"{:04d}".format(iproc) + "/diangle_"+"{:5.3f}".format(diangle_loc[i])
                                                              + "/config_"+"{:04d}".format(j))
        asemol, E = calc.run(asemol)
        ase.io.write(dir_name+"/config_" + "{:04d}".format(j) + ".pdb", asemol)
        st_energies_loc.append(E)
    #--------------------------------------------
    f = open(dir_name+"/config_energies", "w")
    #--------------------------------------------
    for j in range(0, len(st_energies_loc)):
        #----------------------------------------
        f.write("config %04d:    %15.7f\n" % (j, st_energies_loc[j]))
    #--------------------------------------------



MPI.COMM_WORLD.Barrier()

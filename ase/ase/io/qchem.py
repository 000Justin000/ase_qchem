from ase.utils import StringIO
from ase.io import read
from ase.utils import basestring

def read_qchem_sp_output(filename):
    """Method to read final energy from a qchem single point energy output."""

    f = filename
    if isinstance(filename, basestring):
        f = open(filename)

    lines = f.readlines()

    Efinal = None

    i = 0
    #----------------------------------------------------
    while i < len(lines):
        #------------------------------------------------
        if lines[i].find("total energy =") >= 0:
            Efinal = float(lines[i].split()[4])
        elif lines[i].find("Total energy in") >= 0:
            Efinal = float(lines[i].split()[8])
        #------------------------------------------------
        i += 1
        #------------------------------------------------

    if isinstance(filename, basestring):
        f.close()

    return Efinal

def read_qchem_bsse_output(filename):
    """Method to read final energy from a qchem bsse binding energy output."""

    f = filename
    if isinstance(filename, basestring):
        f = open(filename)

    lines = f.readlines()

    Ebinding = None

    i = 0
    #----------------------------------------------------
    while i < len(lines):
        #------------------------------------------------
        if lines[i].find("DE, kJ/mol") >= 0:
            Ebinding = float(lines[i].split()[3])
        #------------------------------------------------
        i += 1
        #------------------------------------------------

    if isinstance(filename, basestring):
        f.close()

    return Ebinding




def read_qchem_opt_output(filename):
    """Method to read geometry and final energy from a qchem optimization output."""

    f = filename
    if isinstance(filename, basestring):
        f = open(filename)

    lines = f.readlines()

    atoms  = None
    Efinal = None

    i = 0
    natoms = -1
    max_cycle_reached = False
    while i < len(lines):
        #------------------------------------------------
        if (natoms == -1):
            if lines[i].find("NAtoms") >= 0:
                natoms = int(lines[i + 1].split()[0])
        #------------------------------------------------
        if lines[i].find("Final energy is") >= 0:
            Efinal = float(lines[i].split()[3])
        #------------------------------------------------
        if lines[i].find("OPTIMIZATION CONVERGED") >= 0:
            if (natoms == -1):
                raise ValueError("Have not find keyword: NAtoms")
            else:
                string = ''
                string += (str(natoms) + "\n")
                string += "\n"
                for j in range(5, natoms + 5):
                    xyzstring = lines[i + j]
                    content = xyzstring.split()
                    string += (content[1] + "    " + content[2] + "    " + content[3] + "    " + content[4] + "\n")
                atoms = read(StringIO(string), format='xyz')
                atoms.set_cell((0., 0., 0.))  # no unit cell defined
                i += natoms + 5
        elif lines[i].find("MAXIMUM OPTIMIZATION CYCLES REACHED") >= 0:
            max_cycle_reached = True
            break
        else:
            i += 1

    if max_cycle_reached:
        i = len(lines) - 1
        while i >= 0:
            if lines[i].find("Optimization Cycle:") >= 0:
                string = ''
                string += (str(natoms) + "\n")
                string += "\n"
                for j in range(4, natoms + 4):
                    xyzstring = lines[i + j]
                    content = xyzstring.split()
                    string += (content[1] + "    " + content[2] + "    " + content[3] + "    " + content[4] + "\n")
                atoms = read(StringIO(string), format='xyz')
                atoms.set_cell((0., 0., 0.))  # no unit cell defined

                Efinal = float(lines[i+7+natoms].split()[2])

                break
            else:
                i -= 1
                
    if isinstance(filename, basestring):
        f.close()

    return atoms, Efinal




def read_qchem(filename):
    """Method to read geometry from an qchem input file."""
    f = filename
    if isinstance(filename, basestring):
        f = open(filename)
    lines = f.readlines()

    # Find geometry region of input file.
    stopline = 0
    for index, line in enumerate(lines):
        if line.find("$molecule") >= 0:
            startline = index + 2
            stopline = -1
        elif (line.find("$end") >= 0 and stopline == -1):
            stopline = index
    # Format and send to read_xyz.
    xyz_text = '%i\n' % (stopline - startline)
    xyz_text += ' geometry\n'
    for line in lines[startline:stopline]:
        xyz_text += line
    atoms = read(StringIO(xyz_text), format='xyz')
    atoms.set_cell((0., 0., 0.))  # no unit cell defined

    if type(filename) == str:
        f.close()

    return atoms



def write_qchem(filename, atoms, comment=None):
    """Method to write nwchem coord file."""

    if isinstance(filename, basestring):
        f = open(filename, 'w')
    else:  # Assume it's a 'file-like object'
        f = filename

    if comment is not None:
        f.write('$comment\n')
        f.write(str(comment)+'\n')
        f.write('$end\n\n')


    f.write('$molecule\n')
    f.write('0  1\n')

    if (type(atoms) == list):
        for mol in atoms:
            f.write("--\n")
            f.write('0  1\n')
            for atom in mol:
                symbol = atom.symbol
                f.write(symbol + "    " + "{:+15.8f}".format(atom.position[0]) \
                               + "    " + "{:+15.8f}".format(atom.position[1]) \
                               + "    " + "{:+15.8f}".format(atom.position[2]) \
                               + "\n")
    else:
        mol = atoms
        for atom in mol:
            symbol = atom.symbol
            f.write(symbol + "    " + "{:+15.8f}".format(atom.position[0]) \
                           + "    " + "{:+15.8f}".format(atom.position[1]) \
                           + "    " + "{:+15.8f}".format(atom.position[2]) \
                           + "\n")

    f.write('$end\n\n')



def save_xyz(filename, atoms, comment=None):
    """Save to xyz file."""

    if isinstance(filename, basestring):
        f = open(filename, 'w')
    else:  # Assume it's a 'file-like object'
        f = filename

    if (type(atoms) == list):
        Natom = 0
        for mol in atoms:
            Natom += len(mol)

        f.write(str(Natom)+'\n')
        f.write(str(comment)+'\n')

        for mol in atoms:
            for atom in mol:
                symbol = atom.symbol
                f.write(symbol + "    " + "{:+15.8f}".format(atom.position[0]) \
                               + "    " + "{:+15.8f}".format(atom.position[1]) \
                               + "    " + "{:+15.8f}".format(atom.position[2]) \
                               + "\n")
    else:
        mol = atoms
        Natom = len(mol)

        f.write(str(Natom)+'\n')
        f.write(str(comment)+'\n')

        for atom in mol:
            symbol = atom.symbol
            f.write(symbol + "    " + "{:+15.8f}".format(atom.position[0]) \
                           + "    " + "{:+15.8f}".format(atom.position[1]) \
                           + "    " + "{:+15.8f}".format(atom.position[2]) \
                           + "\n")

    f.write('$end\n\n')

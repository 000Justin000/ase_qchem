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
        #------------------------------------------------
        i += 1
        #------------------------------------------------

    if isinstance(filename, basestring):
        f.close()

    return Efinal





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
        else:
            i += 1

    if isinstance(filename, str):
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
    for atom in atoms:
        symbol = atom.symbol
        f.write(symbol + ' ' + str(atom.position[0]) + ' ' + str(atom.position[1]) + ' ' + str(atom.position[2]) + '\n')
    f.write('$end\n\n')

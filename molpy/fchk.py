from . import export
from . import tools
import copy
import numpy as np


@export
class MolcasFCHK:

    def __init__(self, filename, mode):
        self.f = open(filename, mode)

    def write(self, wfn):
        wfn = copy.deepcopy(wfn)
        wfn.desymmetrize()
        self.write_header(
            'Molcas -> gaussian formatted checkpoint',
            'type',
            'method',
            'basis'
            )
        self.write_info(
            wfn.natoms_unique,
            wfn.charge,
            wfn.spin_multiplicity,
            wfn.nelec,
            wfn.nelec_a,
            wfn.nelec_b,
            wfn.nbas[0],
            )

        self.write_atom_info(
            wfn.center_charges,
            wfn.center_coordinates,
            )
        self.write_basisset(
            wfn.basisset,
            wfn.center_coordinates,
            )
        self.write_orbitals(
            wfn.nbas,
            tools.scalify(wfn.basis_function_ids),
            tools.scalify(wfn.mo_energies),
            tools.scalify(wfn.mo_occupations),
            tools.scalify(wfn.mo_vectors),
            wfn.uhf,
            tools.scalify(wfn.mo_energies_b),
            tools.scalify(wfn.mo_occupations_b),
            tools.scalify(wfn.mo_vectors_b),
            )

    def write_scalar_int(self, name, value):
        self.f.write('{:40s}   {:1s}     {:12d}\n'.format(name, 'I', value))

    def write_scalar_real(self, name, value):
        self.f.write('{:40s}   {:1s}     {:22.15e}\n'.format(name, 'I', value))

    def write_scalar_string(self, name, value):
        self.f.write('{:40s}   {:1s}     {:12s}\n'.format(name, 'C', value))

    def write_scalar_logical(self, name, value):
        self.f.write('{:40s}   {:1s}     {:1s}\n'.format(name, 'L', 'T' if value else 'F'))

    def write_array_int(self, name, array):
        n = array.size
        a = np.ravel(array, order='F')
        self.f.write('{:40s}   {:1s}   N={:12d}\n'.format(name, 'I', n))
        record_size = 6
        for offset in range(0, n, record_size):
            line = ''.join('{:12d}'.format(x) for x in a[offset:offset+record_size])
            print(line, file=self.f)

    def write_array_real(self, name, array):
        n = array.size
        a = np.ravel(array, order='F')
        self.f.write('{:40s}   {:1s}   N={:12d}\n'.format(name, 'R', n))
        record_size = 5
        for offset in range(0, n, record_size):
            line = ''.join('{:16.8e}'.format(x) for x in a[offset:offset+record_size])
            print(line, file=self.f)

    def write_array_string(self, name, array):
        n = array.size
        a = np.ravel(array, order='F')
        self.f.write('{:40s}   {:1s}   N={:12d}\n'.format(name, 'C', n))
        record_size = 5
        for offset in range(0, n, record_size):
            line = ''.join('{:12s}'.format(x) for x in a[offset:offset+record_size])
            print(line, file=self.f)

    def write_array_logical(self, name, array):
        n = array.size
        a = np.ravel(array, order='F')
        self.f.write('{:40s}   {:1s}   N={:12d}\n'.format(name, 'C', n))
        record_size = 72
        for offset in range(0, n, record_size):
            line = ''.join('{:1s}'.format(x) for x in a[offset:offset+record_size])
            print(line, file=self.f)

    def write_header(self, title, calctype, method, basis):
        self.f.write('{:72s}\n'.format(title))
        self.f.write('{:10s}{:30s}{:30s}\n'.format(calctype, method, basis))

    def write_info(self, natoms, charge, spin, nelec, nelec_a, nelec_b, nbas):
        self.write_scalar_int('Number of atoms', natoms)
        self.write_scalar_int('Charge', charge)
        self.write_scalar_int('Multiplicity', spin)
        self.write_scalar_int('Number of electrons', nelec)
        self.write_scalar_int('Number of alpha electrons', nelec_a)
        self.write_scalar_int('Number of beta electrons', nelec_b)
        self.write_scalar_int('Number of basis functions', nbas)

    def write_atom_info(self, charges, coords):
        self.write_array_int('Atomic numbers', charges.astype(np.int64))
        self.write_array_real('Nuclear charges', charges)
        self.write_array_real('Current cartesian coordinates', coords.T)

    def write_basisset(self, basisset, coords):
        contracted_shells = 0
        largest_contraction = 0
        highest_angmom = 0
        primitive_shells = 0
        for center in basisset:
            for angmom in center['angmoms']:
                highest_angmom = max(angmom['value'], highest_angmom)
                contracted_shells += len(angmom['shells'])
                for shell in angmom['shells']:
                    nprim = len(shell['exponents'])
                    primitive_shells += nprim
                    largest_contraction = max(nprim, largest_contraction)
        self.write_scalar_int('Number of contracted shells', contracted_shells)
        self.write_scalar_int('Number of primitive shells', primitive_shells)
        self.write_scalar_int('Highest angular momentum', highest_angmom)
        self.write_scalar_int('Largest degree of contraction', largest_contraction)
        shell_types = np.empty((contracted_shells,), order='F', dtype=np.int32)
        shell_to_atom_map = np.empty((contracted_shells,), order='F', dtype=np.int32)
        shell_coordinates = np.empty((3,contracted_shells), order='F')
        primitives_per_shell = np.empty((contracted_shells,), order='F', dtype=np.int32)
        exponents = np.empty((primitive_shells,), order='F')
        coefficients = np.empty((primitive_shells,), order='F')
        ishell = 0
        offset = 0
        for center in basisset:
            for angmom in center['angmoms']:
                highest_angmom = max(angmom['value'], highest_angmom)
                contracted_shells += len(angmom['shells'])
                for shell in angmom['shells']:
                    shell_types[ishell] = angmom['value'] * (-1)**(angmom['value']//2)
                    shell_to_atom_map[ishell] = center['id']
                    shell_coordinates[:,ishell] = coords[center['id']-1,:]
                    nprim = len(shell['exponents'])
                    primitives_per_shell[ishell] = nprim
                    exponents[offset:offset+nprim] = shell['exponents']
                    coefficients[offset:offset+nprim] = shell['coefficients']
                    offset += nprim
                    ishell += 1
        self.write_array_int('Shell types', shell_types)
        self.write_array_int('Number of primitives per shell', primitives_per_shell)
        self.write_array_int('Shell to atom map', shell_to_atom_map)
        self.write_array_real('Primitive exponents', exponents)
        self.write_array_real('Contraction coefficients', coefficients)
        self.write_array_real('Coordinates of each shell', shell_coordinates)

    def write_orbitals(self, nbas, ao_labels, ene, occ, coef, uhf, ene_b, occ_b, coef_b):
        if len(nbas) != 1:
            raise Error('orbitals contain symmetry!')
        else:
            nbas = nbas[0]

        ene = tools.safe_select(ene, np.zeros(nbas))

        ao_order = tools.argsort(ao_labels, rank=tools.rank_ao_tuple_molden)
        ao_labels = [ao_labels[i] for i in ao_order]
        self.write_scalar_int('Number of basis functions', nbas)
        self.write_array_real('Alpha Orbital Energies', ene)
        self.write_array_real('Alpha MO coefficients', coef[ao_order,:])
        if uhf:
            self.write_array_real('Beta Orbital Energies', ene_b)
            self.write_array_real('Beta MO coefficients', coef_b[ao_order,:])

from . import export
from .tools import *
from . import mh5
from . import molden
from . import inporb

import numpy as np
import re
import copy
from scipy import linalg


@export
class MolcasWFN:

    def __init__(self, filename=None):

        """ initialize the wavefunction from a file """

        if filename:
            # determine file format
            if (filename.endswith('.h5')):
                fid = mh5.MolcasHDF5(filename, 'r')
            else:
                fid = inporb.MolcasINPORB(filename, 'r')
                fid.read()
            # read in the wavefunction data
            self.read(fid)

    def read(self, f):

        """ read in the wavefunction data from an object """

        # general Molcas info
        self.molcas_version = maybe_get(f, 'molcas_version')
        self.molcas_module = maybe_get(f, 'molcas_module')

        # symmetry/basis information
        self.nsym = maybe_get(f, 'nsym')
        self.nbas = maybe_get(f, 'nbas')

        # early verify of data integrity of nsym/nbas
        if self.nbas is None:
            raise Exception('no basis set info available')
        else:
            if not self.nsym:
                print('no symmetry info, taking length of NBAS')
                self.nsym = len(self.nbas)
            else:
                if self.nsym != len(self.nbas):
                    raise Exception('corrupted data: NSYM != length of NBAS')
        self.irrep_labels = maybe_get(f, 'irrep_labels')

        # basis set information
        self.natoms_unique = maybe_get(f, 'natoms_unique')
        self.center_labels = maybe_get(f, 'center_labels')
        self.center_charges = maybe_get(f, 'center_charges')
        self.center_coordinates = maybe_get(f, 'center_coordinates')
        self.basis_function_ids = maybe_get(f, 'basis_function_ids', self.nbas)

        if self.center_labels is not None:
            self.center_idx = ordered_list(self.center_labels)

        # desymmetrization information
        self.natoms_all = maybe_get(f, 'natoms_all')
        self.desym_center_labels = maybe_get(f, 'desym_center_labels')
        self.desym_center_charges = maybe_get(f, 'desym_center_charges')
        self.desym_center_coordinates = maybe_get(f, 'desym_center_coordinates')
        self.desym_basis_function_ids = maybe_get(f, 'desym_basis_function_ids')

        if self.desym_center_labels is not None:
            self.desym_center_idx = ordered_list(self.desym_center_labels)

        self.desym_matrix = maybe_get(f, 'desym_matrix', self.nbas)

        if self.desym_center_labels is not None:
            self.desym_center_unique_idx = {}
            for idx, lbl in enumerate(self.desym_center_labels):
                unique_center = lbl[:lenin]
                unique_idx = self.center_idx[unique_center]
                element, = re.match('([^\d]+)', unique_center).groups()
                lbl_new = '{:6s}'.format(element + str(1+idx))
                self.desym_center_labels[idx] = lbl_new
                self.desym_center_unique_idx[lbl_new] = unique_idx
            self.desym_center_idx = ordered_list(self.desym_center_labels)

        # basis set for unique centers
        self.basisset = maybe_get(f, 'basisset')

        # AO matrices
        self.ao_overlap_matrix = maybe_get(f, 'ao_overlap_matrix', self.nbas)
        self.ao_fockint_matrix = maybe_get(f, 'ao_fockint_matrix', self.nbas)

        # wavefunction data
        self.uhf = safe_select(maybe_get(f, 'uhf'), False)
        self.mo_occupations, self.mo_occupations_b = maybe_get(f, 'mo_occupations', self.nbas, self.uhf)
        self.mo_energies, self.mo_energies_b = maybe_get(f, 'mo_energies', self.nbas, self.uhf)
        self.mo_typeindices, self.mo_typeindices_b = maybe_get(f, 'mo_typeindices', self.nbas, self.uhf)
        self.mo_vectors, self.mo_vectors_b = maybe_get(f, 'mo_vectors', self.nbas, self.uhf)

        # electron data
        self.spin_multiplicity = safe_select(maybe_get(f, 'ispin'), 1)
        if self.uhf:
            self.nelec_a = int(np.sum(np.sum(occ) for occ in self.mo_occupations))
            self.nelec_b = int(np.sum(np.sum(occ) for occ in self.mo_occupations_b))
            self.nelec = self.nelec_a + self.nelec_b
            self.spin_multiplicity = (self.nelec_a - self.nelec_b) + 1
        else:
            if self.mo_occupations is None:
                self.nelec = int(np.sum(self.center_charges))
            else:
                self.nelec = int(np.sum(np.sum(occ) for occ in self.mo_occupations))
            self.nelec_b = (self.nelec - (self.spin_multiplicity - 1)) // 2
            self.nelec_a = self.nelec - self.nelec_b

        if self.desym_center_charges is not None:
            self.charge = int(np.sum(self.desym_center_charges)) - self.nelec
        elif self.center_charges is not None:
            self.charge = int(np.sum(self.center_charges)) - self.nelec

        self.supsym_nsym = maybe_get(f, 'supsym_nsym')
        self.supsym_irrep_indices, self.supsym_irrep_indices_b = maybe_get(f, 'supsym_irrep_indices', self.nbas, self.uhf)
        self.supsym_irrep_labels = maybe_get(f, 'supsym_irrep_labels')
        if self.supsym_irrep_labels is not None:
            self.supsym_nsym = [len(self.supsym_irrep_labels)]

    def print_orbitals(self, ao_pattern=None, typeid_list=None, linewidth=10):

        """ prints the MO vectors as formatted columns """

        # early check on required data
        if self.mo_vectors is None:
            raise Exception('cannot print orbitals, no MO coefficients present!')

        # substitute optional data
        irrep_labels = safe_select(self.irrep_labels, np.array(['?']*self.nsym, dtype='U1'))
        basis_function_ids = safe_select(self.basis_function_ids, [None for nb in self.nbas])

        for mo_category, mo_typeindices, mo_occupations, mo_energies, mo_vectors, mo_irreps, \
                in zip(('Alpha', 'Beta') if self.uhf else ('', None),
                       (self.mo_typeindices, self.mo_typeindices_b),
                       (self.mo_occupations, self.mo_occupations_b),
                       (self.mo_energies, self.mo_energies_b),
                       (self.mo_vectors, self.mo_vectors_b),
                       (self.supsym_irrep_indices, self.supsym_irrep_indices_b),
                       ):

            if mo_category is None:
                continue

            mo_typeindices = safe_select(mo_typeindices, [np.array(['?']*nb, dtype='U1') for nb in self.nbas])
            mo_occupations = safe_select(mo_occupations, [np.zeros(nb) for nb in self.nbas])
            mo_energies = safe_select(mo_energies, [np.zeros(nb) for nb in self.nbas])

            mo_type = {
                'f': 'frozen',
                'i': 'inactive',
                '1': 'RAS1',
                '2': 'RAS2',
                '3': 'RAS3',
                's': 'secondary',
                'd': 'deleted',
                '?': 'unknown',
                }

            # time for output!
            if mo_category:
                print('{:s} Molecular Orbitals:'.format(mo_category))
                print()

            for isym, (irrep, norb, coef, bfns, occup, energy, typeidx) \
                in enumerate(zip(
                    irrep_labels,
                    self.nbas,
                    mo_vectors,
                    basis_function_ids,
                    mo_occupations,
                    mo_energies,
                    mo_typeindices,
                    )):

                # apply MO filter
                mo_indices = filter_mo_typeindices(typeidx, typeid_list)

                norb = len(mo_indices)
                if norb == 0:
                    continue

                coef = coef[:,mo_indices]
                occup = occup[mo_indices]
                energy = energy[mo_indices]
                typeidx = typeidx[mo_indices]

                # construct AO labels
                if bfns is not None:
                    labels = [format_ao_tuple(ao_tuple, self.center_labels) for ao_tuple in bfns]
                else:
                    labels = np.arange(coef.shape[0]).astype('U')

                # apply AO filter
                ao_indices = filter_ao_labels(labels, pattern=ao_pattern)
                if len(ao_indices) == 0:
                    continue

                if self.supsym_irrep_labels is None:
                    print("Symmetry species {:d} ({:s})".format(isym+1, irrep.strip()))
                print()
                for offset in range(0,norb,linewidth):
                    n = min(norb-offset,linewidth)
                    print_line("", 1+mo_indices[offset:offset+n])
                    print_line("Occupation", occup[offset:offset+n])
                    print_line("Energy", energy[offset:offset+n])
                    print_line("TypeIndex", [mo_type[idx] for idx in typeidx[offset:offset+n]])
                    print()
                    for ib in ao_indices:
                        print_line(labels[ib], coef[ib,offset:offset+n])
                    print()

    def guessorb(self):
        """
        generate a set of initial molecular orbitals
        """
        if self.mo_vectors is None:
            info('no initial MO vectors present for a guess, using identity matrix')
            self.mo_vectors = [np.identity((nb,nb), dtype='f8') for nb in self.nbas]

        if self.supsym_nsym is None:
            self.supsym_nsym = [1 for nb in self.nbas]

        if self.supsym_irrep_indices is None:
            self.supsym_irrep_indices = [np.zeros(nb) for nb in self.nbas]

        self.mo_energies = [np.zeros(nb) for nb in self.nbas]

        for nb, nsupsym, supsym_id, C_mo, E_mo, S_ao, F_ao in zip(
                self.nbas,
                self.supsym_nsym,
                self.supsym_irrep_indices,
                self.mo_vectors,
                self.mo_energies,
                self.ao_overlap_matrix,
                self.ao_fockint_matrix,
                ):
            Smat_ao = np.asmatrix(S_ao)
            Fmat_ao = np.asmatrix(F_ao)
            for isupsym in range(nsupsym):
                cols = np.where(supsym_id == isupsym)[0]
                Cmat = np.asmatrix(C_mo[:,cols])
                Smat_mo = Cmat.T * Smat_ao * Cmat
                # orthonormalize
                s,U = np.linalg.eigh(Smat_mo)
                U_lowdin = U * np.diag(1/np.sqrt(s)) * U.T
                Cmat = Cmat * U_lowdin
                # diagonalize metric Fock
                Fmat_mo = Cmat.T * Smat_ao.T * Fmat_ao * Smat_ao * Cmat
                f,U = np.linalg.eigh(Fmat_mo)
                Cmat = Cmat * U
                # copy back to correct supsym id
                C_mo[:,cols] = Cmat
                E_mo[cols] = f
            # finally, sort symmetry block by orbital energy
            mo_order = np.argsort(E_mo)
            supsym_id[:] = supsym_id[mo_order]
            E_mo[:] = E_mo[mo_order]
            C_mo[:,:] = C_mo[:,mo_order]

    def mulliken(self):
        """
        perform a mulliken population analysis
        """
        if self.nsym > 1:
            self.desymmetrize()


        mulliken_charges = np.zeros(len(self.center_charges))

        if self.mo_occupations is None:
            return mulliken_charges

        for nb, occ, C_mo, S_ao, ao_tuples, in zip(
                self.nbas,
                self.mo_occupations,
                self.mo_vectors,
                self.ao_overlap_matrix,
                self.basis_function_ids,
                ):
            Smat_ao = np.asmatrix(S_ao)
            Cmat = np.asmatrix(C_mo)
            D = Cmat * np.diag(occ) * Cmat.T
            DS = np.multiply(D, Smat_ao)
            for i, (ao, ao_tuple) in enumerate(zip(np.asarray(DS), ao_tuples)):
                pop = np.sum(ao)
                mulliken_charges[ao_tuple[0]-1] += pop

        mulliken_charges = self.center_charges - mulliken_charges
        return mulliken_charges

    def desymmetrize(self):
        """
        remove symmetry from the orbitals
        """

        if self.nsym == 1:
            info('orbitals already desymmetrized')
            return

        if self.desym_matrix is None:
            error('no orbital desymmetrization matrix present')

        self.irrep_labels = ['a']

        nbast = sum(self.nbas)

        # build a new basisset from the unique centers
        self.desym_basisset = []
        if self.basisset is not None:
            for idx, lbl in enumerate(self.desym_center_labels):
                unique_idx = self.desym_center_unique_idx[lbl]
                basisset_center = copy.deepcopy(self.basisset[unique_idx])
                basisset_center['id'] = idx + 1
                self.desym_basisset.append(basisset_center)
            self.basisset = self.desym_basisset

        # reorder the AOs
        ao_order = [i for i, ao_tuple in sorted(enumerate(self.desym_basis_function_ids), key=lambda x: rank_ao_tuple_molcas(x[1]))]
        self.desym_basis_function_ids = [self.desym_basis_function_ids[i] for i in ao_order]
        self.desym_matrix = self.desym_matrix[ao_order,:]

        # tie modified desym data to the regular properties
        self.natoms_unique = self.natoms_all
        self.center_charges = self.desym_center_charges
        self.center_coordinates = self.desym_center_coordinates
        self.center_labels = self.desym_center_labels
        self.center_idx = self.desym_center_idx
        self.basis_function_ids = [self.desym_basis_function_ids]

        if self.ao_overlap_matrix is not None:
            Smat = np.mat(linalg.block_diag(*self.ao_overlap_matrix))
            Dmat = np.mat(self.desym_matrix)
            self.ao_overlap_matrix = [Dmat * Smat * Dmat.T]

        if self.ao_fockint_matrix is not None:
            Fmat = np.mat(linalg.block_diag(*self.ao_fockint_matrix))
            Dmat = np.mat(self.desym_matrix)
            self.ao_fockint_matrix = [Dmat * Fmat * Dmat.T]

        # restricted or alpha MOs
        mo_energies = lst_to_arr(self.mo_energies)

        # resort the total MO list
        mo_typeindices = lst_to_arr(self.mo_typeindices)
        mo_energies = lst_to_arr(self.mo_energies)
        mo_occupations = lst_to_arr(self.mo_occupations)
        mo_order = []
        for typeid in ['f', 'i', '1', '2', '3', 's', 'd']:
            mo_set, = np.where(mo_typeindices == typeid)
            mo_set_order = np.argsort(mo_energies[mo_set])
            if typeid == '1' or typeid == '2' or typeid == '3':
                mo_set_order = np.argsort(mo_occupations[mo_set[mo_set_order]])[::-1]
            mo_order.append(mo_set[mo_set_order])
        mo_order = lst_to_arr(mo_order)

        self.mo_energies = [lst_to_arr(self.mo_energies)[mo_order]]
        self.mo_occupations = [lst_to_arr(self.mo_occupations)[mo_order]]
        self.mo_typeindices = [lst_to_arr(self.mo_typeindices)[mo_order]]

        if self.mo_vectors is not None:
            mo_vectors = np.empty((nbast,nbast), dtype='f8')
            for nb, nb_off, C_mo in zip(self.nbas, offsets(self.nbas), self.mo_vectors):
                Cmat = np.asmatrix(C_mo)
                cols = nb_off + np.arange(nb)
                Dmat = np.asmatrix(self.desym_matrix[:,cols])
                Cmat_desym = np.array(Dmat * Cmat)
                mo_vectors[:,cols] = Cmat_desym
            self.mo_vectors = [mo_vectors[:,mo_order]]

        if self.supsym_nsym is not None:
            self.supsym_irrep_indices = [lst_to_arr(self.supsym_irrep_indices)[mo_order]]
            self.supsym_irrep_labels = [lst_to_arr(self.supsym_irrep_labels)]

        # beta MOs for UHF case
        if self.uhf:
            mo_energies = lst_to_arr(self.mo_energies_b)

            # resort the total MO list
            mo_typeindices = lst_to_arr(self.mo_typeindices_b)
            mo_energies = lst_to_arr(self.mo_energies_b)
            mo_occupations = lst_to_arr(self.mo_occupations_b)
            mo_order = []
            for typeid in ['f', 'i', '1', '2', '3', 's', 'd']:
                mo_set, = np.where(mo_typeindices == typeid)
                mo_set_order = np.argsort(mo_energies[mo_set])
                if typeid == '1' or typeid == '2' or typeid == '3':
                    mo_set_order = np.argsort(mo_occupations[mo_set[mo_set_order]])[::-1]
                mo_order.append(mo_set[mo_set_order])
            mo_order = lst_to_arr(mo_order)

            self.mo_energies_b = [lst_to_arr(self.mo_energies_b)[mo_order]]
            self.mo_occupations_b = [lst_to_arr(self.mo_occupations_b)[mo_order]]
            self.mo_typeindices_b = [lst_to_arr(self.mo_typeindices_b)[mo_order]]

            if self.mo_vectors_b is not None:
                mo_vectors = np.empty((nbast,nbast), dtype='f8')
                for nb, nb_off, C_mo in zip(self.nbas, offsets(self.nbas), self.mo_vectors_b):
                    Cmat = np.asmatrix(C_mo)
                    cols = nb_off + np.arange(nb)
                    Dmat = np.asmatrix(self.desym_matrix[:,cols])
                    Cmat_desym = np.array(Dmat * Cmat)
                    mo_vectors[:,cols] = Cmat_desym
                self.mo_vectors_b = [mo_vectors[:,mo_order]]

            if self.supsym_nsym is not None:
                self.supsym_irrep_indices_b = [lst_to_arr(self.supsym_irrep_indices_b)[mo_order]]
                self.supsym_irrep_labels_b = [lst_to_arr(self.supsym_irrep_labels_b)]

        if self.supsym_nsym is not None:
            self.supsym_nsym = [sum(self.supsym_nsym)]

        self.nbas = [nbast]
        self.nsym = 1

    def grid_that(mo_vector, ao_tuples, bfn_dict):
        # construct r, theta, phi grids
        for coef, ao_tuple in zip(mo_vector, ao_tuples):
            c, n, l, m = ao_tuple
            coef * bfn_dict[ao_tuple](r[c], theta[c], phi[c])

    def basis_function_dict(basisset):
        bfn_dict = {}
        for center in basisset:
            for angmom in center['angmoms']:
                for shell in angmom['shells']:
                    def R(r):
                        s = numpy.zeros(r.size)
                        for c, e in zip(shell['coefficients'], shell['exponents']):
                            s += c * np.exp(-e * r**2)
                        return s
                    l = angmom['value']
                    for m in np.arange(-l, l + 1, 1):
                        def bfn(r, theta, phi):
                            return R(r) * sp.special.sph_harm(m, l, theta, phi)
                        bfn_dict[(center,shell,l,m)] = bfn
        return bfn_dict

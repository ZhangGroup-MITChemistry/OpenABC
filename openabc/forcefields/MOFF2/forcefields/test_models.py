import numpy as np
import pandas as pd
import pickle
import mdtraj
try:
    import openmm as mm
    import openmm.app as app
    import openmm.unit as unit
except ImportError:
    import simtk.openmm as mm
    import simtk.openmm.app as app
    import simtk.unit as unit
from openabc.forcefields.parsers import MOFFParser
from openabc.forcefields import MOFFMRGModel, HPSModel
from openabc.forcefields.functional_terms import ashbaugh_hatch_term, dh_elec_term
from openabc.lib import _amino_acids, _kcal_to_kj
from openabc.forcefields.MOFF2.lib import kB, VEP, EC, NA, _res_group_mappings
from openabc.forcefields.MOFF2.forcefields.parameters import HPS_sigma
from openabc.forcefields.MOFF2.forcefields.openmm_terms.multi_body_terms import density_spline_term_group_i
from openabc.forcefields.MOFF2.utils import select_native_pairs_all_ss_dssp
import sys
import os
from openabc.forcefields.MOFF2.forcefields.openmm_terms.nonbonded_terms import ashbaugh_hatch_with_gaussian_term # new AH+GAU

"""
Define some models for tests.
"""

__location__ = os.path.dirname(os.path.abspath(__file__))

class HPSMOFFCosAngleModel(MOFFMRGModel):
    """
    A model that mixes HPS nonbonded interactions with MOFF bonded interactions. 
    Also the harmonic angle in MOFF is replaced by a cosine angle potential. 
    Note this model only supports existing hydropathy parameters.
    """
    def add_protein_angles(self, force_group=2):
        """
        Use angle potential including cosine function to facilitate stable simulation.
        """
        M = 5
        angles = mm.CustomAngleForce(f'k_angle*{M**2}*(1-cos((theta-theta0)/{M}))')
        angles.addPerAngleParameter('k_angle')
        angles.addPerAngleParameter('theta0')
        for _, row in self.protein_angles.iterrows():
            a1 = int(row['a1'])
            a2 = int(row['a2'])
            a3 = int(row['a3'])
            k_angle = float(row['k_angle'])
            theta0 = float(row['theta0'])
            angles.addAngle(a1, a2, a3, [k_angle, theta0])
        angles.setUsesPeriodicBoundaryConditions(self.use_pbc)
        angles.setForceGroup(force_group)
        self.system.addForce(angles)
    
    def add_contacts(self, hydropathy_scale='Urry', epsilon=0.2*_kcal_to_kj, mu=1, delta=0.08, force_group=2):
        """
        Add nonbonded contacts with specified types of hydrophobic scales. 
        
        The raw hydropathy scale is scaled and shifted by: mu*lambda - delta
        
        Parameters
        ----------
        hydropathy_scale : str
            Hydropathy scale, can be KR or Urry. 
        
        epsilon : float or int
            Contact strength. 
        
        mu : float or int
            Hydropathy scale factor. 
        
        delta : float or int
            Hydropathy shift factor. 
        
        force_group : int
            Force group. 
            
        """
        print('Add nonbonded contacts.')
        resname_list = self.atoms['resname'].tolist()
        atom_types = [_amino_acids.index(x) for x in resname_list]
        if hydropathy_scale == 'KR':
            print('Use KR hydropathy scale.')
            df_contact_parameters = pd.read_csv(f'{__location__}/parameters/HPS_KR_parameters.csv')
        elif hydropathy_scale == 'Urry':
            print('Use Urry hydropathy scale.')
            df_contact_parameters = pd.read_csv(f'{__location__}/parameters/HPS_Urry_parameters.csv')
        else:
            sys.exit(f'Error: hydropathy scale {hydropathy_scale} cannot be recognized!')
        sigma_ah_map, lambda_ah_map = np.zeros((20, 20)), np.zeros((20, 20))
        for i, row in df_contact_parameters.iterrows():
            atom_type1 = _amino_acids.index(row['atom_type1'])
            atom_type2 = _amino_acids.index(row['atom_type2'])
            sigma_ah_map[atom_type1, atom_type2] = row['sigma']
            sigma_ah_map[atom_type2, atom_type1] = row['sigma']
            lambda_ah_map[atom_type1, atom_type2] = row['lambda']
            lambda_ah_map[atom_type2, atom_type1] = row['lambda']
        print(f'Scale factor mu = {mu} and shift delta = {delta}.')
        lambda_ah_map = mu*lambda_ah_map - delta
        force = ashbaugh_hatch_term(atom_types, self.exclusions, self.use_pbc, epsilon, sigma_ah_map, 
                                    lambda_ah_map, force_group)
        self.system.addForce(force)

    # --------------------------------------------------------
    # NEW — AH + Gaussian contacts
    # --------------------------------------------------------
    def add_contacts_ah_gau(self,
                            epsilon,
                            sigma_ah_map,
                            lambda_ah_map,
                            gauss_height_map,
                            gauss_delta_mu=0.25,
                            gauss_width=0.10,
                            force_group=2):

        # Map residues → atom type indexes (0–19)
        resname_list = self.atoms['resname'].tolist()
        atom_types = [_amino_acids.index(x) for x in resname_list]

        # Build unified AH + Gaussian contact force
        contact_force = ashbaugh_hatch_with_gaussian_term(
            atom_types          = atom_types,
            df_exclusions       = self.exclusions,
            use_pbc             = self.use_pbc,
            epsilon             = epsilon,
            sigma_ah_map        = sigma_ah_map,
            lambda_ah_map       = lambda_ah_map,
            gauss_height_map    = gauss_height_map,
            gauss_delta_mu      = gauss_delta_mu,
            gauss_width         = gauss_width,
            force_group         = force_group,
        )

        # Add to system
        self.system.addForce(contact_force)


    def add_dh_elec(self, ldby=1*unit.nanometer, dielectric_water=80.0, cutoff=3.5*unit.nanometer, force_group=3):
        """
        Add Debye-Huckel electrostatic interactions. 
        
        Parameters
        ----------
        ldby : Quantity
            Debye length. 
        
        dielectric_water : float or int
            Dielectric constant of water. 
        
        cutoff : Quantity
            Cutoff distance. 
        
        force_group : int
            Force group. 
        
        """
        print('Add Debye-Huckel electrostatic interactions.')
        print(f'Set Debye length as {ldby.value_in_unit(unit.nanometer)} nm.')
        print(f'Set water dielectric as {dielectric_water}.')
        charges = self.atoms['charge'].tolist()
        force = dh_elec_term(charges, self.exclusions, self.use_pbc, ldby, dielectric_water, cutoff, force_group)
        self.system.addForce(force)


class HPSMOFFCosAngleTestModel(MOFFMRGModel):
    """
    A model that mixes HPS nonbonded interactions with MOFF bonded interactions. 
    Also the harmonic angle in MOFF is replaced by a cosine angle potential. 
    Note this model supports user defined nonbonded contact parameters.
    """
    def add_protein_angles(self, force_group=2):
        """
        Use angle potential including cosine function to facilitate stable simulation.
        
        Parameters
        ----------
        force_group : int
            Force group. 
        
        """
        M = 5
        angles = mm.CustomAngleForce(f'k_angle*{M**2}*(1-cos((theta-theta0)/{M}))')
        angles.addPerAngleParameter('k_angle')
        angles.addPerAngleParameter('theta0')
        for _, row in self.protein_angles.iterrows():
            a1 = int(row['a1'])
            a2 = int(row['a2'])
            a3 = int(row['a3'])
            k_angle = float(row['k_angle'])
            theta0 = float(row['theta0'])
            angles.addAngle(a1, a2, a3, [k_angle, theta0])
        angles.setUsesPeriodicBoundaryConditions(self.use_pbc)
        angles.setForceGroup(force_group)
        self.system.addForce(angles)
    
    def add_contacts(self, epsilon, sigma_ah_map, lambda_ah_map, force_group=5):
        """
        HPS contact potential with epsilon, sigma, and lambda as input parameters. 
        
        Parameters
        ----------
        epsilon : float
            Energy scale.
        
        sigma_ah_map : 2d np.ndarray
            Sigma values. 
        
        lambda_ah_map : 2d np.ndarray
            Lambda values.
        
        force_group : int
            Force group. 
        
        """
        resname_list = self.atoms['resname'].tolist()
        atom_types = [_amino_acids.index(x) for x in resname_list]
        force = ashbaugh_hatch_term(atom_types, self.exclusions, self.use_pbc, epsilon, sigma_ah_map, lambda_ah_map, force_group)
        self.system.addForce(force)
   
    # --------------------------------------------------------
    # NEW — AH + Gaussian
    # --------------------------------------------------------
    def add_contacts_ah_gau(self,
                            epsilon,
                            sigma_ah_map,
                            lambda_ah_map,
                            gauss_height_map,
                            gauss_delta_mu=0.25,
                            gauss_width=0.10,
                            force_group=2):

        # Map residues → atom type indexes (0–19)
        resname_list = self.atoms['resname'].tolist()
        atom_types = [_amino_acids.index(x) for x in resname_list]

        # Build unified AH + Gaussian contact force
        contact_force = ashbaugh_hatch_with_gaussian_term(
            atom_types          = atom_types,
            df_exclusions       = self.exclusions,
            use_pbc             = self.use_pbc,
            epsilon             = epsilon,
            sigma_ah_map        = sigma_ah_map,
            lambda_ah_map       = lambda_ah_map,
            gauss_height_map    = gauss_height_map,
            gauss_delta_mu      = gauss_delta_mu,
            gauss_width         = gauss_width,
            force_group         = force_group,
        )

        # Add to system
        self.system.addForce(contact_force)


 
    def add_dh_elec(self, ldby=1*unit.nanometer, dielectric_water=80.0, cutoff=3.5*unit.nanometer, force_group=6):
        """
        Add Debye-Huckel electrostatic interactions. 
        
        Parameters
        ----------
        ldby : Quantity
            Debye length. 
        
        dielectric_water : float or int
            Dielectric constant of water. 
        
        cutoff : Quantity
            Cutoff distance. 
        
        force_group : int
            Force group. 
        
        """
        charges = self.atoms['charge'].tolist()
        force = dh_elec_term(charges, self.exclusions, self.use_pbc, ldby, dielectric_water, cutoff, force_group)
        self.system.addForce(force)


class _LegacyHPSTestModel(HPSModel):
    def add_contacts(self, epsilon, sigma_ah_map, lambda_ah_map, force_group=2):
        """
        HPS contact potential with epsilon, sigma, and lambda as input parameters. 
        
        Parameters
        ----------
        epsilon : float
            Energy scale.
        
        sigma_ah_map : 2d np.ndarray
            Sigma values. 
        
        lambda_ah_map : 2d np.ndarray
            Lambda values.
        
        force_group : int
            Force group. 
        
        """
        resname_list = self.atoms['resname'].tolist()
        atom_types = [_amino_acids.index(x) for x in resname_list]
        force = ashbaugh_hatch_term(atom_types, self.exclusions, self.use_pbc, epsilon, sigma_ah_map, lambda_ah_map, force_group)
        self.system.addForce(force)

    # NEW — AH + Gaussian
    def add_contacts_ah_gau(self,
                            epsilon,
                            sigma_ah_map,
                            lambda_ah_map,
                            gauss_height_map,
                            gauss_delta_mu=0.25,
                            gauss_width=0.10,
                            force_group=2):

        # Map residues → atom type indexes (0–19)
        resname_list = self.atoms['resname'].tolist()
        atom_types = [_amino_acids.index(x) for x in resname_list]

        # Build unified AH + Gaussian contact force
        contact_force = ashbaugh_hatch_with_gaussian_term(
            atom_types          = atom_types,
            df_exclusions       = self.exclusions,
            use_pbc             = self.use_pbc,
            epsilon             = epsilon,
            sigma_ah_map        = sigma_ah_map,
            lambda_ah_map       = lambda_ah_map,
            gauss_height_map    = gauss_height_map,
            gauss_delta_mu      = gauss_delta_mu,
            gauss_width         = gauss_width,
            force_group         = force_group,
        )

        # Add to system
        self.system.addForce(contact_force)


class MOFF2Model(HPSMOFFCosAngleTestModel):
    """
    MOFF2 model used by current IDP, OP, and MDP simulations.
    """
    default_parameter_file = os.path.join(__location__, 'parameters', 'MOFF2.pkl')

    @classmethod
    def from_folded_pdb(
        cls,
        aa_pdb,
        ca_pdb,
        native_pair_epsilon=6.0,
        k_bond=8000.0,
        r0_bond=0.386,
        box_a=1000,
        box_b=1000,
        box_c=1000,
        bond_force_group=1,
        angle_force_group=2,
        dihedral_force_group=3,
        native_pair_force_group=4,
        default_parse=True,
    ):
        """
        Build a MOFF2 model for an ordered protein from an atomistic PDB.

        The helper parses the atomistic PDB to a CA model, assigns DSSP labels,
        keeps native pairs only inside continuous ordered secondary-structure
        segments on the same chain, corrects HIS/terminal charges, creates the
        OpenMM system, and adds MOFF bonded/angle/dihedral/native-pair forces.
        Learned MOFF2 nonbonded and density terms should then be added with
        add_moff2_forces().
        """
        aa_pdb = os.fspath(aa_pdb)
        ca_pdb = os.fspath(ca_pdb)

        parser = MOFFParser.from_atomistic_pdb(
            aa_pdb,
            ca_pdb,
            default_parse=default_parse,
        )

        ref_traj = mdtraj.load_pdb(aa_pdb)
        assert ref_traj.n_frames == 1
        dssp = mdtraj.compute_dssp(ref_traj)[0].tolist()
        dssp = [x for x in dssp if x != 'NA']

        ca_traj = mdtraj.load_pdb(ca_pdb)
        assert len(dssp) == ca_traj.n_atoms
        parser.atoms['dssp'] = dssp

        parser.protein_bonds.loc[:, 'k_bond'] = k_bond
        parser.protein_bonds.loc[:, 'r0'] = r0_bond

        old_native_pairs = parser.native_pairs
        new_native_pairs = pd.DataFrame(columns=old_native_pairs.columns)
        for _, row in old_native_pairs.iterrows():
            a1 = int(row['a1'])
            a2 = int(row['a2'])
            if a1 > a2:
                a1, a2 = a2, a1
            c1 = parser.atoms.loc[a1, 'chainID']
            c2 = parser.atoms.loc[a2, 'chainID']
            if (c1 == c2) and (dssp[a1] in ['H', 'E']):
                if all([x == dssp[a1] for x in dssp[a1:a2 + 1]]):
                    new_native_pairs.loc[len(new_native_pairs.index)] = row
        new_native_pairs.loc[:, 'epsilon'] = native_pair_epsilon
        parser.native_pairs = new_native_pairs
        parser.parse_exclusions()

        assert ca_traj.n_chains == 1
        flag = parser.atoms['resname'] == 'HIS'
        parser.atoms.loc[flag, 'charge'] = 0.5
        charge = parser.atoms['charge'].to_numpy()
        charge[0] += 1
        charge[-1] -= 1
        parser.atoms['charge'] = charge

        model = cls()
        model.append_mol(parser)
        top = app.PDBFile(ca_pdb).getTopology()
        model.create_system(top=top, box_a=box_a, box_b=box_b, box_c=box_c)
        model.add_protein_bonds(force_group=bond_force_group)
        model.add_protein_angles(force_group=angle_force_group)
        model.add_protein_dihedrals(force_group=dihedral_force_group)
        model.add_native_pairs(force_group=native_pair_force_group)
        return model

    @classmethod
    def from_mdp_pdb(
        cls,
        aa_pdb,
        ca_pdb,
        mdp_od_csv,
        protein_name=None,
        native_pair_epsilon=6.0,
        k_bond=8000.0,
        r0_bond=0.386,
        box_a=1000,
        box_b=1000,
        box_c=1000,
        bond_force_group=1,
        angle_force_group=2,
        dihedral_force_group=3,
        native_pair_force_group=4,
        default_parse=True,
    ):
        """
        Build a MOFF2 model for a multidomain protein from an atomistic PDB.

        The helper reads ordered-domain ranges from mdp_od_csv, keeps angles
        and dihedrals only inside annotated ordered domains, keeps native pairs
        only within the same ordered domain, corrects HIS/terminal charges,
        creates the OpenMM system, and adds MOFF bonded/angle/dihedral/native
        pair forces. Learned MOFF2 nonbonded and density terms should then be
        added with add_moff2_forces().
        """
        aa_pdb = os.fspath(aa_pdb)
        ca_pdb = os.fspath(ca_pdb)
        mdp_od_csv = os.fspath(mdp_od_csv)
        if protein_name is None:
            protein_name = os.path.splitext(os.path.basename(aa_pdb))[0]

        parser = MOFFParser.from_atomistic_pdb(
            aa_pdb,
            ca_pdb,
            default_parse=default_parse,
        )

        parser.protein_bonds.loc[:, 'k_bond'] = k_bond
        parser.protein_bonds.loc[:, 'r0'] = r0_bond

        flag = parser.atoms['resname'] == 'HIS'
        parser.atoms.loc[flag, 'charge'] = 0.5
        charge = parser.atoms['charge'].to_numpy()
        charge[0] += 1
        charge[-1] -= 1
        parser.atoms['charge'] = charge

        df_mdp_od = pd.read_csv(mdp_od_csv).set_index('name')
        if protein_name not in df_mdp_od.index:
            raise KeyError(f'{protein_name} not found in {mdp_od_csv}')

        ca_domain_dict = {i: None for i in range(len(parser.atoms.index))}
        ordered_domains = str(df_mdp_od.loc[protein_name, 'ODs']).split(', ')
        for domain_i, each_domain in enumerate(ordered_domains):
            start_index, end_index = each_domain.split('-')
            start_index = int(start_index)
            end_index = int(end_index)
            for residue_i in range(start_index, end_index + 1):
                ca_domain_dict[residue_i] = domain_i

        traj = mdtraj.load_pdb(aa_pdb)
        assert traj.n_frames == 1
        dssp = mdtraj.compute_dssp(traj)[0].tolist()
        assert 'NA' not in dssp

        ca_traj = mdtraj.load_pdb(ca_pdb)
        assert len(dssp) == ca_traj.n_atoms

        new_angles = pd.DataFrame(columns=parser.protein_angles.columns)
        for _, row in parser.protein_angles.iterrows():
            a1 = int(row['a1'])
            a2 = int(row['a2'])
            a3 = int(row['a3'])
            flag1 = ca_domain_dict[a1] is not None
            flag2 = ca_domain_dict[a1] == ca_domain_dict[a2]
            flag3 = ca_domain_dict[a1] == ca_domain_dict[a3]
            if flag1 and flag2 and flag3:
                new_angles.loc[len(new_angles.index)] = row

        new_dihedrals = pd.DataFrame(columns=parser.protein_dihedrals.columns)
        for _, row in parser.protein_dihedrals.iterrows():
            a1 = int(row['a1'])
            a2 = int(row['a2'])
            a3 = int(row['a3'])
            a4 = int(row['a4'])
            flag1 = ca_domain_dict[a1] is not None
            flag2 = ca_domain_dict[a1] == ca_domain_dict[a2]
            flag3 = ca_domain_dict[a1] == ca_domain_dict[a3]
            flag4 = ca_domain_dict[a1] == ca_domain_dict[a4]
            if flag1 and flag2 and flag3 and flag4:
                new_dihedrals.loc[len(new_dihedrals.index)] = row

        parser.protein_angles = new_angles
        parser.protein_dihedrals = new_dihedrals

        native_pairs = select_native_pairs_all_ss_dssp(parser.native_pairs, aa_pdb)
        new_native_pairs = pd.DataFrame(columns=native_pairs.columns)
        for _, row in native_pairs.iterrows():
            a1 = int(row['a1'])
            a2 = int(row['a2'])
            if (
                ca_domain_dict[a1] is not None and
                ca_domain_dict[a1] == ca_domain_dict[a2]
            ):
                new_native_pairs.loc[len(new_native_pairs.index)] = row
        new_native_pairs.loc[:, 'epsilon'] = native_pair_epsilon
        parser.native_pairs = new_native_pairs
        parser.parse_exclusions()

        model = cls()
        model.append_mol(parser)
        top = app.PDBFile(ca_pdb).getTopology()
        model.create_system(top=top, box_a=box_a, box_b=box_b, box_c=box_c)
        model.add_protein_bonds(force_group=bond_force_group)
        model.add_protein_angles(force_group=angle_force_group)
        model.add_protein_dihedrals(force_group=dihedral_force_group)
        model.add_native_pairs(force_group=native_pair_force_group)
        return model

    @staticmethod
    def load_parameters(results_pkl=None):
        """
        Load MOFF2 parameters from a pickle file.

        If results_pkl is None, load the packaged default parameter file:
        openabc/forcefields/MOFF2/forcefields/parameters/MOFF2.pkl
        """
        if results_pkl is None:
            results_pkl = MOFF2Model.default_parameter_file
        if not os.path.exists(results_pkl):
            raise FileNotFoundError(
                f'MOFF2 parameter file not found: {results_pkl}. '
                'Pass results_pkl explicitly or place MOFF2.pkl in '
                'openabc/forcefields/MOFF2/forcefields/parameters/.'
            )
        with open(results_pkl, 'rb') as f:
            params = pickle.load(f)
        return params

    @staticmethod
    def _build_gauss_height_map(params):
        if 'gauss_height_map' in params:
            return np.asarray(params['gauss_height_map'])

        gauss_coeffs = np.asarray(params.get('gauss_coeffs', np.zeros(210)))
        rows, cols = np.triu_indices(20, k=0)
        gauss_height_map = np.zeros((20, 20))
        gauss_height_map[rows, cols] = gauss_coeffs
        gauss_height_map[cols, rows] = gauss_coeffs
        return gauss_height_map

    def add_moff2_forces(
        self,
        results_pkl=None,
        temperature=298.0 * unit.kelvin,
        ionic_strength=150.0 * unit.millimolar,
        dielectric_water=80.0,
        epsilon=0.2 * _kcal_to_kj,
        sigma_ah_map=None,
        res_group_mapping='default',
        contact_force_group=2,
        elec_force_group=3,
        density_force_group_start=7,
    ):
        """
        Add the learned MOFF2 nonbonded, electrostatic, and density terms.

        Parameters
        ----------
        results_pkl : str or None
            Path to a MOFF2 results.pkl/MOFF2.pkl file. If None, the packaged
            default parameters/MOFF2.pkl file is used.

        temperature : openmm.unit.Quantity or float
            Simulation temperature. Floats are interpreted as Kelvin.

        ionic_strength : openmm.unit.Quantity or float
            Ionic strength. Floats are interpreted as millimolar.

        dielectric_water : float
            Relative dielectric constant used for Debye-Huckel electrostatics.

        epsilon : float
            AH LJ energy scale in kJ/mol.

        sigma_ah_map : np.ndarray or None
            20 x 20 sigma matrix. If None, the packaged HPS sigma matrix is used.

        res_group_mapping : str
            Key into _res_group_mappings used by the density term.
        """
        params = self.load_parameters(results_pkl)

        if not hasattr(temperature, 'unit'):
            temperature = float(temperature) * unit.kelvin
        if not hasattr(ionic_strength, 'unit'):
            ionic_strength = float(ionic_strength) * unit.millimolar
        if sigma_ah_map is None:
            sigma_ah_map = HPS_sigma

        hydrophobic_scale = np.asarray(params['hydrophobic_scale'])
        spl_values = np.asarray(params['spl_values'])
        eta = params['eta']
        r0 = params['r0']
        rho_min = params['rho_min']
        rho_max = params['rho_max']
        gauss_height_map = self._build_gauss_height_map(params)
        gauss_delta_mu = params.get('gauss_delta_mu', 0.25)
        gauss_width = params.get('gauss_width', 0.10)

        self.add_contacts_ah_gau(
            epsilon,
            sigma_ah_map,
            hydrophobic_scale,
            gauss_height_map,
            gauss_delta_mu,
            gauss_width,
            force_group=contact_force_group,
        )

        ldby = (
            kB * temperature * VEP * dielectric_water /
            (2 * NA * ionic_strength * EC**2)
        )**0.5
        elec_cutoff = 5.0 * ldby
        self.add_dh_elec(
            ldby,
            dielectric_water,
            elec_cutoff,
            force_group=elec_force_group,
        )

        resnames = self.atoms['resname'].tolist()
        atom_types = np.array([_amino_acids.index(x) for x in resnames])
        group_map = _res_group_mappings[res_group_mapping]
        atom_group_names = [group_map[x] for x in resnames]
        group_names = sorted(set(group_map.values()))

        for i, group_name in enumerate(group_names):
            group_i_mask = np.array(atom_group_names) == group_name
            group_i_mask = group_i_mask.astype(int)
            density_force = density_spline_term_group_i(
                atom_types,
                group_i_mask,
                self.use_pbc,
                spl_values[:, i, :],
                eta,
                r0,
                rho_min,
                rho_max,
                force_group=density_force_group_start + i,
            )
            self.system.addForce(density_force)

        return params


# Backward compatibility: old scripts that import HPSTestModel now receive
# the same public MOFF2 model class.
HPSTestModel = MOFF2Model

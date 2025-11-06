from openabc.forcefields import MOFFMRGModel
from openabc.forcefields.functional_terms.nonbonded_terms import dh_elec_term_map
import pandas as pd
try:
    import openmm as mm
    import openmm.unit as unit
except ImportError:
    import simtk.openmm as mm
    import simtk.unit as unit

class MOFFNEATDNAModel(MOFFMRGModel):
    def __init__(self):
        """
        Initialize. 
        """
        self.atoms = None
        self.bonded_attr_names = ['protein_bonds', 'protein_angles', 'protein_dihedrals','native_pairs','dna_bonds',
                                  'dna_angles', 'dna_fan_bonds', 'dna_bp_distance_bonds','dna_bp_angles','dna_bp_dihedrals','exclusions']

    def append_mol(self, new_mol, verbose=False):
        """
        The method can append new molecules by concatenating atoms and bonded interaction information saved in dataframes. 
        
        Parameters
        ----------
        new_mol : a consistent parser object or CG model object
            The object of a new molecule including atom and bonded interaction information. 
        
        verbose : bool
            Whether to report the appended attributes. 
        
        """
        new_atoms = new_mol.atoms.copy()
        if hasattr(self, 'atoms'):
            if self.atoms is None:
                add_index = 0
                self.atoms = new_atoms
            else:
                add_index = len(self.atoms.index)
                self.atoms = pd.concat([self.atoms, new_atoms], ignore_index=True)
        else:
            add_index = 0
            self.atoms = new_atoms
        for each_attr_name in self.bonded_attr_names:
            if verbose:
                print(f'Append attribute: {each_attr_name}. ')
            if hasattr(new_mol, each_attr_name):
                if getattr(new_mol, each_attr_name) is not None:
                    new_attr = getattr(new_mol, each_attr_name).copy()
                    for each_col in ['a1', 'a2', 'a3', 'a4', 'a5', 'a6']:
                        if each_col in new_attr.columns:
                            new_attr[each_col] += add_index
                    if hasattr(self, each_attr_name):
                        if getattr(self, each_attr_name) is None:
                            setattr(self, each_attr_name, new_attr)
                        else:
                            combined_attr = pd.concat([getattr(self, each_attr_name).copy(), new_attr], 
                                                      ignore_index=True)
                            setattr(self, each_attr_name, combined_attr)
                    else:
                        setattr(self, each_attr_name, new_attr)
    def add_dna_bonds(self, force_group=5):
        """
        Add DNA bonds. 
        
        Parameters
        ----------
        force_group : int
            Force group. 
        
        """
        if hasattr(self, 'dna_bonds'):
            print('Add DNA Bonds.')
            bonds = mm.CustomBondForce('k2*(r-r0)^2 + k3*(r-r0)^3 + k4*(r-r0)^4')
            bonds.addPerBondParameter('k2')
            bonds.addPerBondParameter('k3')
            bonds.addPerBondParameter('k4')
            bonds.addPerBondParameter('r0')
            for _, row in self.dna_bonds.iterrows():
                a1 = int(row['a1'])
                a2 = int(row['a2'])
                parameters = row[['k2','k3','k4','r0']].tolist()
                bonds.addBond(a1, a2, parameters)
            bonds.setUsesPeriodicBoundaryConditions(self.use_pbc)
            bonds.setForceGroup(force_group)
            self.system.addForce(bonds)
    
    def add_dna_angles(self, force_group=6):
        """
        Add DNA angles. 
        
        Parameters
        ----------
        force_group : int
            Force group. 
        
        """
        if hasattr(self, 'dna_angles'):
            print('Add DNA Angles.')

            energy_function = 'k2*(theta-theta0)^2 + k3*(theta-theta0)^3 + k4*(theta-theta0)^4'
            angles = mm.CustomAngleForce(energy_function)
            angles.addPerAngleParameter('k2')
            angles.addPerAngleParameter('k3')
            angles.addPerAngleParameter('k4')
            angles.addPerAngleParameter('theta0')
            for _, row in self.dna_angles.iterrows():
                a1 = int(row['a1'])
                a2 = int(row['a2'])
                a3 = int(row['a3'])
                parameters = row[['k2', 'k3', 'k4', 'theta0']].tolist()
                angles.addAngle(a1,a2,a3,parameters)
            angles.setUsesPeriodicBoundaryConditions(self.use_pbc)
            angles.setForceGroup(force_group)
            self.system.addForce(angles)

    def add_dna_breakable_fan_bonds(self, force_group=8):
        if hasattr(self, 'dna_fan_bonds'):
            print('Add DNA Breakable Fan Bonds.')
            energy_function =  ""
            energy_function += "- kr*(distance(p2, p5) - r0)^2"
            energy_function += "- kt*(angle(p2, p5, p4) - theta1)^2"
            energy_function += "- kt*(angle(p5, p2, p1) - theta1)^2"
            energy_function += "- kt*(angle(p2, p5, p6) - theta2)^2"
            energy_function += "- kt*(angle(p5, p2, p3) - theta2)^2"
            energy_function += "- kp*(1. + cos(dihedral(p4, p5, p2, p1) + phi1))"
            energy_function += "- kp*(1. + cos(dihedral(p6, p5, p2, p3) + phi2))"

            energy_function = "(-uhb) * exp(" + energy_function + ")"
            hb_force = mm.CustomCompoundBondForce(6,energy_function)
            hb_force.addPerBondParameter('r0')
            hb_force.addPerBondParameter('kr')
            hb_force.addPerBondParameter('kt')
            hb_force.addPerBondParameter('kp')
            hb_force.addPerBondParameter('theta1')
            hb_force.addPerBondParameter('theta2')
            hb_force.addPerBondParameter('phi1')
            hb_force.addPerBondParameter('phi2')
            hb_force.addPerBondParameter('uhb')

            for _, row in self.dna_fan_bonds.iterrows():
                particles = row[['a1', 'a2', 'a3', 'a4', 'a5', 'a6']].tolist()
                parameters = row[['r0', 'kr', 'kt', 'kp', 'theta1', 'theta2', 'phi1', 'phi2', 'uhb']].tolist()
                hb_force.addBond(particles,parameters)
            hb_force.setUsesPeriodicBoundaryConditions(self.use_pbc)
            hb_force.setForceGroup(force_group)
            self.system.addForce(hb_force)
        else:
            print("ATTRIBUTE dna_fan_bonds NOT FOUND")
            exit()

    def add_dna_bp_bonds(self, force_group=8):
        """
        Add DNA Base-Pairing Bonds. 
        Parameters
        ----------
        force_group : int
            Force group. 
        """
        if hasattr(self, 'dna_bp_distance_bonds'):
            print('Add DNA Base-Pairing Bonds.')

            bonds = mm.CustomBondForce('k2*(r-r0)^2 + k3*(r-r0)^3 + k4*(r-r0)^4')
            bonds.addPerBondParameter('k2')
            bonds.addPerBondParameter('k3')
            bonds.addPerBondParameter('k4')
            bonds.addPerBondParameter('r0')
            for _, row in self.dna_bp_distance_bonds.iterrows():
                a1 = int(row['a1'])
                a2 = int(row['a2'])
                parameters = row[['k2','k3','k4','r0']].tolist()
                bonds.addBond(a1, a2, parameters)
            bonds.setUsesPeriodicBoundaryConditions(self.use_pbc)
            bonds.setForceGroup(force_group)
            self.system.addForce(bonds)
        else:
            print("ATTRIBUTE dna_bp_distance_bonds NOT FOUND")
            exit()

    def add_dna_bp_dihedrals(self, force_group=7):
        if hasattr(self, 'dna_bp_dihedrals'):
            print('Add DNA Base-Pairing Dihedrals.')

            diheds = mm.PeriodicTorsionForce()
            for _, row in self.dna_bp_dihedrals.iterrows():
                a1 = int(row['a1'])
                a2 = int(row['a2'])
                a3 = int(row['a3'])
                a4 = int(row['a4'])
                phase = -float(row['phi0'])
                k = float(row['kp'])
                diheds.addTorsion(a1,a2,a3,a4,1,phase,k)
            diheds.setUsesPeriodicBoundaryConditions(self.use_pbc)
            diheds.setForceGroup(force_group)
            self.system.addForce(diheds)
        else:
            print("ATTRIBUTE dna_bp_dihedrals NOT FOUND")
            exit()

    def add_dna_bp_angles(self, force_group=7):
        if hasattr(self, 'dna_bp_angles'):
            print('Add DNA Base-Pairing Angles.')
            angles = mm.HarmonicAngleForce()
            for _, row in self.dna_bp_angles.iterrows():
                p1 = int(row['a1'])
                p2 = int(row['a2'])
                p3 = int(row['a3'])
                angle = float(row['theta0'])
                k = float(row['kt']) * 2.0
                angles.addAngle(p1,p2,p3,angle,k)
            angles.setUsesPeriodicBoundaryConditions(self.use_pbc)
            angles.setForceGroup(force_group)
            self.system.addForce(angles)
        else:
            print("ATTRIBUTE dna_bp_angles NOT FOUND")
            exit()

    def add_elec(self, salt_conc=150.0*unit.millimolar, temperature=300.0*unit.kelvin,
                 cutoff=4.0*unit.nanometer, force_group=9,
                 manning_scalar=0.6**2, distance_dependent_dielectric=False):
        print(f'Add DNA electrostatic interactions with constant dielectric, no-switch, and 0.6 correction.')
        force1 = dh_elec_term_map(self, salt_conc=salt_conc, temperature=temperature,
                                  cutoff=cutoff, distance_dependent_dielectric=distance_dependent_dielectric,
                                  manning_scalar=manning_scalar, force_group=force_group)
        self.system.addForce(force1)

    def add_all_default_forces(self,salt_conc=150.0*unit.millimolar,temperature=300.0*unit.kelvin):
        """
        Add all the forces with default settings. 
        """
        print('Add all forces with default settings.')
        self.add_protein_bonds(force_group=1)
        self.add_protein_angles(force_group=2)
        self.add_protein_dihedrals(force_group=3)
        self.add_native_pairs(force_group=4)
        self.add_dna_bonds(force_group=5)
        self.add_dna_angles(force_group=6)
        self.add_dna_breakable_fan_bonds(force_group=7)
        self.add_dna_bp_bonds(force_group=8)
        self.add_dna_bp_angles(force_group=9)
        self.add_dna_bp_dihedrals(force_group=10)
        self.add_contacts(force_group=11)
        self.add_elec(salt_conc=salt_conc,temperature=temperature,force_group=12)

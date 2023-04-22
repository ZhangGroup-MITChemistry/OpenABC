import numpy as np
import pandas as pd
import simtk.openmm as mm
import simtk.openmm.app as app
import simtk.unit as unit
from openmmplumed import PlumedForce
from openabc.forcefields.rigid import createRigidBodies
import openabc.utils.helper_functions as helper_functions
import sys
import os

class CGModel(object):
    """
    The general class with general methods that can be inherited by any other CG model classes. 
    
    Each child class of CGModel has to set self.bonded_attr_names, which is used to append molecules and set rigid bodies. 
    """
    def __init__(self):
        """
        Initialize. 
        """
        self.atoms = None
        
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
                    for each_col in ['a1', 'a2', 'a3', 'a4']:
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
    
    def atoms_to_pdb(self, cg_pdb, reset_serial=True):
        """
        Save atoms to pdb as topology will be read from the pdb file.
        PDB file is required for OpenMM simulation.
        
        Parameters
        ----------
        cg_pdb : str
            Output path for CG PDB file.
        
        reset_serial : bool
            Reset serial to 1, 2, ...
        
        """
        # do not write charge due to pdb space limit
        atoms = self.atoms.copy()
        atoms.loc[:, 'charge'] = ''
        if reset_serial:
            atoms['serial'] = list(range(1, len(atoms.index) + 1))
        helper_functions.write_pdb(atoms, cg_pdb)
    
    def create_system(self, top, use_pbc=True, box_a=500, box_b=500, box_c=500, remove_cmmotion=True):
        """
        Create OpenMM system for simulation. 
        Need to further add forces to this OpenMM system. 
        
        Parameters
        ----------
        top : OpenMM Topology
            The OpenMM topology. 
        
        use_pbc : bool
            Whether to use periodic boundary condition (PBC). 
        
        box_a : float or int
            Box length along x-axis. 
        
        box_b : float or int
            Box length along y-axis. 
        
        box_c : float or int
            Box length along z-axis. 
        
        remove_cmmotion : bool
            Whether to add CMMotionRemover to remove center of mass motions. 
        
        """
        self.top = top
        self.use_pbc = use_pbc
        self.system = mm.System()
        if self.use_pbc:
            box_vec_a = np.array([box_a, 0, 0])*unit.nanometer
            box_vec_b = np.array([0, box_b, 0])*unit.nanometer
            box_vec_c = np.array([0, 0, box_c])*unit.nanometer
            self.system.setDefaultPeriodicBoxVectors(box_vec_a, box_vec_b, box_vec_c)
        mass = self.atoms['mass'].tolist()
        for each in mass:
            self.system.addParticle(each)
        if remove_cmmotion:
            force = mm.CMMotionRemover()
            self.system.addForce(force)
    
    def set_rigid_bodies(self, rigid_coord, rigid_bodies, keep_unchanged=[]):
        """
        Set rigid bodies and remove bonded interactions within the same rigid body. 
        
        Directly run `createRigidBodies` if the user does not want to change any bonded interactions or exclusions.
        
        Parameters
        ----------
        rigid_coord : Quantity, shape = (n_atoms, 3)
            Atom coordinates to build rigid body restrictions. 
        
        rigid_bodies : list 
            A list includes many sublists. Atoms in each sublist are recognized as one rigid body. 
        
        keep_unchanged : list
            A list including attribute names that user intends to keep unchanged. 
        
        """
        rigid_body_id_dict = {}
        for i in range(len(self.atoms.index)):
            rigid_body_id_dict[i] = None
        for i in range(len(rigid_bodies)):
            for j in rigid_bodies[i]:
                rigid_body_id_dict[j] = i
        for each_attr_name in self.bonded_attr_names:
            if hasattr(self, each_attr_name) and (each_attr_name not in keep_unchanged):
                each_attr = getattr(self, each_attr_name)
                if each_attr is not None:
                    new_attr = pd.DataFrame(columns=each_attr.columns)
                    for i, row in each_attr.iterrows():
                        bonded_atoms = []
                        for each_col in ['a1', 'a2', 'a3', 'a4']:
                            if each_col in each_attr.columns:
                                bonded_atoms.append(int(row[each_col]))
                        bonded_atom_rigid_body_id = [rigid_body_id_dict[x] for x in bonded_atoms]
                        flag = True
                        if bonded_atom_rigid_body_id[0] is not None:
                            if all([x==bonded_atom_rigid_body_id[0] for x in bonded_atom_rigid_body_id]):
                                flag = False
                        if flag:
                            new_attr.loc[len(new_attr.index)] = row
                    setattr(self, each_attr_name, new_attr)
        createRigidBodies(self.system, rigid_coord, rigid_bodies)
        
    def add_plumed(self, plumed_script_path):
        """
        Add OpenMM plumed. 
        
        Parameters
        ----------
        plumed_script_path : str
            Input plumed script path. 
        
        Reference
        ---------
            PLUMED website: https://www.plumed.org/
        
        """
        with open(plumed_script_path, 'r') as input_reader:
            force = PlumedForce(input_reader.read())
        self.system.addForce(force)
    
    def save_system(self, system_xml='system.xml'):
        """
        Save system to readable xml format. 
        
        Parameters
        ----------
        system_xml : str
            Output system xml file path. 
        
        """
        with open(system_xml, 'w') as output_writer:
            output_writer.write(mm.XmlSerializer.serialize(self.system))
    
    def save_state(self, state_xml='state.xml'):
        """
        Save simulation state to readable xml format. 
        
        Parameters
        ----------
        state_xml : str
            Output state xml file path. 
        
        """
        with open(state_xml, 'w') as output_writer:
            state = self.simulation.context.getState(getPositions=True, getVelocities=True, getForces=True, 
                                                     getEnergy=True, getParameters=True, 
                                                     enforcePeriodicBox=self.use_pbc)
            output_writer.write(mm.XmlSerializer.serialize(state))
    
    def set_simulation(self, integrator, platform_name='CPU', properties={'Precision': 'mixed'}, init_coord=None):
        platform = mm.Platform.getPlatformByName(platform_name)
        """
        Set OpenMM simulation.
        
        Parameters
        ----------
        integrator : OpenMM Integrator
            OpenMM integrator. 
        
        platform_name : str
            OpenMM simulation platform name. It can be one from: Reference or CPU or CUDA or OpenCL. 
        
        properties : dict
            OpenMM simulation platform properties. 
        
        init_coord : None or array-like
            Initial coordinate. 
        
        """
        print(f'Use platform: {platform_name}')
        if platform_name in ['CUDA', 'OpenCL']:
            if 'Precision' not in properties:
                properties['Precision'] = 'mixed'
            precision = properties['Precision']
            print(f'Use precision: {precision}')
            self.simulation = app.Simulation(self.top, self.system, integrator, platform, properties)
        else:
            self.simulation = app.Simulation(self.top, self.system, integrator, platform)
        if init_coord is not None:
            self.simulation.context.setPositions(init_coord)
    
    def move_COM_to_box_center(self):
        """
        Move center of mass (COM) to simulation box center. 
        """
        print('Move center of mass (COM) to box center')
        state = self.simulation.context.getState(getPositions=True, enforcePeriodicBox=self.use_pbc)
        positions = np.array(state.getPositions().value_in_unit(unit.nanometer))
        n_atoms = positions.shape[0]
        mass = []
        for i in range(n_atoms):
            mass.append(self.system.getParticleMass(i).value_in_unit(unit.dalton))
        mass = np.array(mass)
        weights = mass/np.sum(mass)
        box_vec_a, box_vec_b, box_vec_c = self.system.getDefaultPeriodicBoxVectors()
        box_vec_a = np.array(box_vec_a.value_in_unit(unit.nanometer))
        box_vec_b = np.array(box_vec_b.value_in_unit(unit.nanometer))
        box_vec_c = np.array(box_vec_c.value_in_unit(unit.nanometer))
        box_center = 0.5*(box_vec_a + box_vec_b + box_vec_c)
        center_of_mass = np.average(positions, axis=0, weights=weights)
        positions = positions - center_of_mass + box_center
        self.simulation.context.setPositions(positions*unit.nanometer)
    
    def add_reporters(self, report_interval, output_dcd='output.dcd', report_dcd=True, report_state=True):
        """
        Add reporters for OpenMM simulation. 
        
        Parameters
        ----------
        report_interval : int
            Report dcd and report state interval.
        
        output_dcd : str
            Output dcd file path. 
        
        report_dcd : bool
            Whether to output dcd file. 
        
        report_state : bool
            Whether to output simulation state. 
        
        """
        if report_dcd:
            dcd_reporter = app.DCDReporter(output_dcd, report_interval, enforcePeriodicBox=self.use_pbc)
            self.simulation.reporters.append(dcd_reporter)
        if report_state:
            state_reporter = app.StateDataReporter(sys.stdout, report_interval, step=True, time=True, 
                                                   potentialEnergy=True, kineticEnergy=True, totalEnergy=True, 
                                                   temperature=True, speed=True)
            self.simulation.reporters.append(state_reporter)
        
    
    


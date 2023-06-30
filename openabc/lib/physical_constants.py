try:
    import openmm.unit as unit
except ImportError:
    import simtk.unit as unit

# define some constants based on CODATA
NA = unit.AVOGADRO_CONSTANT_NA # Avogadro constant
kB = unit.BOLTZMANN_CONSTANT_kB  # Boltzmann constant
EC = 1.602176634e-19*unit.coulomb # elementary charge
VEP = 8.8541878128e-12*unit.farad/unit.meter # vacuum electric permittivity
GAS_CONST = 1.0*unit.BOLTZMANN_CONSTANT_kB*unit.AVOGADRO_CONSTANT_NA # gas constant


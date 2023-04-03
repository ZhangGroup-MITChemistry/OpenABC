# check-HPS

Test HPS model energy by comparing with HOOMD-Blue outputs. 

Run command `python rerun_hps_zero_offset.py` to compute energy of snapshots for DDX4 and FUS-LC with HPS model Urry scale (optimal parameter) and KR scale (no additional scaling or drift), and compare with HOOMD-Blue outputs. 

Note as HOOMD-Blue output energy does not offset (offset means shift the nonbonded potential by constant so it is continuous at cutoff), we compare an HPS model without offset version with HOOMD-Blue output. Such OpenMM zero offset version HPS model is only used for this test. 


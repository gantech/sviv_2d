spackMgrEnv
spackSVIV
spack load openfast

cd dist_edge
beamdyn_driver bd_driver_edge.inp

cd ../dist_flap
beamdyn_driver bd_driver_flap.inp

cd ../dist_twist
beamdyn_driver bd_driver_twist.inp

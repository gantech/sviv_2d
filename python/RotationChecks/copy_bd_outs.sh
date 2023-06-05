mkdir bd_outputs

# Off Axis
cp ../../OpenFAST/BeamDyn/rotation_center/off_axis/bd_dyn_off_axis.BD.sum.yaml ./bd_outputs/.
cp ../../OpenFAST/BeamDyn/rotation_center/off_axis/bd_static_off_axis.out ./bd_outputs/.

# On Axis
cp ../../OpenFAST/BeamDyn/rotation_center/on_axis/bd_dyn_on_axis.BD.sum.yaml ./bd_outputs/.
cp ../../OpenFAST/BeamDyn/rotation_center/on_axis/bd_static_on_axis.out ./bd_outputs/.


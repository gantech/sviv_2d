# Collection of utilities to generate nalu-wind input files for airfoil
# simulations
#
# Generate a set of inputs for prescribed motion simulations

import yaml, json, glob, sys, os
from pathlib import Path
import numpy as np
import pandas as pd


def gen_fsi_case(af_name, mesh_file, dir_name, freq, ampX, ampY, sim_vel, mech_model='nalu_inputs/template/chord_3dof.yaml', run_folder='nalu_runs', 
                 template="nalu_inputs/template/airfoil_osc.yaml", nominal_St=0.16, nominal_visc=1e-5, nflow_throughs=160, dtflowthrough=0.01):
    """Generate a nalu input file for simulation of flow past an airfoil
    using k-w-SST turbulence model

    Args:
        af_name (string): name of airfoil
        mesh_file (string): name of mesh
        freq (double): frequency to excite the airfoil at.
        mech_ind (int): index in the list of different mechanical 3DOF models
        mech_model (string): yaml file with the 3DOF mechanical model details to be used for the simulation.
        run_folder (string) : folder to create the runs in.
        template (string): Path to template nalu input file in yaml format
        nominal_St (double): Nominal value of the Strouhal number to use in calculating the excitation freq
        nominal_visc (double): Nominal viscosity for the real airfoil that is scaled to match Reynolds number for simulations

        nflow_throughs - number of flow throughs for the velocity over the simulated airfoil
        dtflowthrough - fraction of flow through time to use for the time step (varies with inflow velocity)

    Returns:
        None

    """

    
    ###### Check Input Files
    if ( not Path(template).exists() ):
        print("Template file ", template, " doesn't exist. Please check your inputs")
        sys.exit()

    tfile = yaml.load(open(template),Loader=yaml.UnsafeLoader)

    if ( not Path(mesh_file).exists() ):
        print("Mesh file ", mesh_file, " doesn't exist. Please check your inputs")
        sys.exit()


    if ( not Path(mech_model).exists() ):
        print("Mechanical/Structural model file ", mech_model, " doesn't exist. Please check your inputs")
        sys.exit()

    mechfile = yaml.load(open(mech_model),Loader=yaml.UnsafeLoader)

    ###### Create folder and modify template

    Path(run_folder+'/{}/{}/amp_{:.3f}'.format(af_name, dir_name, np.round(np.sqrt(ampX**2+ampY**2), 3))).mkdir(
        parents=True, exist_ok=True)

    # Copy the mechanical model to the structure level folder for future reference
    mech_dest = str(Path(run_folder \
                + '/{}/{}/amp_{:.3f}'.format(af_name, dir_name, np.round(np.sqrt(ampX**2+ampY**2), 3)) \
                + '/.'))

    os.system('cp {} {}'.format(mech_model, mech_dest))

    ### Copy mechanical model details from YAML

    # Chord sizes - simulated and nominal. Setting to be same.
    chord_sim = mechfile['chord_length'] # simulating true size now.
    chord_nominal = mechfile['chord_length']
    

    # Angle of Attack
    aoa = float(mechfile['angle'])

    # # mass, stiffness, damping, force transform matrices
    # tfile['realms'][0]['mesh_motion'][0]['motion'][0]['mass_matrix'] = \
    #              mechfile['mass_matrix']

    # tfile['realms'][0]['mesh_motion'][0]['motion'][0]['stiffness_matrix'] = \
    #              mechfile['stiffness_matrix']

    # tfile['realms'][0]['mesh_motion'][0]['motion'][0]['damping_matrix'] = \
    #              mechfile['damping_matrix']

    # tfile['realms'][0]['mesh_motion'][0]['motion'][0]['force_transform_matrix'] = \
    #              mechfile['force_transform_matrix']

    # tfile['realms'][0]['mesh_motion'][0]['motion'][0]['loads_scale'] = \
    #              mechfile['loads_scale']

    ### Set Strouhal and Reynolds Numbers
    # nominal = desired simulation parameters
    # sim = simulation parameters - simulated chord size as real.

    # ## calculate velocity based on Strouhal
    # nominal_vel = freq * chord_nominal * np.sin(aoa*np.pi/180) / nominal_St

    # sim_vel = float(freq * chord_sim * np.sin(aoa*np.pi/180) / nominal_St)

    # if( tfile['realms'][0]['material_properties']['specifications'][0]['name'] == 'density' ):
    #     nominal_density = float(tfile['realms'][0]['material_properties']['specifications'][0]['value'])
    # else:
    #     print("Property density is not in the expected place.")
    #     sys.exit()

    # if not ( tfile['realms'][0]['material_properties']['specifications'][1]['name'] == 'viscosity'):
    #     print("Property viscosity is not in the expected place.")
    #     sys.exit()

    # ## calculate viscosity

    # nominal_Re = nominal_density * nominal_vel * chord_nominal / nominal_visc

    # sim_visc = float(nominal_density * sim_vel * chord_sim / nominal_Re)


    print('Commenting out velocity setting')


    # # set velocity
    # tfile['realms'][0]['initial_conditions'][0]['value']['velocity'] = [float(sim_vel), 0.0]
    # tfile['realms'][0]['boundary_conditions'][1]['inflow_user_data']['velocity'] = [float(sim_vel), 0.0]

    # # # set viscosity
    # # tfile['realms'][0]['material_properties']['specifications'][1]['value'] = sim_visc

    # ### Set time step information based on velocity.

    # tfile['Time_Integrators'][0]['StandardTimeIntegrator']['termination_step_count'] = int(nflow_throughs/dtflowthrough)

    # tfile['Time_Integrators'][0]['StandardTimeIntegrator']['time_step'] = float(chord_sim/sim_vel * dtflowthrough)

    print('End of velocity comment out')

    ### Delayed Start for loads with ramp

    # load_ramp_flows_start = 10 # start ramp up after 10 flow through times
    # load_ramp_flows_end = 11 #End ramp up of loads after 11 flow through times

    # tfile['realms'][0]['mesh_motion'][0]['motion'][0]['load_transition_start'] = float(load_ramp_flows_start*chord_sim/sim_vel)
    # tfile['realms'][0]['mesh_motion'][0]['motion'][0]['load_transition_end'] = float(load_ramp_flows_end*chord_sim/sim_vel)

    # Prescribe Frequency and Amplitude 
    tfile['realms'][0]['mesh_motion'][0]['motion'][0]['freq'] = [float(freq), float(freq), float(freq)]
    tfile['realms'][0]['mesh_motion'][0]['motion'][0]['amp'] =  [float(ampX), float(ampY), 0.0]

    ### More general settings
    tfile['realms'][0]['mesh'] = str(Path(mesh_file).absolute())
    tfile['realms'][0]['output']['output_data_base_name'] = 'results/{}.e'.format(af_name)

    # # Previous step from old generation script, not sure if needed.
    # tfile['linear_solvers'][1]['muelu_xml_file_name'] = str(
    #     Path('nalu_inputs/template/milestone_aspect_ratio_gs.xml').absolute() )

    ### Output file

    yaml.dump(tfile, 
              open(run_folder \
                + '/{}/{}/amp_{:.3f}/{}_amp{:.3f}.yaml'.format(af_name, dir_name, np.sqrt(ampX**2+ampY**2), af_name, np.sqrt(ampX**2+ampY**2)), \
                'w'), 
              default_flow_style=False)

def gen_prescribed(amp, vel=15, freq=1.0, mech_model='nalu_inputs/template/chord_3dof.yaml', run_folder='prescribe_runs', template="nalu_inputs/template/prescribed_motion.yaml"):
    """Generate FSI cases for the FFAW3211 airfoil

    Args:
        freq (list): List of frequencies for the nominal conditions to test
        mech_model (list) : List of yaml files to produce the mechanical model based on. These include AOA and chord length for scaling. Also mode shape info.
        run_folder : folder to create the runs in
        template : template file for the nalu-wind input deck

    Return:
       None

    """

    mesh_name = '/projects/sviv/ganeshv/sviv_2d/cases/af_3d/ffaw3211_3d_scaled_aoa50_2.exo' # thin mesh to check for convergence and such

    aoa = 50

    # list of x and y components of the amplitude
    xComp = [ np.cos(aoa*180/np.pi), np.sin(aoa*180/np.pi)]
    yComp = [-np.sin(aoa*180/np.pi), np.cos(aoa*180/np.pi)]
    dir_name = ['Edge', 'Flap']

    for dir_ind in range(len(xComp)):
        for amp_curr in amp:
            gen_fsi_case('ffaw3211', mesh_name, dir_name[dir_ind], freq, amp_curr*xComp[dir_ind], amp_curr*yComp[dir_ind], vel, mech_model, run_folder, template)


if __name__=="__main__":

    # Selection of amplitudes to consider, separately in edge and flap
    amp = [0.1, 0.2, 0.4, 0.6, 0.9, 1.3, 1.8, 3.0, 4.5, 6.0, 7.0]
    gen_prescribed(amp, freq=0.47)


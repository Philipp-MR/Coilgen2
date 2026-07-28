
%%%% Autor: Philipp Amrein
%%%% Dezember 2025



clc; clear all; 

if ispc
cd('..\');
else
cd('../');
end


%% Run the algorithm


    coil_layouts.out=CoilGen(...
    'field_shape_function','x',... % definition of the target field
    'coil_mesh_file','split_bore_flange_inner_and_ring_v_8_1_2026.stl', ...    
    'target_mesh_file','none', ... 
    'target_region_radius',0.15,...  % in meter
    'use_only_target_mesh_verts',false, ...
    'sf_source_file','none', ...
    'levels',22, ... % the number of potential steps that determines the later number of windings (Stream function discretization)
    'pot_offset_factor',0.5, ... % a potential offset value for the minimal and maximal contour potential ; must be between 0 and 1
    'is_flange_cylinder_flag',true, ...
    'interconnection_cut_width',0.05, ... % the width for the interconnections are interconnected; in meter
    'normal_shift_length',0.01, ... % the length for which overlapping return paths will be shifted along the surface normals; in meter
    'iteration_num_mesh_refinement',0, ... % the number of refinements for the mesh;
    'set_roi_into_mesh_center',true, ...
    'skip_normal_shift',false,...
    'level_set_method','primary',... %Specify one of the three ways the level sets are calculated: "primary","combined", or "independent"
    'interconnection_method','regular',...
    'skip_postprocessing',false,...
    'skip_normal_shift',true,...
    'skip_inductance_calculation',false,...
    'conductor_thickness',0.01,...
    'smooth_flag',false,...
    'tikonov_reg_factor',10); %Tikonov regularization factor for the SF optimization




%% Plot results
close all;

coil_name='Coil';

if ispc
addpath(strcat(pwd,'\','plotting'));
else
addpath(strcat(pwd,'/','plotting'));
end
%Chose a even leveled solution for plotting
%solutions_to_plot=find(arrayfun(@(x) ~isempty(coil_layouts(x).out),1:numel(coil_layouts)));
single_ind_to_plot= find_even_leveled_solution(coil_layouts);
plot_error_different_solutions(coil_layouts,single_ind_to_plot,coil_name);
plot_2D_contours_with_sf(coil_layouts,single_ind_to_plot,coil_name);
plot_3D_contours_with_sf(coil_layouts,single_ind_to_plot,coil_name);
plot_groups_and_interconnections(coil_layouts,single_ind_to_plot,coil_name);
plot_coil_parameters(coil_layouts,coil_name);
plot_coil_track_with_resulting_bfield(coil_layouts,single_ind_to_plot,coil_name);
plot_various_error_metrics(coil_layouts,single_ind_to_plot,coil_name);
plot_resulting_gradient(coil_layouts,single_ind_to_plot,coil_name);
rmpath('plotting');


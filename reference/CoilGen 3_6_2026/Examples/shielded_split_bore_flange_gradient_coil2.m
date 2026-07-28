
%%%% Autor: Philipp Amrein, University Freiburg, Medical Center, Radiology,
%%%% Medical Physics
%%%% February 2022

%This scripts generates a shielded y gradient coil


clc; 
clear all; 

if ispc
cd('..\');
else
cd('../');
end



%% Run the algorithm

%tikonov = [0.1:0.2:1];
% tikonov = [0.01,0.1,0.2];
% r_inner = [0.35 0.40 0.45];
% gap     = [0.25 0.30 0.35];
r_inner = [0.2];
r_outer = [1.5];
gap     = [1.1];
%tikonov = [0.1,2:2:10];
%tikonov = [1:10];
tikonov = 10;

primary_shield_seperation = 0.1;

[T,R,G] = ndgrid(tikonov, r_inner, gap);
N = numel(T);

runs = table(T(:), R(:), G(:), repmat({[]},N,1), ...
    'VariableNames', {'tikonov','r_inner','gap','out'});

for i = 1:N
%     try
        runs.out{i} = CoilGen( ...
            'field_shape_function','y', ...
            'coil_mesh_file','create shielded flanged split bore mesh', ...
            'shielded_flanged_split_bore_mesh_parameter_list',[ ...
                0.08, ... % edge_len [m]
                R(i), ... % r_inner [m]  (swept)
                r_outer, ... % r_outer [m]
                G(i), ... % gap [m]      (swept)
                0.01, ... % half_length [m]
                0.08,  ... % shield_edge_len [m]
                r_inner+primary_shield_seperation, ... % shield_r_inner [m]
                r_outer,  ... % shield_r_outer [m]
                gap+primary_shield_seperation, ... % shield_gap [m]
                NaN   ... % shield_half_length [m] 
                ], ...
            'target_mesh_file','none', ...
            'create_split_bore_field_suppression_surface','true', ...
            'split_bore_area_suppression_parameters',[0.08,r_inner+0.3,gap+0.5,0.01], ... % [triangle_len, r_inner, gap, half_length]
            'secondary_target_weight',0.8, ...
            'target_region_radius',0.2, ...
            'use_only_target_mesh_verts',false, ...
            'sf_source_file','none', ...
            'levels',24, ...
            'pot_offset_factor',0.25, ...
            'is_flange_cylinder_flag',true, ...
            'interconnection_cut_width',0.05, ...
            'normal_shift_length',0.01, ...
            'iteration_num_mesh_refinement',0, ...
            'set_roi_into_mesh_center',true, ...
            'force_cut_selection',{'high'}, ...
            'level_set_method','combined', ...
            'interconnection_method','regular', ...
            'skip_postprocessing',true, ...
            'skip_inductance_calculation',true, ...
            'tikonov_reg_factor', runs.tikonov(i));
%     catch
%         runs.out{i} = [];
%     end
end


%% Select solutions

mask = runs.r_inner==r_inner & runs.gap==gap;

stdg = cellfun(@(o) get_nested(struct('out',o), "out.layout_gradient.std_gradient_in_target_direction", NaN), runs.out(mask));
mean = cellfun(@(o) get_nested(struct('out',o), "out.layout_gradient.mean_gradient_in_target_direction", NaN), runs.out(mask));
err  = cellfun(@(o) get_nested(struct('out',o), "out.error_vals.mean_rel_error_layout_vs_target", NaN), runs.out(mask));
len  = cellfun(@(o) get_nested(struct('out',o), "out.coil_parts.combined_loop_length", NaN), runs.out(mask));
tik  = runs.tikonov(mask);

%% Plot results
close all;

coil_name='Coil';

coil_layouts = struct( ...
    'out', runs.out, ...
    'tikonov_reg_factor', num2cell(runs.tikonov), ...
    'r_inner', num2cell(runs.r_inner), ...
    'gap', num2cell(runs.gap) );

if ispc
addpath(strcat(pwd,'\','plotting'));
else
addpath(strcat(pwd,'/','plotting'));
end
%Chose a even leveled solution for plotting
%solutions_to_plot=find(arrayfun(@(x) ~isempty(coil_layouts(x).out),1:numel(coil_layouts)));
single_ind_to_plot= 1;
%plot_error_different_solutions(coil_layouts,solutions_to_plot,coil_name);
plot_2D_contours_with_sf(coil_layouts,single_ind_to_plot,coil_name);
plot_3D_contours_with_sf(coil_layouts,single_ind_to_plot,coil_name);
plot_groups_and_interconnections(coil_layouts,single_ind_to_plot,coil_name);
plot_coil_parameters(coil_layouts,coil_name);
plot_coil_track_with_resulting_bfield(coil_layouts,single_ind_to_plot,coil_name);
plot_various_error_metrics(coil_layouts,single_ind_to_plot,coil_name);
plot_resulting_gradient(coil_layouts,single_ind_to_plot,coil_name);
plot_3D_current_density(coil_layouts,single_ind_to_plot,coil_name);
rmpath('plotting');


%%


function v = get_nested(s,p,d)
if nargin<3, d=NaN; end, v=d;
try, for f=strsplit(char(p),'.'), s=s.(f{1}); end, v=s; end
end


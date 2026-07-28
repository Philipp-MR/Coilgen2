
%%%% Autor: Philipp Amrein, University Freiburg, Medical Center, Radiology,
%%%% Medical Physics
%%%% February 2022

%This scripts generates a shielded y gradient coil


clc; 
%clear all; 

if ispc
cd('..\');
else
cd('../');
end



%% Run the algorithm

tikonov = [0.01,0.1,0.2:0.1:1,10,100];

tikonov = tikonov(7)

% r_inner = [0.35 0.40 0.45];
% gap     = [0.25 0.30 0.35];
r_inner = [0.40];
gap     = [0.30];


[T,R,G] = ndgrid(tikonov,r_inner,gap);
sz = size(T);

coil_layouts = repmat(struct('out',[],'tikonov_reg_factor',[],'r_inner',[],'gap',[]), sz);

for i = 1:numel(T)
    [a,b,c] = ind2sub(sz,i);
    coil_layouts(a,b,c).tikonov_reg_factor = T(i);
    coil_layouts(a,b,c).r_inner = R(i);
    coil_layouts(a,b,c).gap     = G(i);
    try
        coil_layouts(a,b,c).out = CoilGen( ...
            'field_shape_function','y', ...
            'coil_mesh_file','create shielded flanged split bore mesh', ...
            'shielded_flanged_split_bore_mesh_parameter_list',[ ...
                0.05, ... % edge_len [m]
                R(i), ... % r_inner [m]  (swept)
                0.60, ... % r_outer [m]
                G(i), ... % gap [m]      (swept)
                0.60, ... % half_length [m]
                NaN,  ... % shield_edge_len [m]
                0.45, ... % shield_r_inner [m]
                0.6,  ... % shield_r_outer [m]
                0.31, ... % shield_gap [m]
                NaN   ... % shield_half_length [m]
            ], ...
            'target_mesh_file','none', ...
            'secondary_target_mesh_file','Open_cylinder_r750mm_length_1500mm.stl', ...
            'secondary_target_weight',0.5, ...
            'target_region_radius',0.15, ...
            'use_only_target_mesh_verts',false, ...
            'sf_source_file','none', ...
            'levels',26, ...
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
            'tikonov_reg_factor', T(i) );
    catch
        coil_layouts(a,b,c).out = [];
    end
end
%%

std_gradient_in_target_direction = arrayfun(@(x) get_nested(x,"out.layout_gradient.std_gradient_in_target_direction",NaN), coil_layouts);
mean_error = arrayfun(@(x) get_nested(x,"out.error_vals.mean_rel_error_layout_vs_target",NaN), coil_layouts);
combined_loop_length = arrayfun(@(x) get_nested(x,"out.coil_parts.combined_loop_length",NaN), coil_layouts);



%% Plot results
close all;

coil_name='Coil';

coil_layouts = coil_layouts(:);

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


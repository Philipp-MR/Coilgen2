
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

tikonov = [5,10,20];
% tikonov = [0.01,0.1,0.2];
% r_inner = [0.35 0.40 0.45];
% gap     = [0.25 0.30 0.35];
r_inner = [0.30];
r_outer = [1];
%gap     = [0.90];
gap     = [0.80:0.02:1.2];
tikonov = 10;
edge_len = 0.1;
primary_shield_seperation = 0.1;

[T,R,G] = ndgrid(tikonov, r_inner, gap);
N = numel(T);

runs = table(T(:), R(:), G(:), repmat({[]},N,1), ...
    'VariableNames', {'tikonov','r_inner','gap','out'});

for i = 1:N
%     try
        runs.out{i} = CoilGen( ...
            'field_shape_function','y', ...
            'coil_mesh_file','create shielded bi-planary mesh', ...
            'shielded_circular_biplanar_mesh_parameter_list',[ ...
                r_outer,   ... % (1) primary_outer_radius
                edge_len,   ... % (2) primary_edge_len
                r_outer,   ... % (3) shield_outer_radius
                edge_len,   ... % (4) shield_edge_len
                0,0,1, ... % (5-7) target_normal_x,y,z  (z-axis)
                0,0,0, ... % (8-10) center_x,y,z        (origin)
                runs.gap(i),    ... % (11) plane_distance (primary separation)
                primary_shield_seperation    ... % (12) shield_gap (offset primary->shield, each side)
            ], ...
            'target_mesh_file','none', ...
            'create_circular_biplanar_field_suppression_surface','true', ...
            'circular_biplanar_area_suppression_parameters',[ ...
                edge_len,  ... % edge_len
                r_outer,  ... % outer_radius
                runs.gap(i)+5*primary_shield_seperation   ... % plane_distance (disc spacing)
            ], ...
            'secondary_target_weight',0.8, ...
            'target_region_radius',0.15, ...
            'use_only_target_mesh_verts',false, ...
            'sf_source_file','none', ...
            'levels',25, ...
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

mask = runs.r_inner == r_inner;   % N×1 logical
outs = runs.out(mask);

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

mask = (runs.r_inner == r_inner);

g = runs.gap(mask);
err  =      cellfun(@(o) get_nested(o,'error_vals.mean_rel_error_layout_vs_target',NaN), runs.out(mask));          % %
meang =     cellfun(@(o) get_nested(o,'layout_gradient.mean_gradient_in_target_direction',NaN), runs.out(mask)); % mT/m/A
len   =     cellfun(@(o) get_nested(o,'coil_parts.combined_loop_length',NaN), runs.out(mask));                  % m

[g,ix] = sort(g);  err = err(ix);  meang = meang(ix);  len = len(ix);
v = isfinite(g) & isfinite(err) & isfinite(meang) & isfinite(len);
g = g(v); err = err(v); meang = meang(v); len = len(v);

figure('Color','w');
tiledlayout(1,3,'TileSpacing','compact','Padding','compact');

nexttile;
plot(g,meang,'-o','LineWidth',1.5,'MarkerSize',5);
grid on; box on; 
%ylim([0 max(meang)*1.05]);
xlabel('Gap(m)'); ylabel('Mean gradient (mT/m/A)'); title('Gradient(mT/m/A)');

nexttile;
plot(g,err,'-o','LineWidth',1.5,'MarkerSize',5);
grid on; box on; 
%ylim([0 max(err)*1.05]);
xlabel('Gap(m)'); ylabel('Mean rel. error (%)'); title('Error(%)');

nexttile;
plot(g,len,'-o','LineWidth',1.5,'MarkerSize',5);
grid on; box on; 
%ylim([0 max(len)*1.05]);
xlabel('Gap(m)'); ylabel('Wire length (m)'); title('Wire Length(m)');


%%


function v = get_nested(s,p,d)
if nargin<3, d=NaN; end, v=d;
try, for f=strsplit(char(p),'.'), s=s.(f{1}); end, v=s; end
end
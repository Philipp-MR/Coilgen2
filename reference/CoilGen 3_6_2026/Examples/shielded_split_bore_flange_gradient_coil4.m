
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
r_inner = [0.40:0.02:0.6];
r_outer = [1.2];
gap     = [0.40];
%tikonov = [0.1,2:2:10];
%tikonov = [20:5:50];
tikonov = 30;


primary_shield_seperation = 0.1;

[T,R,G] = ndgrid(tikonov, r_inner, gap);
N = numel(T);

runs = table(T(:), R(:), G(:), repmat({[]},N,1), ...
    'VariableNames', {'tikonov','r_inner','gap','out'});

for i = 1:N
    ri = runs.r_inner(i);
    ga = runs.gap(i);

    runs.out{i} = CoilGen( ...
        'field_shape_function','y', ...
        'coil_mesh_file','create shielded flanged split bore mesh', ...
        'shielded_flanged_split_bore_mesh_parameter_list',[ ...
            0.1, ... % edge_len [m]
            ri,  ... % r_inner [m]  (swept)
            r_outer, ... % r_outer [m]
            ga, ... % gap [m]      (fixed, but use ga for correctness)
            0.60, ... % half_length [m]
            0.1,  ... % shield_edge_len [m]
            ri + primary_shield_seperation, ... % shield_r_inner [m]
            r_outer,  ... % shield_r_outer [m]
            ga + primary_shield_seperation, ... % shield_gap [m]
            NaN   ... % shield_half_length [m]
        ], ...
        'target_mesh_file','none', ...
        'create_split_bore_field_suppression_surface','true', ...
        'split_bore_area_suppression_parameters',[0.1, ri+0.3, ga+0.3, 0.6], ... % [triangle_len, r_inner, gap, half_length]
        'secondary_target_weight',0.8, ...
        'target_region_radius',0.2, ...
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
        'tikonov_reg_factor', runs.tikonov(i));
end

%% Select solutions (plot vs r_inner)

mask = ismembertol(runs.gap, gap, 1e-12) & ismembertol(runs.tikonov, tikonov, 1e-12);
x    = runs.r_inner(mask);
outs = runs.out(mask);

err_raw   =     cellfun(@(o) get_nested(o,'error_vals.mean_rel_error_layout_vs_target',NaN), outs);          % %
meang_raw =     cellfun(@(o) get_nested(o,'layout_gradient.mean_gradient_in_target_direction',NaN), outs);   % mT/m/A
len_raw   =     cellfun(@(o) get_nested(o,'coil_parts.combined_loop_length',NaN), outs);                     % m

[xs,ix] = sort(x);
err   = err_raw(ix);
meang = meang_raw(ix);
len   = len_raw(ix);

v = isfinite(xs) & isfinite(err) & isfinite(meang) & isfinite(len);
xs = xs(v); err = err(v); meang = meang(v); len = len(v);

figure('Color','w');
tiledlayout(1,3,'TileSpacing','compact','Padding','compact');

nexttile;
plot(xs,err,'-o','LineWidth',1.5,'MarkerSize',5);
grid on; box on; ylim([0 max(err)*1.05]);
xlabel('r_{inner} (m)'); ylabel('Mean rel. error (%)'); title('Error');

nexttile;
plot(xs,meang,'-o','LineWidth',1.5,'MarkerSize',5);
grid on; box on; ylim([0 max(meang)*1.05]);
xlabel('r_{inner} (m)'); ylabel('Mean gradient (mT/m/A)'); title('Gradient');

nexttile;
plot(xs,len,'-o','LineWidth',1.5,'MarkerSize',5);
grid on; box on; ylim([0 max(len)*1.05]);
xlabel('r_{inner} (m)'); ylabel('Wire length (m)'); title('Length');

%% Plot results with provided plotting utilities

coil_name = 'Coil';

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

% Choose the best (minimum error) among the swept set for 3D/contour plotting
idx_mask = find(mask);
tmp = err_raw; tmp(~isfinite(tmp)) = inf;
[~,k] = min(tmp);
single_ind_to_plot = idx_mask(k);

plot_2D_contours_with_sf(coil_layouts,single_ind_to_plot,coil_name);
plot_3D_contours_with_sf(coil_layouts,single_ind_to_plot,coil_name);
plot_groups_and_interconnections(coil_layouts,single_ind_to_plot,coil_name);
plot_coil_parameters(coil_layouts,coil_name);
plot_coil_track_with_resulting_bfield(coil_layouts,single_ind_to_plot,coil_name);
plot_various_error_metrics(coil_layouts,single_ind_to_plot,coil_name);
plot_resulting_gradient(coil_layouts,single_ind_to_plot,coil_name);
plot_3D_current_density(coil_layouts,single_ind_to_plot,coil_name);

rmpath('plotting');

%% Local function (must be last in the file!)
function v = get_nested(s,p,d)
    if nargin<3, d=NaN; end
    v = d;
    try
        parts = strsplit(char(p),'.');
        for k = 1:numel(parts)
            s = s.(parts{k});
        end
        v = s;
    catch
        v = d;
    end
end

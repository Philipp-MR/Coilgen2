
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




primary_shield_seperation = 0.05;



tikonov = [0.1:0.2:1,10,100];
%tikonov = [0.01,0.1,0.2];
% r_inner = [0.35 0.40 0.45];
gap     = [0.8:0.02:1];
%gap     = [0.30];


shield_separation = 0.05;
flange_disc_width = 0.6;

r_inner = [0.40];
r_outer = r_inner + flange_disc_width;


r_inner_shield = r_inner + shield_separation;
r_outer_shield = r_inner_shield + flange_disc_width;



[T,R,G] = ndgrid(tikonov, r_inner, gap);
N = numel(T);

runs = table(T(:), R(:), G(:), ...
    nan(N,1), nan(N,1), nan(N,1), nan(N,1), ...
    strings(N,1), repmat({[]},N,1), ...
    'VariableNames', {'tikonov','r_inner','gap','stdg','meanGrad','err','len','status','out'});


for i = 1:N
    fprintf('Run %d/%d | tik=%.3g r=%.3f gap=%.3f ... ', ...
        i, N, runs.tikonov(i), runs.r_inner(i), runs.gap(i));
    try
        o = CoilGen( ...
            'field_shape_function','y', ...
            'coil_mesh_file','create shielded flanged split bore mesh', ...
            'shielded_flanged_split_bore_mesh_parameter_list',[ ...
                0.05, ... % edge_len [m]
                R(i), ... % r_inner [m]  (swept)
                r_outer, ... % r_outer [m]
                G(i), ... % gap [m]      (swept)
                0.60, ... % half_length [m]
                NaN,  ... % shield_edge_len [m]
                r_inner_shield, ... % shield_r_inner [m]
                r_outer_shield,  ... % shield_r_outer [m]
                G(i), ... % shield_gap [m]
                NaN   ... % shield_half_length [m] 
                ], ...
            'target_mesh_file','none', ...
            'create_split_bore_field_suppression_surface','true', ...
            'split_bore_area_suppression_parameters',[0.1,r_inner+0.3,gap+0.3,0.6], ... % [triangle_len, r_inner, gap, half_length]
            'secondary_target_weight',0.8, ...
            'target_region_radius',0.15, ...
            'use_only_target_mesh_verts',false, ...
            'sf_source_file','none', ...
            'levels',32, ...
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

        %runs.out{i} = o;
        runs.stdg(i)     = get_nested(struct('out',o), "out.layout_gradient.std_gradient_in_target_direction", NaN);
        runs.meanGrad(i) = get_nested(struct('out',o), "out.layout_gradient.mean_gradient_in_target_direction", NaN);
        runs.err(i)      = get_nested(struct('out',o), "out.error_vals.mean_rel_error_layout_vs_target", NaN);
        runs.len(i)      = get_nested(struct('out',o), "out.coil_parts.combined_loop_length", NaN);

        runs.status(i) = "ok";

        fprintf('ok | err=%g meanG=%g stdG=%g len=%g\n', ...
            runs.err(i), runs.meanGrad(i), runs.stdg(i), runs.len(i));


    catch ME
        runs.out{i} = [];
        runs.status(i) = "fail: " + string(ME.identifier);

        fprintf(2, 'FAILED | %s\n', ME.message);  % prints to stderr (red in many consoles)
    end
end

fname = ['parameter_study_' char(datetime('now','Format','yyyyMMdd_HHmm')) '.mat'];
save(fname, 'runs', '-v7.3')



%% Select solutions (best tikonov per gap = lowest error)

% If you only want a specific r_inner (you already set r_inner=[0.40], but keep this for safety)
mask = runs.r_inner == 0.40;

% Keep only successful runs with finite error
tmp = runs(mask & runs.status=="ok" & isfinite(runs.err), :);

% Sort so that within each gap the lowest error comes first
tmp = sortrows(tmp, {'gap','err'});

% Take the first row for each gap (i.e., best tikonov for that gap)
[~, ia] = unique(tmp.gap, 'stable');
bestRuns = tmp(ia, :);

% Optional: sort by gap for nicer output/plotting
bestRuns = sortrows(bestRuns, 'gap');

% Convenience vectors (all aligned by gap)
gap_best  = bestRuns.gap;
tik_best  = bestRuns.tikonov;
err_best  = bestRuns.err;
mean_best = bestRuns.meanGrad;
stdg_best = bestRuns.stdg;
len_best  = bestRuns.len;

% Show summary table in command window
disp(bestRuns(:, {'gap','tikonov','err','meanGrad','stdg','len'}));

% Plot mean gradient as function of gap (using best tikonov per gap)
figure;
plot(gap_best, mean_best, 'o-');
xlabel('gap (m)');
ylabel('mean gradient in target direction');
grid on;

% (Optional) also plot error vs gap
figure;
plot(gap_best, err_best, 'o-');
xlabel('gap (m)');
ylabel('mean relative error (layout vs target)');
grid on;







%% Select solutions

mask = runs.r_inner==r_inner & runs.gap==0.9;

stdg = cellfun(@(o) get_nested(struct('out',o), "out.layout_gradient.std_gradient_in_target_direction", NaN), runs.out(mask));
mean = cellfun(@(o) get_nested(struct('out',o), "out.layout_gradient.mean_gradient_in_target_direction", NaN), runs.out(mask));
err  = cellfun(@(o) get_nested(struct('out',o), "out.error_vals.mean_rel_error_layout_vs_target", NaN), runs.out(mask));
len  = cellfun(@(o) get_nested(struct('out',o), "out.coil_parts.combined_loop_length", NaN), runs.out(mask));
tik  = runs.tikonov(mask);

% % % %% Plot results
% % % close all;
% % % 
% % % coil_name='Coil';
% % % 
% % % coil_layouts = struct( ...
% % %     'out', runs.out, ...
% % %     'tikonov_reg_factor', num2cell(runs.tikonov), ...
% % %     'r_inner', num2cell(runs.r_inner), ...
% % %     'gap', num2cell(runs.gap) );
% % % 
% % % if ispc
% % % addpath(strcat(pwd,'\','plotting'));
% % % else
% % % addpath(strcat(pwd,'/','plotting'));
% % % end
% % % %Chose a even leveled solution for plotting
% % % %solutions_to_plot=find(arrayfun(@(x) ~isempty(coil_layouts(x).out),1:numel(coil_layouts)));
% % % single_ind_to_plot= 5;
% % % %plot_error_different_solutions(coil_layouts,solutions_to_plot,coil_name);
% % % plot_2D_contours_with_sf(coil_layouts,single_ind_to_plot,coil_name);
% % % plot_3D_contours_with_sf(coil_layouts,single_ind_to_plot,coil_name);
% % % plot_groups_and_interconnections(coil_layouts,single_ind_to_plot,coil_name);
% % % plot_coil_parameters(coil_layouts,coil_name);
% % % plot_coil_track_with_resulting_bfield(coil_layouts,single_ind_to_plot,coil_name);
% % % plot_various_error_metrics(coil_layouts,single_ind_to_plot,coil_name);
% % % plot_resulting_gradient(coil_layouts,single_ind_to_plot,coil_name);
% % % plot_3D_current_density(coil_layouts,single_ind_to_plot,coil_name);
% % % rmpath('plotting');


%%


function v = get_nested(s,p,d)
if nargin<3, d=NaN; end, v=d;
try, for f=strsplit(char(p),'.'), s=s.(f{1}); end, v=s; end
end


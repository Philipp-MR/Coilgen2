function coil_parts=parameterize_mesh(coil_parts,input)
%create the paramterized 2D mesh
%
%The non-cylidnrical parameterization is taken from "matlabmesh @ Ryan
%Schmidt  rms@dgp.toronto.edu" based on desbrun et al (2002),
%"Intrinsic Parameterizations of Surface Meshes",

surface_is_cylinder = input.surface_is_cylinder_flag;
circular_factor     = input.circular_diameter_factor_cylinder_parameterization;

% NEW: flange-half mode (matches your Python half mesh)
is_flange_cylinder = false;
if isfield(input,'is_flange_cylinder')
    is_flange_cylinder = logical(input.is_flange_cylinder);
elseif isfield(input,'is_flange_cylinder_flag')
    is_flange_cylinder = logical(input.is_flange_cylinder_flag);
end

% flange tuning
diskScale = 1.0;  % disc radial distance -> axial extension distance
if isfield(input,'flange_disk_scale')
    diskScale = input.flange_disk_scale;
end

for part_ind=1:numel(coil_parts)

    tri0 = triangulation(coil_parts(part_ind).coil_mesh.faces',coil_parts(part_ind).coil_mesh.vertices');
    face_normals=faceNormal(tri0);
    max_face_normal_std=max([std(face_normals(:,1)) std(face_normals(:,2)) std(face_normals(:,3))]);

    coil_parts(part_ind).coil_mesh.v=coil_parts(part_ind).coil_mesh.vertices';
    coil_parts(part_ind).coil_mesh.fn=faceNormal(triangulation(coil_parts(part_ind).coil_mesh.faces',coil_parts(part_ind).coil_mesh.v));

    if ~(max_face_normal_std<1e-6)

        if is_flange_cylinder
            coil_parts(part_ind).coil_mesh = flange_half_disc_as_cyl_extension_polar( ...
                coil_parts(part_ind).coil_mesh, circular_factor, diskScale);

        else
            if ~surface_is_cylinder
                coil_parts(part_ind).coil_mesh = mesh_parameterization_iterative( coil_parts(part_ind).coil_mesh );
            else
                % --- your original cylinder planarization (unchanged) ---
                boundary_edges=freeBoundary(triangulation(coil_parts(part_ind).coil_mesh.faces',coil_parts(part_ind).coil_mesh.vertices'));
                boundary_loop_nodes = boundary_loops_from_edges(boundary_edges, size(coil_parts(part_ind).coil_mesh.vertices,2));

                opening_mean=mean(coil_parts(part_ind).coil_mesh.vertices(:,boundary_loop_nodes{1}),2);
                overall_mean=mean(coil_parts(part_ind).coil_mesh.vertices,2);
                old_orientation_vector=(opening_mean-overall_mean)./repmat(vecnorm(opening_mean-overall_mean),[3 1]);
                z_vec=[0 0 1]';
                sina = vecnorm(cross(old_orientation_vector,z_vec)) / ( vecnorm(old_orientation_vector) * vecnorm(z_vec) );
                cosa = vecnorm(dot(old_orientation_vector,z_vec)) / ( vecnorm(old_orientation_vector) * vecnorm(z_vec) );
                angle = atan2( sina, cosa );
                rotation_vector=cross(old_orientation_vector,[0 0 1]')./repmat(vecnorm(cross(old_orientation_vector,[0 0 1]')),[3 1]);
                rot_mat= calc_3d_rotation_matrix_by_vector(rotation_vector,angle);

                point_coords=rot_mat*coil_parts(part_ind).coil_mesh.vertices;
                min_z_cylinder=min(point_coords(3,:));
                point_coords(3,:)=point_coords(3,:)+min_z_cylinder;

                phi_coord=atan2(point_coords(2,:),point_coords(1,:));
                r_coord=(point_coords(1,:).^2+point_coords(2,:).^2).^(1/2);

                u_coord=( point_coords(3,:)-mean(r_coord)*circular_factor).*sin(phi_coord);
                v_coord=( point_coords(3,:)-mean(r_coord)*circular_factor).*cos(phi_coord);

                coil_parts(part_ind).coil_mesh.uv=[u_coord;v_coord];
                coil_parts(part_ind).coil_mesh.n=vertexNormal(triangulation(coil_parts(part_ind).coil_mesh.faces',coil_parts(part_ind).coil_mesh.vertices'))';
                coil_parts(part_ind).coil_mesh.boundary=boundary_loop_nodes;
            end
        end

    else
        % planar case (unchanged)
        mean_norm=mean(face_normals,1);
        new_norm=[0 0 1];
        v_c=cross(mean_norm,new_norm);
        if norm(v_c)>1e-8
            v_d=dot(mean_norm,new_norm);
            mat_v=[0 (-1)*v_c(3) v_c(2); v_c(3) 0 (-1)*v_c(1); (-1)*v_c(2) v_c(1) 0];
            rot_mat=eye(3)+mat_v+mat_v*mat_v*(1/(1+v_d));
            out_a=sum(repmat(rot_mat(1,:),[size(coil_parts(part_ind).coil_mesh.vertices,2) 1]).*coil_parts(part_ind).coil_mesh.vertices',2);
            out_b=sum(repmat(rot_mat(2,:),[size(coil_parts(part_ind).coil_mesh.vertices,2) 1]).*coil_parts(part_ind).coil_mesh.vertices',2);
            coil_parts(part_ind).coil_mesh.uv=[out_a'; out_b'];
        else
            coil_parts(part_ind).coil_mesh.uv=[coil_parts(part_ind).coil_mesh.vertices(1,:); coil_parts(part_ind).coil_mesh.vertices(2,:)];
        end

        boundary_edges=freeBoundary(triangulation(coil_parts(part_ind).coil_mesh.faces', ...
            [coil_parts(part_ind).coil_mesh.uv; zeros(1,size(coil_parts(part_ind).coil_mesh.uv,2))]'));
        coil_parts(part_ind).coil_mesh.boundary = boundary_loops_from_edges(boundary_edges, size(coil_parts(part_ind).coil_mesh.uv,2));

        coil_parts(part_ind).coil_mesh.n=vertexNormal(triangulation(coil_parts(part_ind).coil_mesh.faces',coil_parts(part_ind).coil_mesh.vertices'))';
    end

end

end

function mesh = flange_half_disc_as_cyl_extension_polar(mesh, circular_factor, diskScale)
% Fits a split-bore flange half:
% - inner cylinder surface at ~r_inner
% - annular flange disc at z_split
%
% The key fix is axis detection:
% choose the axis that gives the strongest axial concentration peak
% (the split plane), instead of the old low-radius-subset heuristic.

V = mesh.vertices; % 3xN
F = mesh.faces;    % 3xM
N = size(V,2);

% ---------- scale ----------
medEdge = median_edge_length(V, F);
if ~(isfinite(medEdge) && medEdge > 0)
    bbox = max(V,[],2) - min(V,[],2);
    medEdge = 1e-3 * max([bbox(:); 1]);
end

% ---------- pick flange axis robustly ----------
axis_vec = pick_flange_axis_by_split_peak(V, medEdge);
axis_vec = axis_vec / max(norm(axis_vec), 1e-12);

% ---------- rotate chosen axis to +z ----------
z_vec = [0;0;1];
v = cross(axis_vec, z_vec);
s = norm(v);
cosa = dot(axis_vec, z_vec);

if s < 1e-12
    if cosa > 0
        R = eye(3);
    else
        % 180 deg rotation about x-axis
        R = [1 0 0; 0 -1 0; 0 0 -1];
    end
else
    vx = [   0   -v(3)  v(2);
           v(3)   0    -v(1);
          -v(2)  v(1)   0  ];
    R = eye(3) + vx + vx*vx*((1-cosa)/(s^2));
end

Vr = R * V;
x = Vr(1,:);
y = Vr(2,:);
z = Vr(3,:);
r = hypot(x,y);

% ---------- detect split plane ----------
[z_split, disc_mask, ~] = detect_split_plane(z, medEdge);

% choose cylinder direction away from split plane
pos_span = max(z) - z_split;
neg_span = z_split - min(z);
if pos_span >= neg_span
    sideSign = +1;
else
    sideSign = -1;
end

% ---------- estimate join radius from non-disc vertices ----------
away = ~disc_mask;
if nnz(away) >= 10
    r_join = median(r(away));
else
    r_join = prctile(r, 10);
end

if ~(isfinite(r_join) && r_join > 0)
    r_pos = r(r > 0);
    if isempty(r_pos)
        r_join = 1.0;
    else
        r_join = median(r_pos);
    end
end

% disc radial distance from join
dr = r - r_join;
dr(dr < 0) = 0;

% effective axial coordinate:
% cylinder: positive extension away from split plane
% disc: negative extension proportional to radial growth
z_cyl = sideSign * (z - z_split);
z_cyl(z_cyl < 0) = 0;

z_eff = z_cyl;
z_eff(disc_mask) = -diskScale * dr(disc_mask);

% shift so everything is nonnegative
z_eff = z_eff - min(z_eff);

% ---------- seam-free polar embedding ----------
rho0 = max(r_join * circular_factor, 1e-6);
rho  = z_eff + rho0;

rr = max(r, 1e-12);
u  = rho .* (x ./ rr);
v2 = rho .* (y ./ rr);

mesh.uv = [u; v2];

% optional orientation cleanup
if is_uv_mirrored(mesh.uv, F)
    mesh.uv(1,:) = -mesh.uv(1,:);
end

mesh.n = vertexNormal(triangulation(F', V'))';

bnd_edges_uv = freeBoundary(triangulation(F', [mesh.uv; zeros(1,N)]'));
mesh.boundary = boundary_loops_from_edges(bnd_edges_uv, N);

end

% =========================================================================
function axis_vec = pick_flange_axis_by_split_peak(V, medEdge)
% Choose among PCA axes the one that produces the strongest axial peak.
% For a flange half, the true axis is the one normal to the split plane,
% because many vertices lie on that plane.

N  = size(V,2);
c0 = mean(V,2);
X  = V - c0;

C = (X*X.') / max(1, N-1);
[Q,D] = eig(C);
[~,ord] = sort(diag(D), 'descend');
Q = Q(:,ord);

h = max(0.5 * medEdge, 1e-6);

bestScore = -Inf;
axis_vec  = Q(:,3);

for ii = 1:3
    a = Q(:,ii);
    a = a / max(norm(a), 1e-12);

    t = a.' * X;

    bins = floor((t - min(t)) / h);
    bins = bins - min(bins);
    cnt  = accumarray(bins(:)+1, 1, [max(bins)+1 1]);

    peakFrac = max(cnt) / N;

    % tie-breaker:
    % true axis should also give a relatively tight inner-radius subset
    P = X - a*t;
    d = sqrt(sum(P.^2,1));
    d20 = prctile(d, 20);
    cylStd = std(d(d <= d20));

    score = peakFrac - 0.05 * (cylStd / max(medEdge, 1e-12));

    if score > bestScore
        bestScore = score;
        axis_vec = a;
    end
end
end

% =========================================================================
function [z_split, disc_mask, zTol] = detect_split_plane(z, medEdge)

h = max(0.5 * medEdge, 1e-6);

bins = floor((z - min(z)) / h);
bins = bins - min(bins);
cnt  = accumarray(bins(:)+1, 1, [max(bins)+1 1]);

[~,imax] = max(cnt);
mask0 = (bins == (imax-1));

z_split = median(z(mask0));

zTol = max(1.5*h, 0.5*medEdge);
disc_mask = abs(z - z_split) <= zTol;
end

% =========================================================================
function loops = boundary_loops_from_edges(edges, nVerts)
edges = double(edges);
adj = cell(nVerts,1);
for i=1:size(edges,1)
    a = edges(i,1); b = edges(i,2);
    adj{a} = [adj{a} b];
    adj{b} = [adj{b} a];
end

visited = sparse(nVerts,nVerts);
loops = {};

for i=1:size(edges,1)
    a = edges(i,1); b = edges(i,2);
    if visited(a,b), continue; end

    start = a;
    prev  = 0;
    curr  = a;
    next  = b;

    loop = curr;

    while true
        loop(end+1) = next; %#ok<AGROW>
        visited(curr,next) = 1;
        visited(next,curr) = 1;

        prev = curr;
        curr = next;

        if curr == start
            break;
        end

        nb = adj{curr};
        if isempty(nb), break; end

        if numel(nb) == 1
            next = nb(1);
        else
            if nb(1) ~= prev
                next = nb(1);
            else
                next = nb(2);
            end
        end

        if numel(loop) > nVerts + 5
            break;
        end
    end

    loops{end+1} = loop(:); %#ok<AGROW>
end
end

function m = median_edge_length(V,F)
i = F(1,:); j = F(2,:); k = F(3,:);
E = [i j; j k; k i];
E = sort(E,2);
E = unique(E,'rows');
d = vecnorm(V(:,E(:,1)) - V(:,E(:,2)), 2, 1);
m = median(d);
end

function tf = is_uv_mirrored(uv,F)
p = uv';
a = p(F(1,:),:);
b = p(F(2,:),:);
c = p(F(3,:),:);
area2 = (b(:,1)-a(:,1)).*(c(:,2)-a(:,2)) - (b(:,2)-a(:,2)).*(c(:,1)-a(:,1));
area2 = area2(isfinite(area2));
if isempty(area2)
    tf = false;
else
    tf = (median(area2) < 0);
end
end

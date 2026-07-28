% =========================
% File: build_split_bore_flange_mesh.m
% =========================
function mesh = build_split_bore_flange_mesh(edge_len, r_inner, r_outer, gap, half_length, varargin)
%BUILD_SPLIT_BORE_FLANGE_MESH
% Two disconnected split-bore "flanges" surface mesh including ONLY:
%   - inner cylindrical mantle at r = r_inner
%   - flange end ring (annulus) at z = +/- gap/2 spanning r_inner..r_outer
%
% Excludes:
%   - outer mantle
%   - far end caps (open at far ends)
%
% Output mesh struct:
%   mesh.vertices : (N x 3)
%   mesh.faces    : (M x 3)
%
% OPTIONAL (shielded system):
% You can generate TWO concentric split-bore flange meshes and return them merged
% into ONE mesh (useful as a shielded coil/support pair, or coil+shield surfaces).
%
% Usage:
%   mesh = build_split_bore_flange_mesh(edge_len, r_inner, r_outer, gap, half_length)
%
%   mesh = build_split_bore_flange_mesh(edge_len, r_inner, r_outer, gap, half_length, ...
%       'shield', true, ...
%       'shield_edge_len', edge_len2, ...
%       'shield_r_inner', r_inner2, ...
%       'shield_r_outer', r_outer2, ...
%       'shield_gap', gap2, ...
%       'shield_half_length', half_length2);
%
% Notes:
% - If 'shield' is true, the returned mesh contains BOTH surfaces concatenated.
% - To also get them separately, request outputs:
%     [mesh, mesh_primary, mesh_shield] = build_split_bore_flange_mesh(...)

% ---- parse optional shield arguments ----
p = inputParser;
p.FunctionName = 'build_split_bore_flange_mesh';

addParameter(p,'shield',false,@(x)islogical(x) || isnumeric(x));

addParameter(p,'shield_edge_len',[],@(x)isempty(x) || (isscalar(x) && x>0));
addParameter(p,'shield_r_inner', [],@(x)isempty(x) || isscalar(x));
addParameter(p,'shield_r_outer', [],@(x)isempty(x) || isscalar(x));
addParameter(p,'shield_gap',     [],@(x)isempty(x) || isscalar(x));
addParameter(p,'shield_half_length',[],@(x)isempty(x) || isscalar(x));

parse(p,varargin{:});
opt = p.Results;

% ---- build primary mesh ----
mesh_primary = build_one(edge_len, r_inner, r_outer, gap, half_length);

% ---- optionally build shield mesh and merge ----
if logical(opt.shield)
    edge_len2 = opt.shield_edge_len;
    r_inner2  = opt.shield_r_inner;
    r_outer2  = opt.shield_r_outer;
    gap2      = opt.shield_gap;
    half_len2 = opt.shield_half_length;

    if isempty(edge_len2), edge_len2 = edge_len; end
    if isempty(gap2),      gap2      = gap;      end
    if isempty(half_len2), half_len2 = half_length; end

    if isempty(r_inner2) || isempty(r_outer2)
        error('Shield requested, but shield_r_inner / shield_r_outer were not provided.');
    end

    mesh_shield = build_one(edge_len2, r_inner2, r_outer2, gap2, half_len2);

    mesh.vertices = [mesh_primary.vertices; mesh_shield.vertices];
    mesh.faces    = [mesh_primary.faces; mesh_shield.faces + size(mesh_primary.vertices,1)];
else
    mesh = mesh_primary;
    mesh_shield = [];
end

% allow optional outputs
if nargout >= 2
    varargout{1} = mesh_primary; %#ok<NASGU>
end
if nargout >= 3
    varargout{2} = mesh_shield; %#ok<NASGU>
end

    function mesh_out = build_one(edge_len_in, r_inner_in, r_outer_in, gap_in, half_length_in)

        edge_len_in = double(edge_len_in);
        r_inner_in  = double(r_inner_in);
        r_outer_in  = double(r_outer_in);
        gap_in      = double(gap_in);

        if nargin < 5 || isempty(half_length_in)
            half_length_in = 2.0 * r_outer_in;
        end
        half_length_in = double(half_length_in);

        if edge_len_in <= 0, error('edge_len must be > 0'); end
        if r_outer_in <= r_inner_in, error('r_outer must be > r_inner'); end
        if gap_in < 0, error('gap must be >= 0'); end
        if half_length_in <= 0, error('half_length must be > 0'); end

        n_theta = max(24, ceil((2*pi*r_outer_in)/edge_len_in));
        n_z     = max(1,  ceil(half_length_in/edge_len_in));
        n_r     = max(1,  ceil((r_outer_in-r_inner_in)/edge_len_in));

        theta = linspace(0, 2*pi, n_theta+1); theta(end) = [];
        ct = cos(theta);
        st = sin(theta);

        % Negative half: split at z=-gap/2, extends to more negative z
        z_neg_split = -0.5 * gap_in;
        z_neg_far   = z_neg_split - half_length_in;
        [Vn, Fn] = build_half(z_neg_split, z_neg_far);

        % Positive half: split at z=+gap/2, extends to more positive z
        z_pos_split =  0.5 * gap_in;
        z_pos_far   = z_pos_split + half_length_in;
        [Vp, Fp] = build_half(z_pos_split, z_pos_far);

        V = [Vn; Vp];
        F = [Fn; Fp + size(Vn,1)];

        mesh_out.vertices = V;
        mesh_out.faces    = F;

        % -------------------------
        % Nested builders
        % -------------------------
        function [Vh, Fh] = build_half(z_split, z_far)
            % Always use increasing z_levels to keep cylinder face winding consistent.
            if z_far < z_split
                z_levels = linspace(z_far, z_split, n_z+1);  % increasing
                split_j = n_z + 1;                           % last ring is split plane
            else
                z_levels = linspace(z_split, z_far, n_z+1);  % increasing
                split_j = 1;                                 % first ring is split plane
            end

            Vh = zeros(0,3);
            Fh = zeros(0,3);

            % ---- Inner mantle vertices ----
            inner_idx = zeros(n_z+1, n_theta);
            for j = 1:(n_z+1)
                zz = z_levels(j);
                base = size(Vh,1);
                Vh = [Vh; [r_inner_in*ct(:), r_inner_in*st(:), zz*ones(n_theta,1)]];
                inner_idx(j,:) = (base+1):(base+n_theta);
            end

            % ---- Inner mantle faces (normal toward axis => inward radial) ----
            Fh = [Fh; cylinder_faces(inner_idx, false)]; % outward=false => flip

            % ---- Flange annulus at split plane ----
            inner_ring = inner_idx(split_j, :);

            % Create intermediate + outer rings at z_split
            rings = zeros(n_r+1, n_theta);
            rings(1,:) = inner_ring;

            for k = 2:n_r
                rr = r_inner_in + (r_outer_in-r_inner_in)*((k-1)/n_r);
                base = size(Vh,1);
                Vh = [Vh; [rr*ct(:), rr*st(:), z_split*ones(n_theta,1)]];
                rings(k,:) = (base+1):(base+n_theta);
            end

            % Outer ring (k = n_r+1)
            base = size(Vh,1);
            Vh = [Vh; [r_outer_in*ct(:), r_outer_in*st(:), z_split*ones(n_theta,1)]];
            rings(n_r+1,:) = (base+1):(base+n_theta);

            % Ring normal toward the gap:
            %   z_split < 0  -> +z
            %   z_split > 0  -> -z
            normal_plus_z = (z_split < 0);

            Fh = [Fh; annulus_faces(rings, normal_plus_z)];
        end

        function Fc = cylinder_faces(ring_idx, outward)
            % ring_idx: (n_z+1 x n_theta), z increasing with j
            Fc = zeros(0,3);
            for j = 1:n_z
                for i = 1:n_theta
                    i2 = i + 1; if i2 > n_theta, i2 = 1; end

                    v00 = ring_idx(j,i);
                    v10 = ring_idx(j,i2);
                    v01 = ring_idx(j+1,i);
                    v11 = ring_idx(j+1,i2);

                    if mod(i+j,2) == 0
                        tris = [v00 v10 v11; v00 v11 v01];
                    else
                        tris = [v00 v10 v01; v10 v11 v01];
                    end

                    if ~outward
                        tris = tris(:,[1 3 2]);
                    end

                    Fc = [Fc; tris];
                end
            end
        end

        function Fa = annulus_faces(rings, normal_plus_z)
            % rings: (n_r+1 x n_theta), radial index increasing with row
            Fa = zeros(0,3);
            for k = 1:n_r
                for i = 1:n_theta
                    i2 = i + 1; if i2 > n_theta, i2 = 1; end

                    v00 = rings(k,   i);
                    v10 = rings(k,   i2);
                    v01 = rings(k+1, i);
                    v11 = rings(k+1, i2);

                    if mod(i+k,2) == 0
                        tris = [v00 v10 v11; v00 v11 v01];
                    else
                        tris = [v00 v10 v01; v10 v11 v01];
                    end

                    if ~normal_plus_z
                        tris = tris(:,[1 3 2]);
                    end

                    Fa = [Fa; tris];
                end
            end
        end

    end
end
function mesh = create_split_cylinder_mesh(edge_len, r_inner, gap, half_length)
% Creates a single split inner-cylinder mantle mesh (two disconnected halves).
% No flange, no caps, no outer surface.
% INPUTS
%   edge_len    (scalar, >0)  Target mesh edge length (approx). Controls
%                             discretization in theta and z.
%
%   r_inner     (scalar, >0)  Cylinder radius of the mantle surface.
%
%   gap         (scalar, >=0) Axial separation between the two halves.
%                             The split planes are at z = -gap/2 and z = +gap/2.
%                             If gap = 0, the two halves meet at z = 0 but
%                             remain disconnected (no faces across the plane).
%
%   half_length (scalar, >0)  Axial length of each half (measured away from the
%                             split plane). Each half spans:
%                               negative half: z in [-(gap/2 + half_length), -gap/2]
%                               positive half: z in [ +gap/2, +(gap/2 + half_length)]
edge_len    = double(edge_len);
r_inner     = double(r_inner);
gap         = double(gap);
half_length = double(half_length);

if edge_len <= 0,    error('edge_len must be > 0'); end
if r_inner <= 0,     error('r_inner must be > 0'); end
if gap < 0,          error('gap must be >= 0'); end
if half_length <= 0, error('half_length must be > 0'); end

n_theta = max(24, ceil((2*pi*r_inner)/edge_len));
n_z     = max(1,  ceil(half_length/edge_len));

theta = linspace(0, 2*pi, n_theta+1); theta(end) = [];
ct = cos(theta(:));
st = sin(theta(:));

% split locations
z_neg_far   = -gap/2 - half_length;  z_neg_split = -gap/2;
z_pos_split =  gap/2;                z_pos_far   =  gap/2 + half_length;

[Vn, Fn] = build_half(z_neg_far,  z_neg_split);
[Vp, Fp] = build_half(z_pos_split, z_pos_far);

mesh.vertices = [Vn; Vp];
mesh.faces    = [Fn; Fp + size(Vn,1)];

    function [Vh, Fh] = build_half(z0, z1)
        z = linspace(min(z0,z1), max(z0,z1), n_z+1).';           % increasing
        Z = repelem(z, n_theta);
        X = r_inner * repmat(ct, numel(z), 1);
        Y = r_inner * repmat(st, numel(z), 1);
        Vh = [X Y Z];

        idx = reshape(1:(n_theta*(n_z+1)), n_theta, n_z+1).';    % (n_z+1 x n_theta)
        Fh = zeros(2*n_theta*n_z, 3);
        t = 0;
        for j = 1:n_z
            a = idx(j,:);                 b = idx(j,[2:end 1]);
            c = idx(j+1,:);               d = idx(j+1,[2:end 1]);

            tri1 = [a(:) b(:) d(:)];
            tri2 = [a(:) d(:) c(:)];

            % flip winding so normals point toward axis (inward)
            tri1 = tri1(:,[1 3 2]);
            tri2 = tri2(:,[1 3 2]);

            Fh(t+(1:n_theta),:) = tri1; t = t + n_theta;
            Fh(t+(1:n_theta),:) = tri2; t = t + n_theta;
        end
    end
end
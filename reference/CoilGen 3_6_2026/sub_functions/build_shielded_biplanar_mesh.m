% =========================
% File: build_shielded_biplanar_mesh.m
% =========================
function biplanar_mesh = build_shielded_biplanar_mesh( ...
    primary_outer_radius, primary_edge_len, ...
    shield_outer_radius,  shield_edge_len, ...
    target_normal_x, target_normal_y, target_normal_z, ...
    center_position_x, center_position_y, center_position_z, ...
    plane_distance, shield_gap)
%BUILD_SHIELDED_BIPLANAR_MESH
% Two pairs of coaxial circular discs (primary + shield), oriented by target normal.
%
% Inputs:
%   primary_outer_radius, primary_edge_len : primary disc radius + target edge length
%   shield_outer_radius,  shield_edge_len  : shield disc radius + target edge length
%   target_normal_*                       : plane normal (will be normalized)
%   center_position_*                     : center point of the biplanar system
%   plane_distance                        : distance between the two primary planes
%   shield_gap                            : offset between primary and shield on each side
%                                          (shield planes at z = +/- (plane_distance/2 + shield_gap))
%
% Output:
%   biplanar_mesh.vertices (N x 3)
%   biplanar_mesh.faces    (M x 3)

% ---- validate ----
R1 = double(primary_outer_radius);  h1 = double(primary_edge_len);
R2 = double(shield_outer_radius);   h2 = double(shield_edge_len);
d  = double(plane_distance);        g  = double(shield_gap);

if R1 <= 0 || R2 <= 0, error('outer_radius must be > 0'); end
if h1 <= 0 || h2 <= 0, error('edge_len must be > 0'); end
if d <  0, error('plane_distance must be >= 0'); end
if g <  0, error('shield_gap must be >= 0'); end

target_normal = [double(target_normal_x); double(target_normal_y); double(target_normal_z)];
nt = norm(target_normal);
if nt == 0, error('target_normal must be nonzero'); end
target_normal = target_normal / nt;

center = [double(center_position_x); double(center_position_y); double(center_position_z)];

% ---- base meshes in XY (faces oriented for +Z) ----
[P1, F1] = disc_trimesh(R1, h1);
[P2, F2] = disc_trimesh(R2, h2);

% ---- rotation: map +Z to target_normal ----
Rmat = rot_from_z_to(target_normal);

% plane locations in local Z
zP = d/2;            % primary half distance
zS = d/2 + g;        % shield half distance (offset from primary)

% ---- build 4 discs: primary(+/-), shield(+/-) ----
[Vpt, Fpt] = place_disc(P1, F1, +zP, true);   % top primary, normal +target
[Vpb, Fpb] = place_disc(P1, F1, -zP, false);  % bottom primary, normal -target
[Vst, Fst] = place_disc(P2, F2, +zS, true);   % top shield,  normal +target
[Vsb, Fsb] = place_disc(P2, F2, -zS, false);  % bottom shield,normal -target

% ---- merge ----
V = [Vpt; Vpb; Vst; Vsb];
n1 = size(Vpt,1);
n2 = size(Vpb,1);
n3 = size(Vst,1);

F = [Fpt; ...
     Fpb + n1; ...
     Fst + n1 + n2; ...
     Fsb + n1 + n2 + n3];

biplanar_mesh.vertices = V;
biplanar_mesh.faces    = F;

% =========================
% Helpers
% =========================
    function [Vdisc, Fdisc] = place_disc(Pxy, Fxy, z0, normal_plus)
        % Pxy: (N x 2), Fxy: (M x 3) oriented +Z in local XY
        N = size(Pxy,1);
        Plocal = [Pxy(:,2)'; Pxy(:,1)'; z0*ones(1,N)]; % keep (y,x,z) convention from original code
        Vdisc = (Rmat * Plocal + center);
        Vdisc = Vdisc.'; % (N x 3)

        if normal_plus
            Fdisc = Fxy;
        else
            Fdisc = Fxy(:,[1 3 2]); % flip -> -Z in local -> -target after rotation
        end
    end

    function [P, F] = disc_trimesh(R, h)
        % Near-uniform triangles via hex lattice points + Delaunay.
        dy = h*sqrt(3)/2;
        ys = (-R:dy:R).';
        P  = zeros(0,2);

        for r = 1:numel(ys)
            y0 = ys(r);
            xmax = sqrt(max(0, R*R - y0*y0));
            xoff = mod(r,2) * (h/2);
            xs = (-xmax:h:xmax) + xoff;
            xs = xs(abs(xs) <= xmax + 1e-12);
            P  = [P; [xs(:), y0*ones(numel(xs),1)]];
        end

        nB = max(24, ceil((2*pi*R)/h));
        th = linspace(0,2*pi,nB+1).'; th(end) = [];
        P  = [P; [R*cos(th), R*sin(th)]];

        % remove near-duplicates
        tol = h*1e-6;
        Pq = round(P/tol)*tol;
        [~,ia] = unique(Pq,'rows','stable');
        P = P(ia,:);

        T = delaunayTriangulation(P);
        F = T.ConnectivityList;

        % orient faces to +Z in local XY
        x = P(:,1); y = P(:,2);
        i = F(:,1); j = F(:,2); k = F(:,3);
        s = (x(j)-x(i)).*(y(k)-y(i)) - (y(j)-y(i)).*(x(k)-x(i));
        flip = (s < 0);
        F(flip,[2 3]) = F(flip,[3 2]);
    end

    function R = rot_from_z_to(n)
        % Robust rotation mapping [0;0;1] -> n (unit).
        ez = [0;0;1];
        c  = max(-1, min(1, dot(ez,n)));
        if abs(c-1) < 1e-14
            R = eye(3);
            return;
        elseif abs(c+1) < 1e-14
            % 180 deg about X (any axis orthogonal to ez works)
            R = [1 0 0; 0 -1 0; 0 0 -1];
            return;
        end
        v = cross(ez,n);
        s = norm(v);
        v = v / s;
        ang = acos(c);
        R = axis_angle(v, ang);
    end

    function R = axis_angle(u, a)
        % Rodrigues' rotation formula
        ux=u(1); uy=u(2); uz=u(3);
        sa = sin(a); ca = cos(a); t = 1-ca;
        R = [ca+ux*ux*t,    ux*uy*t-uz*sa, ux*uz*t+uy*sa; ...
             uy*ux*t+uz*sa, ca+uy*uy*t,    uy*uz*t-ux*sa; ...
             uz*ux*t-uy*sa, uz*uy*t+ux*sa, ca+uz*uz*t];
    end
end

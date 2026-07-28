function mesh = build_two_coaxial_discs_mesh(edge_len, outer_radius, spacing)
% Two filled discs at z = ±spacing/2 with near-uniform triangle shapes.

edge_len     = double(edge_len);
outer_radius = double(outer_radius);
spacing      = double(spacing);

if edge_len <= 0,     error('edge_len must be > 0'); end
if outer_radius <= 0, error('outer_radius must be > 0'); end
if spacing < 0,       error('spacing must be >= 0'); end

P = build_disk_points(edge_len, outer_radius);          % (N x 2) in XY
F2 = delaunayTriangulation(P).ConnectivityList;         % triangles (unoriented)

% enforce CCW winding in XY -> +Z normal
x = P(:,1); y = P(:,2);
i = F2(:,1); j = F2(:,2); k = F2(:,3);
s = (x(j)-x(i)).*(y(k)-y(i)) - (y(j)-y(i)).*(x(k)-x(i));
flip = (s < 0);
F2(flip,[2 3]) = F2(flip,[3 2]);

z1 = -spacing/2; z2 = +spacing/2;
V1 = [P, z1*ones(size(P,1),1)];
V2 = [P, z2*ones(size(P,1),1)];

% make top disc face downward (toward the gap). If you want both +Z, use F2 for both.
Fbot = F2;                 % +Z
Ftop = F2(:,[1 3 2]);      % -Z

mesh.vertices = [V1; V2];
mesh.faces    = [Fbot; Ftop + size(V1,1)];

    function P = build_disk_points(h, R)
        % Hex/triangular lattice points inside circle + dense boundary ring.
        dy = h*sqrt(3)/2;
        ys = (-R:dy:R).';
        P = zeros(0,2);

        for row = 1:numel(ys)
            y0  = ys(row);
            xoff = mod(row,2) * (h/2);
            xs  = (-R:h:R) + xoff;
            xx  = xs(:);
            yy  = y0 * ones(size(xx));
            m   = (xx.^2 + yy.^2) <= (R^2 + 1e-12);
            P   = [P; [xx(m) yy(m)]];
        end

        nB = max(24, ceil((2*pi*R)/h));
        th = linspace(0,2*pi,nB+1).'; th(end) = [];
        P  = [P; [R*cos(th) R*sin(th)]];

        % remove near-duplicates robustly
        tol = h*1e-6;
        Pq  = round(P/tol)*tol;
        [~,ia] = unique(Pq,'rows','stable');
        P = P(ia,:);
    end
end
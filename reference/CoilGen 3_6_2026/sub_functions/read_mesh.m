function [coil_mesh,target_mesh,shielded_mesh]=read_mesh(input)


%Read the input mesh
if strcmp(input.sf_source_file,'none')


if strcmp(input.coil_mesh_file(end-3:end),'.stl')
%Load the stl file; read the coil mesh surface
if ispc
coil_mesh = stlread_local(strcat(input.geometry_source_path,'\',input.coil_mesh_file));
else
coil_mesh = stlread_local(strcat(input.geometry_source_path,'/',input.coil_mesh_file));
end
coil_mesh=create_unique_noded_mesh(coil_mesh);
%Change to format convenction used here
coil_mesh.vertices=coil_mesh.vertices'; 
coil_mesh.faces=coil_mesh.faces'; 

elseif strcmp(input.coil_mesh_file,'create cylinder mesh')
%no external mesh is specified by stl file; create default cylndrical mesh
coil_mesh=build_cylinder_mesh(input.cylinder_mesh_parameter_list(1),input.cylinder_mesh_parameter_list(2),input.cylinder_mesh_parameter_list(3),input.cylinder_mesh_parameter_list(4),...
                                                input.cylinder_mesh_parameter_list(5),input.cylinder_mesh_parameter_list(6),input.cylinder_mesh_parameter_list(7),input.cylinder_mesh_parameter_list(8));
coil_mesh=create_unique_noded_mesh(coil_mesh);
coil_mesh.vertices=coil_mesh.vertices'; 
coil_mesh.faces=coil_mesh.faces'; 

elseif strcmp(input.coil_mesh_file,'create double cone mesh')
%no external mesh is specified by stl file; create default cylndrical mesh
coil_mesh=build_double_cone_mesh(input.double_cone_mesh_parameter_list(1),input.double_cone_mesh_parameter_list(2),input.double_cone_mesh_parameter_list(3),input.double_cone_mesh_parameter_list(4),...
                                                input.double_cone_mesh_parameter_list(5),input.double_cone_mesh_parameter_list(6),input.double_cone_mesh_parameter_list(7),input.double_cone_mesh_parameter_list(8),...
                                                input.double_cone_mesh_parameter_list(9),input.double_cone_mesh_parameter_list(10));
coil_mesh=create_unique_noded_mesh(coil_mesh);
coil_mesh.vertices=coil_mesh.vertices'; 
coil_mesh.faces=coil_mesh.faces'; 

elseif strcmp(input.coil_mesh_file,'create planary mesh')
%no external mesh is specified by stl file; create default planar mesh
coil_mesh=build_planar_mesh(input.planar_mesh_parameter_list(1),input.planar_mesh_parameter_list(2),input.planar_mesh_parameter_list(3),input.planar_mesh_parameter_list(4),...
                                                input.planar_mesh_parameter_list(5),input.planar_mesh_parameter_list(6),input.planar_mesh_parameter_list(7),input.planar_mesh_parameter_list(8),...
                                                input.planar_mesh_parameter_list(9),input.planar_mesh_parameter_list(10),input.planar_mesh_parameter_list(11));
coil_mesh=create_unique_noded_mesh(coil_mesh);
coil_mesh.vertices=coil_mesh.vertices'; 
coil_mesh.faces=coil_mesh.faces'; 


elseif strcmp(input.coil_mesh_file,'create circular mesh')
%no external mesh is specified by stl file; create default planar mesh
coil_mesh=build_circular_mesh(input.circular_mesh_parameter_list(1),input.circular_mesh_parameter_list(2),input.circular_mesh_parameter_list(3),input.circular_mesh_parameter_list(4),...
                                                input.circular_mesh_parameter_list(5),input.circular_mesh_parameter_list(6),input.circular_mesh_parameter_list(7),input.circular_mesh_parameter_list(8),...
                                                input.circular_mesh_parameter_list(9));
coil_mesh=create_unique_noded_mesh(coil_mesh);
coil_mesh.vertices=coil_mesh.vertices'; 
coil_mesh.faces=coil_mesh.faces'; 


elseif strcmp(input.coil_mesh_file,'create bi-planary mesh')
%no external mesh is specified by stl file; create default planar mesh
coil_mesh=build_biplanar_mesh(input.biplanar_mesh_parameter_list(1),input.biplanar_mesh_parameter_list(2),input.biplanar_mesh_parameter_list(3),input.biplanar_mesh_parameter_list(4),...
                                                input.biplanar_mesh_parameter_list(5),input.biplanar_mesh_parameter_list(6),input.biplanar_mesh_parameter_list(7),input.biplanar_mesh_parameter_list(8),...
                                                input.biplanar_mesh_parameter_list(9),input.biplanar_mesh_parameter_list(10),input.biplanar_mesh_parameter_list(11));
coil_mesh=create_unique_noded_mesh(coil_mesh);
coil_mesh.vertices=coil_mesh.vertices'; 
coil_mesh.faces=coil_mesh.faces'; 

elseif strcmp(input.coil_mesh_file,'create shielded bi-planary mesh')
    % no external mesh is specified by stl file; create shielded circular biplanar mesh (4 discs)
    p = input.shielded_circular_biplanar_mesh_parameter_list; 
    % expected:
    % [primary_outer_radius primary_edge_len shield_outer_radius shield_edge_len ...
    %  target_normal_x target_normal_y target_normal_z ...
    %  center_x center_y center_z plane_distance shield_gap]

    coil_mesh = build_shielded_biplanar_mesh( ...
        p(1), p(2), p(3), p(4), ...
        p(5), p(6), p(7), ...
        p(8), p(9), p(10), ...
        p(11), p(12));

    coil_mesh = create_unique_noded_mesh(coil_mesh);
    coil_mesh.vertices = coil_mesh.vertices';
    coil_mesh.faces    = coil_mesh.faces';

elseif strcmp(input.coil_mesh_file,'create flanged split bore mesh')
% Create split-bore flange mesh (primary only)
v = input.flanged_split_bore_mesh_parameter_list(:).';

if numel(v) ~= 5
    error('flanged_split_bore_mesh_parameter_list must be [edge_len r_inner r_outer gap half_length].');
end
coil_mesh = build_split_bore_flange_mesh(v(1), v(2), v(3), v(4), v(5));
coil_mesh = create_unique_noded_mesh(coil_mesh);
coil_mesh.vertices = coil_mesh.vertices';
coil_mesh.faces    = coil_mesh.faces';

elseif strcmp(input.coil_mesh_file,'create shielded flanged split bore mesh')
% Create split-bore flange mesh + shield mesh merged into ONE surface mesh
v = input.shielded_flanged_split_bore_mesh_parameter_list(:).';
if numel(v) ~= 10
    error(['shielded_flanged_split_bore_mesh_parameter_list must be ' ...
           '[edge_len r_inner r_outer gap half_length shield_edge_len shield_r_inner shield_r_outer shield_gap shield_half_length].']);
end
edge_len  = v(1);  r_inner  = v(2);  r_outer  = v(3);  gap  = v(4);  half_len  = v(5);
edge_len2 = v(6);  r_inner2 = v(7);  r_outer2 = v(8);  gap2 = v(9);  half_len2 = v(10);
% allow NaN to inherit from primary
if isnan(edge_len2), edge_len2 = edge_len; end
if isnan(gap2),      gap2      = gap;      end
if isnan(half_len2), half_len2 = half_len; end
coil_mesh = build_split_bore_flange_mesh(edge_len, r_inner, r_outer, gap, half_len, ...
    'shield', true, ...
    'shield_edge_len', edge_len2, ...
    'shield_r_inner',  r_inner2, ...
    'shield_r_outer',  r_outer2, ...
    'shield_gap',      gap2, ...
    'shield_half_length', half_len2);

coil_mesh = create_unique_noded_mesh(coil_mesh);
coil_mesh.vertices = coil_mesh.vertices';
coil_mesh.faces    = coil_mesh.faces';

end



else

if ispc
loaded_file=load(strcat(cd,'\',input.sf_source_file));
else
loaded_file=load(strcat(cd,'/',input.sf_source_file));
end
coil_mesh=loaded_file.coil_mesh;

end


%Read the target mesh surface
if ~strcmp(input.target_mesh_file,'none')
if ispc
target_mesh = stlread_local(strcat(input.geometry_source_path,'\',input.target_mesh_file));
else
target_mesh = stlread_local(strcat(input.geometry_source_path,'/',input.target_mesh_file));
end
target_mesh=create_unique_noded_mesh(target_mesh);
else
target_mesh=[];
end

%Read the shielded mesh surface
if ~strcmp(input.secondary_target_mesh_file,'none')
if ispc
shielded_mesh = stlread_local(strcat(input.geometry_source_path,'\',input.secondary_target_mesh_file));
else
shielded_mesh = stlread_local(strcat(input.geometry_source_path,'/',input.secondary_target_mesh_file));
end
shielded_mesh=create_unique_noded_mesh(shielded_mesh);
else
shielded_mesh=[];
end


% Create suppression surface mesh
if strcmp(input.create_split_bore_field_suppression_surface,'true')
    p = input.split_bore_area_suppression_parameters; % [edge_len r_inner gap half_length]
    shielded_mesh = create_split_cylinder_mesh(p(1), p(2), p(3), p(4));
elseif strcmp(input.create_circular_biplanar_field_suppression_surface,'true')
    p = input.circular_biplanar_area_suppression_parameters; % [edge_len outer_radius plane_distance]
    shielded_mesh = build_two_coaxial_discs_mesh(p(1), p(2), p(3));
end



% %Permute the nodes of the current carrying mesh as well as the target
% %points
% %Distribue the index of the nodes randomly
% permutet_indices=randperm(size(coil_mesh.vertices,2));
% coil_mesh.faces=[permutet_indices(coil_mesh.faces(:,1))' permutet_indices(coil_mesh.faces(:,2))' permutet_indices(coil_mesh.faces(:,3))'];
% non_permutet_vertices=coil_mesh.vertices;
% for vert_ind=1:size(coil_mesh.vertices,2)
%     coil_mesh.vertices(:,vert_ind)=non_permutet_vertices(:,permutet_indices==vert_ind);
% end





function unique_noded_mesh=create_unique_noded_mesh(non_unique_mesh)
faces=non_unique_mesh.faces;
verts=non_unique_mesh.vertices;
if size(faces,2)~=3
faces=faces';
end
if size(verts,2)~=3
verts=verts';
end
%Remove double nodes
[unique_verts,~,unique_inds] = unique(verts, 'rows','stable');
unique_assignments= [[1:numel(unique_inds)]' unique_inds];
for tri_ind=1:size(faces,1)
faces(tri_ind,1)=unique_assignments(faces(tri_ind,1),2);
faces(tri_ind,2)=unique_assignments(faces(tri_ind,2),2);
faces(tri_ind,3)=unique_assignments(faces(tri_ind,3),2);
end
unique_noded_mesh.faces=faces;
unique_noded_mesh.vertices=unique_verts;
end



function varargout = stlread_local(file)

    if ~exist(file,'file')
        error(['File ''%s'' not found. If the file is not on MATLAB''s path' ...
               ', be sure to specify the full path to the file.'], file);
    end
    
    fid = fopen(file,'r');    
    if ~isempty(ferror(fid))
        error(lasterror); %#ok
    end
    
    M = fread(fid,inf,'uint8=>uint8');
    fclose(fid);
    
    [f,v,n] = stlbinary(M);
    %if( isbinary(M) ) % This may not be a reliable test
    %    [f,v,n] = stlbinary(M);
    %else
    %    [f,v,n] = stlascii(M);
    %end
    
    varargout = cell(1,nargout);
    switch nargout        
        case 2
            varargout{1} = f;
            varargout{2} = v;
        case 3
            varargout{1} = f;
            varargout{2} = v;
            varargout{3} = n;
        otherwise
            varargout{1} = struct('faces',f,'vertices',v);
    end

function [F,V,N] = stlbinary(M)
    F = [];
    V = [];
    N = [];
    
    if length(M) < 84
        error('MATLAB:stlread:incorrectFormat', ...
              'Incomplete header information in binary STL file.');
    end
    
    % Bytes 81-84 are an unsigned 32-bit integer specifying the number of faces
    % that follow.
    numFaces = typecast(M(81:84),'uint32');
    %numFaces = double(numFaces);
    if numFaces == 0
        warning('MATLAB:stlread:nodata','No data in STL file.');
        return
    end
    
    T = M(85:end);
    F = NaN(numFaces,3);
    V = NaN(3*numFaces,3);
    N = NaN(numFaces,3);
    
    numRead = 0;
    while numRead < numFaces
        % Each facet is 50 bytes
        %  - Three single precision values specifying the face normal vector
        %  - Three single precision values specifying the first vertex (XYZ)
        %  - Three single precision values specifying the second vertex (XYZ)
        %  - Three single precision values specifying the third vertex (XYZ)
        %  - Two unused bytes
        i1    = 50 * numRead + 1;
        i2    = i1 + 50 - 1;
        facet = T(i1:i2)';
        
        n  = typecast(facet(1:12),'single');
        v1 = typecast(facet(13:24),'single');
        v2 = typecast(facet(25:36),'single');
        v3 = typecast(facet(37:48),'single');
        
        n = double(n);
        v = double([v1; v2; v3]);
        
        % Figure out where to fit these new vertices, and the face, in the
        % larger F and V collections.        
        fInd  = numRead + 1;        
        vInd1 = 3 * (fInd - 1) + 1;
        vInd2 = vInd1 + 3 - 1;
        
        V(vInd1:vInd2,:) = v;
        F(fInd,:)        = vInd1:vInd2;
        N(fInd,:)        = n;
        
        numRead = numRead + 1;
    end
    
end
function [F,V,N] = stlascii(M)
    warning('MATLAB:stlread:ascii','ASCII STL files currently not supported.');
    F = [];
    V = [];
    N = [];
end
% TODO: Change the testing criteria! Some binary STL files still begin with
% 'solid'.
function tf = isbinary(A)
% ISBINARY uses the first line of an STL file to identify its format.
    if isempty(A) || length(A) < 5
        error('MATLAB:stlread:incorrectFormat', ...
              'File does not appear to be an ASCII or binary STL file.');
    end    
    if strcmpi('solid',char(A(1:5)'))
        tf = false; % ASCII
    else
        tf = true;  % Binary
    end
end
end


end

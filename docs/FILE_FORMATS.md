# File Formats

## CSV Target Field Maps

Vector field schema:

```text
x_m,y_m,z_m,bx,by,bz
```

Scalar Bz-map schema:

```text
x_m,y_m,z_m,lin_x,lin_y,lin_z,weight,suppressed,group
```

Set `target_field_definition_field_name = "lin_y"` to use the `lin_y` column as
Bz. Optional columns are `weight`, `suppressed`, `group`, and
`dbz_dx,dbz_dy,dbz_dz`.

CoilGen treats imported field values as tesla unless
`target_field_units = "Hz"` is set. Hz maps are converted with the proton
gyromagnetic ratio `42.57747892e6 Hz/T`; `field_display_units = "Hz"` only
changes reporting and plotting scale.

Supported target-map files: `.csv`, `.tsv`, `.json`, `.npz`.

## Preoptimized Stream Functions

Preferred CSV/TSV schema:

```text
part,vertex_index,x_m,y_m,z_m,stream_function
1,0,0.1,0.0,-0.125,-0.125
```

`part` is 1-based for readability. `vertex_index` may be 0-based or 1-based.
Coordinate columns are optional when indices are present, but they make files
auditable. The mesh used during import must match the mesh used to create the
stream-function values.

Supported stream-function files: `.csv`, `.tsv`, `.json`, `.npz`.

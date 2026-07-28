from __future__ import annotations

from projects._shared.object_parameters import object_parameters

from projects._shared.example_utils import example_output_dir, parse_example_parameters

from coilgen import write_coil_geometry_api_template


from projects.general_examples.parameters import (
    CREATE_NEW_GEOMETRY_API_TEMPLATE_DEFAULT_PARAMETERS as DEFAULT_PARAMETERS,
)


################################################################################
# USER SETTINGS
################################################################################
# Run this when a new generated geometry is added to coilgen.geometry.
PARAMETERS = object_parameters(
    {
        "template": {
            # Human-readable geometry name. The template turns this into a CoilDesign subclass.
            "geometry_name": "custom saddle surface",
            # Python class name for the user-facing design object.
            "class_name": "CustomSaddleSurfaceCoilDesign",
            # Generated template file below outputs/new_geometry_api_template/.
            "output_filename": "custom_saddle_surface_design_api.py",
        }
    }
)


################################################################################
# SCRIPT ENTRY POINT
################################################################################


def main():
    parameters = parse_example_parameters(
        PARAMETERS,
        DEFAULT_PARAMETERS,
        example_name="create_new_geometry_api_template",
    )
    settings = parameters["template"]
    output_path = example_output_dir("new_geometry_api_template") / settings["output_filename"]
    path = write_coil_geometry_api_template(
        settings["geometry_name"],
        output_path,
        class_name=settings["class_name"],
    )
    print(f"Wrote user-facing geometry API template: {path}")
    return path


if __name__ == "__main__":
    path = main()

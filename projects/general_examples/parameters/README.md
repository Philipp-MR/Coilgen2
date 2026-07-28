# General Example Default Parameters

This folder stores fallback parameter dictionaries for the runnable scripts in
`projects/general_examples/scripts`.

Normal use:

1. Open an example script in Spyder.
2. Edit only its `PARAMETERS` block.
3. Run the script.

If a known parameter is missing from the script, `parse_example_parameters`
fills it from `defaults.py` and prints the value that was used.

Edit `defaults.py` only when changing shared advanced fallback settings or when
adding a new general example script.

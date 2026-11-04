# CoilGen2 browser workbench

## Run your first coil

1. From the CoilGen2 folder on Windows, double-click `Setup CoilGen2.cmd`. Once setup finishes, open `Start CoilGen2 Browser.cmd`. Keep the launcher window open while you work. By default the browser opens [the local workbench](http://127.0.0.1:8765/) with no login.
2. Choose **Browse examples…**. Under **Template**, select **Gradients · Cylindrical Y gradient (Gy)** and click **Load template**. The preview shows the cylindrical support. The target is a 20 mT/m gradient along world y over a 40 mm diameter spherical imaging volume.
3. Click **Compute full design** in the toolbar. Follow the active channel and revision in **Design plan**. To stop a queued or running job, use **Cancel**. For a partial run with saved results, choose the desired stage under **Compute to** before starting.
4. When the run finishes, the result opens above the plot. Check the design name and revision in **Viewing**. If you switched to the model preview, click **Coil results** to return. **Run design choices** shows the submitted support, target, optimization, smoothing, winding and validation settings. In the reference Gy run, the stream function field passed its validation check and the conductor failed the bend and clearance checks. Open **Engineering checks → Strict fabrication review** to read the measured values and limits.
5. Use **Review conductor centerlines** to inspect the connected wire. Switch to **Contours before routing** to see the fitted loops, or **Independent stream function field validation** with **Component** set to **BdotB0** to inspect the stream function field. Open **File → Save / download**, click **Save project**, then **Download project** to keep the editable `.coilgen` file with its verified results.
6. To export geometry for inspection, choose **Export conductor geometry** in **Model Builder**, expand **Choose files and formats** and select the paths or solids. Choose the length unit for STL/CSV. If exporting a solid with failed checks, review the DRAFT notice and acknowledgment. Click **Export selected** and download the ZIP. Its manifest lists the geometry, units, checksums and source revision.

[Choosing an example](../validation/gui_winding_quality_2026-11-03/browser-workbench-start.jpg) · [Completed winding in the browser workbench](../validation/gui_winding_quality_2026-11-03/browser-workbench-results.jpg)

### Install, update or choose a port

CoilGen2 requires Python 3.10 or newer. Windows setup installs the checkout's dependencies, including the `web` extra, in `%USERPROFILE%\.coilgen2\venv` by default. Set `COILGEN_VENV` to choose another environment. After updating or relocating the checkout, close the hub, rerun setup from that folder and restart the launcher.

In an existing virtual environment, install and start from the repository root.

```powershell
python -m pip install -e ".[web]"
python -m coilgen.web
```

`python -m coilgen.web --port 8877` uses a different port. `--no-browser` prints the URL without opening another tab. The browser and desktop use the same Python design engine. When comparing runs, record the checkout revision, package versions and thread settings as well as the project revision. [Installation guide](INSTALL.md)

Use `--threads-per-worker 2` when launching the browser in either local or team mode to repeat a study with the same native thread count. The launcher prints the effective value and sets the BLAS and OpenMP limits in each new worker before libraries load, including solver-bundled OpenBLAS. In the retest, some rejected connected paths differed by millimetres between runs made with different thread counts and software revisions even though the stream function fields agreed to numerical precision. Compare physical paths, spacing and bend checks alongside field maps.

## Edit the design

The browser and desktop arrange a wide window the same way. **Model Builder** is on the left, the plot and results occupy the center, and **Design plan** is on the right. On a narrow browser window, choose the panel you need using **Model Builder**, **Plot and results** or **Design plan**. File transfers go through the browser workspace. The desktop provides additional dedicated field-analysis and research windows. [Desktop workbench](NATIVE_WORKBENCH.md)

Select a section in **Model Builder** to open its settings. **Apply** commits the edits. **Discard edits** restores the last applied values and leaves the settings open. **Cancel** closes the settings and discards unapplied changes. Choose Apply or Cancel before switching sections, channels or projects, saving, or computing. Use **Undo** and **Redo** in Model Builder to revisit applied changes.

Use **File** to switch examples, open a project or save your work. **New cylindrical design** starts with an editable cylinder and a Y-gradient target. To reproduce a gallery study, choose **Browse examples…** and load the named template.

- **Geometry and mesh.** Apply a support selection to see its dimensions. Cylinder length is the physical span and axial refinement preserves it. Its axis starts along world z. **Rotation axis x/y/z** and **Cylinder rotation angle** specify a right-handed rotation in degrees. Set the three components and angle before **Apply**. The Halbach template uses (1, 0, 0) and +90°, placing its bore along world y. Biplanar controls include plate width, height, full separation, world normal and center. A conical support has separate end radii. For other generated supports, the form identifies their parameter block in **Advanced**. For an STL, expand **Import a support STL** and set **STL scale to metres** *before* choosing the file. Use `0.001` for millimetres or `1` for metres. Choosing the STL stages the import. **Apply** stores the mesh in metres. **Cancel** keeps the existing support. Import a replacement STL to change that shape. Target and B0 directions use world coordinates.
- **Field, conductor and optimization.** Selecting a conductor cross-section preset stages its dimensions until you click **Apply**. Check the target expression or imported target, ROI, field component, B0, copper dimensions, contour count and solve tolerance against your intended study. Halbach gallery models use a bore along world y and a 50 mT uniform B0 along world z. An edit to the MRI reference also updates a matching uniform mechanical background. When **Readiness** shows different mechanical and MRI vectors, inspect `research_physics.b0_T` in **Advanced…**. Set it to the intended uniform load case. A deliberately distinct or per-face mechanical field remains an independent model input.
- **Advanced settings.** Edit configuration and study JSON when the form does not expose the required parameter. **Apply** validates and commits the JSON edits. **Cancel** discards them. Physical SI inputs, imported assets and configured study limits still apply. The desktop guide covers dedicated field-analysis and engineering dialogs. [Native workbench](NATIVE_WORKBENCH.md)
- **Winding and routing.** Gallery examples use **Surface B-spline (mesh scale)** on closed contours before routing: mesh gain 0.1, maximum displacement 0.5 mm and **Guard inter-turn spacing during smoothing** enabled. **Try local corner blends if a fit is rejected** rounds eligible corners within both limits. If no fit passes, the extracted loop is kept. To compare with untreated paths on the same mesh, choose **Original contours**, **Apply** and compute at the same stage. Each run keeps its own inputs; compare the fields at the recorded original contour current. **Resample without smoothing** changes sampling without rounding corners.

For an assembly, choose **Editing channel** and expand **Manage channels** to set **Include channel in batch** for each intended channel. **Duplicate channel** creates a separately owned design. The toolbar runs the active channel. The rows in **Design plan** show each channel's applied study. **Compute enabled channels** sets every included channel to the selected stage and submits those channels independently.

After a batch, choose **View result** beside a completed job in **Design plan → Jobs**. Opening that job verifies its saved evidence. The **Viewing** line identifies its channel and revision, while **Editing** continues to identify the inputs you can change.

If a batch would repeat an active channel at the same revision and stage, the button explains the conflict. You can still run another idle channel from the toolbar. **Cancel** changes to **Stopping…** while the worker acknowledges the request. If an attempt fails or stops without a result, the workbench identifies the job and explains how to retry. A completed run with **Engineering checks failed** has a result you can inspect. Open the failed checks to see the measured values and limits.

## Choose a computation boundary

| Compute to | Evidence to inspect |
|---|---|
| **Prepare model** | Support mesh, world orientation, target samples and region. |
| **Optimize stream function** | Stream function, surface current density, predicted magnetic field and independent field validation where available. |
| **Generate contours** | Discrete loops and contour levels on each support. |
| **Connect / sweep conductors** | Routes and any swept geometry requested by winding settings. |
| **Full design evaluation** | Field of the realized winding, configured independent checks, and fabrication review when enabled. |

A completed job has reached its requested boundary. For a partial run, later checks can be unperformed. For full evaluation, compare the numerical error and its configured limit. Even if the stream function field passes validation, the winding can miss its field budget, intersect another turn or violate the bend radius. Read the recorded failure reasons and inspect the appropriate saved field or geometry plot. **No construction release recorded** applies even when the modeled checks pass. [Result interpretation](RESULT_INTERPRETATION.md)

**Engineering review incomplete** means at least one verdict is missing and no check recorded a failure. Open **Engineering checks**, then expand **Strict fabrication review** or **Pipeline conductor sweep** to see individual findings and recorded measurements. Geometric distances are shown in mm. When an old result lacks a B0 direction, choose a Cartesian component to inspect its field.

### Find and export a saved run

To continue on the same hub after a restart, choose **Open project…**, select a filename under **Saved in this workspace** and click **Reopen saved project**. This reads the verified model and results directly from the workspace. To bring in a file from your computer, use the file picker in the same dialog. If a project is already open, go through **File**.

**Open project** accepts a self-contained compact `.coilgen` bundle or project JSON. The limit is 512 MiB per upload and 2 GiB of expanded numerical data. The indicator measures bytes sent to your workspace. The workbench then uses a single-use ticket, valid for one hour in that workspace, to check the project and its evidence before replacing the active design. For a legacy project with separate asset files, open it with those assets in the native workbench and save a self-contained copy first.

Under **File → Save / download**, **Save project** writes to this hub's workspace. **Download project** copies that file to the browser's download destination. Choose **Save JSON** and then **Download project** for inline JSON. Binary evidence generally makes that file larger than the compact bundle. Advanced settings in the browser accept built-in supports, project-owned assets and numerical field expressions. The workspace supplies its own output directories.

Choose **Model Builder → Results and provenance → Completed runs**. The **Completed run** list normally shows results for the active channel. Select **Show runs from other projects and channels** to include the rest of the workspace history. With no project open, choose **Review workspace runs** on the welcome screen. Select a run by design name, revision, stage and job ID. The workbench identifies the result while it loads.

Use **Model** above the plot to switch to the editable design. **Coil results** returns to the saved run. Saved field plots use the recorded run's B0 direction for projection. **Component** appears on field plots. The available plots depend on which stages and operations the run completed.

When you save while reviewing an earlier run of the editing channel, the project remembers that selection. Reopening it in the browser restores the same verified result, including selections saved by the desktop. Without a saved selection, the browser selects a run matching the editing revision when one exists. You can then choose any earlier result explicitly.

### Plot shielding at the magnet boundary

Load **Browse examples… → Coil systems · Actively shielded XYZ foil gradient system (Gx, Gy, Gz)** and
choose **Compute enabled channels**. Open **View result** for a completed channel and click **Shielding surface**.
The example evaluates each gradient on a common closed observation cylinder outside all six windings.
The same control appears for other runs that contain field vectors on saved suppression samples.

Use **Field source** to compare fields from the stream function, contours and routed winding at the recorded excitation.
The common scale shows how much of the primary field the active shield cancels. Choose **Resolve the net
field on its own scale** to see the spatial pattern of the remaining leakage. **B · n** shows its signed
normal component when surface normals are available. The RMS ratio is the net RMS divided by the
primary-only RMS for that channel and component, using identical samples and weights.
The detail view marks the peak sample in world coordinates. **Complete electrical
paths** is available when closed paths, original currents and a colocated net field
are recorded in the fabrication review.
Select a **triangle-interior check** under **Field source** to reevaluate the stream function field
or wire field at three interior points per saved triangle. The plot reports their
area-weighted RMS. This checks spatial sampling independently of the design
vertices and uses each facet's normal for **B · n**. Closing the dialog cancels
an active calculation.
For an archived run without separate contributions, the plot shows the saved net field. The maps describe
the static excitation applied to the observed structure. An eddy-current or force prediction also needs
the conducting structure and drive waveform.

**Run design choices** shows the selected run’s original request in a read-only window. Check the physical dimensions and smoothing policy alongside **Recorded smoothing outcome**, including the accepted and rejected fits and their largest applied displacement on each support. **Full submitted request (JSON)** contains every recorded configuration and study field at its stored precision, including any engineering limits, scene or array study. Large requests provide a JSON download in the same window. Editing the current model keeps historical choices intact. Partial and older results may lack smoothing outcomes. If the original request is missing or fails verification, the window explains why it cannot display it.

The cards above the result plot identify the field and sample set used for each error. For a reviewed conductor, check the design DSV error and the recorded current policy. For a stream function solution, check the design-sample and independent-sample errors when available. **Field map** opens the source of a displayed metric when the matching data is available. **Coil stats** gives the recorded quantities with their units. Open **Engineering checks** to see the affected bodies, measured values and limits.

When a full run has a reviewed wire, its first result plot shows **Review conductor centerlines**, including cuts, connections and mounting. An outside-mounted wire opens against the solid former so the near-side turns are easy to trace. Rotate the coil to inspect the opposite side, or toggle **See through support** beside **Fit view** to see both sides at once. Other mounting modes use a transparent support. **Fit view** brings the whole model back into frame.

Under **Plot**, choose **Contours before routing** to inspect the fitted loops before cuts and connections. The contour fit count describes those loops. [X²−Y² example and winding gallery](../validation/gui_winding_quality_2026-11-03/README.md)

In **Export conductor geometry**, use the suggested STL selection or expand **Choose files and formats** for other recorded solids or 3D paths. Choose the STL/CSV length unit and click **Export selected**. Download the resulting ZIP. STEP retains its declared unit. **Include matching run project** includes the exact source revision when available. For an older run without that revision, open the matching project or clear this option for a geometry-only package. DRAFT solids require explicit acknowledgment of the failed checks. The package manifest records source identity, units, checks and artifact hashes. The saved project and CAD ZIP have separate download links.

## Choose the gallery or a script study

The gallery offers cylindrical, biplanar, Halbach and shielded gradients, shims and multi-channel coil systems. Two examples illustrate how their inputs differ from the named script studies.

- The **cylindrical XYZ gallery system** uses radii of 80, 86 and 92 mm, a 200 mm physical span and a 40 mm diameter spherical DSV at 20 mT/m. The standard cylindrical **script project** specifies diameters of 130, 132 and 134 mm, a 250 mm nominal coil length and a 50 mm DSV at 10 mT/m. [Standard cylindrical project](../projects/standard_cylindrical_3axis_gradient/README.md)
- The **Halbach gallery gradients** use a generated cylinder of 80 mm radius, 200 mm physical span and 40 mm DSV at 20 mT/m. The Halbach **script project** loads separate octagonal winding-surface STLs for Gx, Gy and Gz, and uses a 50 mm DSV at 10 mT/m. Other project scripts can also require local data. [Halbach project](../projects/halbach_cylindrical_3axis_gradient/README.md) · [Project data](PROJECT_DATA.md)

## Host a team workspace

1. On the host, start `Start CoilGen2 Browser.cmd --team` from a command prompt. In PowerShell use `& ".\Start CoilGen2 Browser.cmd" --team`. The hub prints the team URL and team password. Set `COILGEN_WEB_PASSWORD` before launch if you want to supply the team password.
2. Enter an unused **Workspace name** (1–40 letters, digits, hyphens or underscores), the **Team password**, and your own **Private workspace passphrase** (12–256 characters, different from the team password). Choose **Create workspace**. On return, enter the same workspace name and both credentials, then **Open existing workspace**.
3. Each workspace has a separate workbench process and job queue. The hub serves pages, downloads and WebSocket traffic through one port. The default data root is `%USERPROFILE%\CoilGen2 Browser Workspaces`. Set `--data-root DIR` to choose another location. Projects, jobs, uploads and exports live under its `users` directory. `--max-users` sets the number of seats (default 8) and divides the host's native threads among them. `--idle-minutes` sets the timeout since the last connection activity (default 120). Queued, running and cancelling jobs continue after disconnection or logout. The hub checks idle workers once per minute and releases their seats when the timeout has elapsed and the queue is finished. Choose **Log out** when finished.
4. Local mode binds to loopback. Team mode defaults to all interfaces. Use a trusted network for direct HTTP. For remote access, put HTTPS in front of the hub with a reverse proxy that forwards WebSocket upgrades and downloads. Plain HTTP transmits the login credentials in clear text.

**Recover access or enroll a legacy workspace.** The host operator must stop the hub using that data root first. Run the following in the host's CoilGen2 environment, replacing the directory and name.

```powershell
python -m coilgen.web --data-root "D:\CoilGen Workspaces" --reset-workspace-password NAME
```

Enter the new private passphrase twice at the prompts. Existing workspace files remain in place. Restart the hub with the **same data root** and have the participant use **Open existing workspace**. Workspaces created before private passphrases were introduced require this operator step before their first login.

The no-login launcher owns the reserved workspace name `local`. If an older team used that name, a device name, or two names differing only by case, stop the hub and verify which files belong to each participant. Copy that participant's workspace data into a new, uniquely named directory under `users`, preserving the original directory. Run the recovery command for the new name and give its private passphrase to the owner.

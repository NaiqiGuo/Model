# Model Inspect

A VS Code sidebar for this repository's `inspect_sid.py`. Includes category-specific Build, three comparison commands and Python environment selection.

## Installation

1. In VS Code, open Extensions with **Cmd+Shift+X** (Windows/Linux: **Ctrl+Shift+X**).
2. Select **… → Install from VSIX…**.
3. Choose `vscode-inspect/model-inspect-0.1.7.vsix`.
4. Open the Model project and click **Inspect** in the Activity Bar.

If upgrading from an earlier version, install this newer VSIX and reload VS Code if prompted.
Node.js, TypeScript, and the Microsoft Python extension are not required for use.
You need VS Code 1.85 or later and a Python environment that can run `inspect_sid.py`.
The workspace must be trusted before running scripts.

## Usage

- The project is detected from the open workspace containing `inspect_sid.py`; there is
  no Project row. With multiple matching workspace folders, choose one when running.
  If none matches, a folder picker is available as a fallback.
- **Python environment**: Choose a discovered Conda environment, project `.venv` /
  `venv`, or Python on PATH. The executable path is shown beside each name.
  No environment is forced or selected automatically. Manual entry and browsing
  remain available for environments in custom locations. The choice is verified and saved.
- **Export name**: Passed directly to Build, e.g. `analysis_export` or
  `trial_01`. The current Python script determines the output layout and
  whether existing files are updated.
- **Build**: Choose errors, training, realizations or heatmaps. Only the selected category is exported.
- Errors, training and realizations use `<name>_bridge/<quantity>/<source>/`
  and `<name>_frame/<quantity>/<source>/`, without a repeated structure directory.
  Compare also accepts older exports with the extra `bridge/` or `frame/` layer.
  Existing exports are not moved; if both layouts coexist, the flat copy takes precedence
  for matching paths. Use a new export name for a clean directory tree.
- Training exports use one losslessly compressed `training/<event>.npz` per event,
  containing `dt`, `time`, `ground`, and `structure`. Original training CSVs are unchanged.
  Training comparison supports both legacy CSV exports and NPZ exports, including
  comparisons between the two formats. If both exist for an event, NPZ takes precedence.
  Use a new export name to avoid retaining old CSV copies alongside NPZ files.
- **Directory A / B**: Select the exact export folders to compare, such as
  `reference_bridge` and `analysis_export_bridge`. There is no latest-run lookup.
- **Compare errors / realizations / training**: Run the selected category.
- Each comparison asks where to save its TXT report. Cancel the save dialog to
  cancel the comparison. The sidebar passes `--silent --to-file <chosen path>`.
- Frame and bridge folders may be compared without a structure restriction.
- Environment export/comparison has been removed; Python interpreter selection remains.

Only one command runs at a time. While running, click **Running (click to stop)**
or the toolbar Stop button to cancel; interrupted exports may be incomplete.
Completion and failure notifications appear directly in VS Code. Failures include
an excerpt of the command output. There is no Run log entry or output panel, and
no log file is saved. Exit codes 0 and 1 indicate a completed comparison when stderr is empty;
Python errors are shown as failures. Report paths come from the save dialog,
not from parsing script output. The completion notification offers **Open report**.

Selections are stored in local VS Code workspace state, not in the repository or
synced by this extension. The extension runs the selected interpreter against the
current project script. Updating the Python script does not require reinstalling
the extension; updating the extension itself requires installing a newer VSIX.
There is no automatic execution, telemetry, or GitHub upload.

## Sharing through GitHub

Commit `vscode-inspect/` (including the VSIX) along with the project scripts.
Collaborators can pull, install the VSIX, and select their own Python environment.
Creating this extension does not automatically commit, push, or publish it.

## Development and packaging

Only maintainers need Node.js:

```bash
cd vscode-inspect
npm test
npm run package
```

To include the integration test, set `INSPECT_TEST_PYTHON` to a Python executable
with NumPy installed. Tests use temporary result folders, not project outputs.

The runtime is plain JavaScript with no build step or runtime npm dependencies.
Packaging downloads Microsoft's `@vscode/vsce` tool through npm.
The sidebar uses the [Tree View API](https://code.visualstudio.com/api/extension-guides/tree-view)
and the [official VSIX packaging workflow](https://code.visualstudio.com/api/working-with-extensions/publishing-extension).

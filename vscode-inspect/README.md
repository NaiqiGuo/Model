# Model Inspect

A VS Code sidebar for this repository's `inspect.py`. Includes Build, four
comparison commands and Python environment selection. No Heatmaps or Reports section.

## Installation

1. In VS Code, open Extensions with **Cmd+Shift+X** (Windows/Linux: **Ctrl+Shift+X**).
2. Select **… → Install from VSIX…**.
3. Choose `vscode-inspect/model-inspect-0.1.6.vsix`.
4. Open the Model project and click **Inspect** in the Activity Bar.

If upgrading from 0.1.0, install this newer VSIX and reload VS Code if prompted.
Node.js, TypeScript, and the Microsoft Python extension are not required for use.
You need VS Code 1.85 or later and a Python environment that can run `inspect.py`.
The workspace must be trusted before running scripts.

## Usage

- The project is detected from the open workspace containing `inspect.py`; there is
  no Project row. With multiple matching workspace folders, choose one when running.
  If none matches, a folder picker is available as a fallback.
- **Python environment**: Choose a discovered Conda environment, project `.venv` /
  `venv`, or Python on PATH. The executable path is shown beside each name.
  No environment is forced or selected automatically. Manual entry and browsing
  remain available for environments in custom locations. The choice is verified and saved.
- **Export name**: Passed directly to Build, e.g. `analysis_export` or
  `trial_01`. The current Python script determines the output layout and
  whether existing files are updated.
- **Build**: Export all three categories and environment information.
- **Directory A / B**: Select the exact export folders to compare, such as
  `reference_bridge` and `analysis_export_bridge`. There is no latest-run lookup.
- **Compare errors / realizations / training**: Run the selected category.
- **Compare environment**: Set A/B to the two `_environment` folders. Compares
  Python version, platform and package versions; paths and capture times are
  informational. Writes `compare_environment.txt` beside B. These are export-time
  snapshots, not proof of the original computation environment or identical BLAS builds.
To compare frame data, select the corresponding frame folders. Generated reports
remain on disk; the sidebar does not include report links.

Only one command runs at a time. While running, click **Running (click to stop)**
or the toolbar Stop button to cancel; interrupted exports may be incomplete.
Completion and failure notifications appear directly in VS Code. Failures include
an excerpt of the command output. There is no Run log entry or output panel, and
no log file is saved. Exit code 1 is treated as differences or missing data only
when the script successfully reports a saved comparison file. Report paths are
read from script output rather than inferred.

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

'use strict';

const vscode = require('vscode');
const fs = require('node:fs');
const path = require('node:path');
const os = require('node:os');
const { spawn, execFile } = require('node:child_process');
const { promisify } = require('node:util');
const execute = promisify(execFile);
const categories = ['errors', 'realizations', 'training'];

function expandHome(value) {
  return value === '~' ? os.homedir() : value.startsWith('~/') ? path.join(os.homedir(), value.slice(2)) : value;
}

async function discoverEnvironments(root, selected) {
  const candidates = new Map();
  const add = (executable, label) => {
    if (executable && fs.existsSync(executable) && !candidates.has(executable))
      candidates.set(executable, { label, description: executable, executable });
  };
  const addEnv = (prefix, label) => add(path.join(prefix, process.platform === 'win32' ? 'python.exe' : 'bin/python'), label);
  if (selected) add(selected, 'Current selection');
  if (root) for (const name of ['.venv', 'venv']) addEnv(path.join(root, name), name);
  for (const key of ['CONDA_PREFIX', 'VIRTUAL_ENV']) if (process.env[key]) addEnv(process.env[key], path.basename(process.env[key]));
  try {
    const registered = fs.readFileSync(path.join(os.homedir(), '.conda/environments.txt'), 'utf8');
    for (const prefix of registered.split(/\r?\n/).filter(Boolean)) addEnv(prefix, path.basename(prefix));
  } catch { /* Conda is optional. */ }
  for (const name of ['miniforge3', 'mambaforge', 'miniconda3', 'anaconda3']) {
    const base = path.join(os.homedir(), name);
    addEnv(base, `${name} (base)`);
    try {
      for (const entry of fs.readdirSync(path.join(base, 'envs'), { withFileTypes: true })) {
        if (entry.isDirectory()) addEnv(path.join(base, 'envs', entry.name), entry.name);
      }
    } catch { /* Distribution not installed. */ }
  }
  try {
    const { stdout } = await execute(process.env.CONDA_EXE || 'conda', ['env', 'list', '--json'], { timeout: 5000 });
    for (const prefix of JSON.parse(stdout).envs || []) addEnv(prefix, path.basename(prefix));
  } catch { /* Conda may not be on the editor PATH. */ }
  for (const dir of (process.env.PATH || '').split(path.delimiter).filter(Boolean)) {
    for (const name of process.platform === 'win32' ? ['python.exe'] : ['python3', 'python']) add(path.join(dir, name), `${name} (PATH)`);
  }
  return [...candidates.values()];
}

function argumentsFor(root, kind, state) {
  if (kind === 'build') return ['-u', path.join(root, 'inspect.py'), 'build', state.prefix];
  if (!categories.includes(kind)) throw new Error('Unknown comparison category');
  return ['-u', path.join(root, 'inspect.py'), 'compare', kind, state.a, state.b];
}

function completion(kind, code, output, root) {
  const match = output.match(/^Comparison saved to (.+)\r?$/m);
  const report = match ? path.resolve(root, match[1].trim()) : undefined;
  if (kind === 'build' && code === 0) return { success: true, message: 'Build completed.' };
  if (kind !== 'build' && report && (code === 0 || code === 1)) {
    return { success: true, report, message: code === 0 ? `${kind}: match within tolerance.` : `${kind}: differences or missing data found.` };
  }
  const detail = output.trim().split(/\r?\n/).filter(Boolean).slice(-3).join('\n').slice(-1200);
  return { success: false, message: `Command failed (exit ${code}).${detail ? `\n${detail}` : ''}` };
}

class InspectController {
  constructor(context) {
    this.context = context;
    this.state = context.workspaceState.get('inspectState', { prefix: 'analysis_export', reports: {} });
    if (this.state.prefix === 'naiqi') this.state.prefix = 'analysis_export';
    this.state.reports ||= {};
    this.changed = new vscode.EventEmitter();
    this.onDidChangeTreeData = this.changed.event;
    context.subscriptions.push(this.changed);
  }

  async save() {
    await this.context.workspaceState.update('inspectState', this.state);
    this.changed.fire();
  }

  row(label, command, description, icon = 'chevron-right') {
    const item = new vscode.TreeItem(label);
    item.command = { command: `modelInspect.${command}`, title: label };
    item.description = description;
    item.tooltip = description ? `${label}: ${description}` : label;
    item.iconPath = new vscode.ThemeIcon(icon);
    return item;
  }

  getTreeItem(item) { return item; }

  getChildren(item) {
    if (item) return item.children || [];
    const group = (label, children) => {
      const parent = new vscode.TreeItem(label, vscode.TreeItemCollapsibleState.Expanded);
      parent.children = children;
      return parent;
    };
    const s = this.state;
    return [
      this.row('Python environment', 'python', s.python ? path.basename(path.dirname(process.platform === 'win32' ? s.python : path.dirname(s.python))) : 'Select an environment', 'terminal'),
      group('Export', [
        this.row('Export name', 'prefix', s.prefix, 'edit'),
        this.row('Build', 'build', 'Export all data and environment information', 'play')
      ]),
      group('Compare', [
        this.row('Directory A', 'selectA', s.a || 'Not selected', 'folder'),
        this.row('Directory B', 'selectB', s.b || 'Not selected', 'folder'),
        ...categories.map(k => this.row(`Compare ${k}`, k, undefined, 'diff'))
      ]),
      ...(this.child ? [this.row('Running (click to stop)', 'stop', this.runningKind, 'debug-stop')] : [])
    ];
  }

  async folder(title, current) {
    const selected = await vscode.window.showOpenDialog({
      title, canSelectFiles: false, canSelectFolders: true, canSelectMany: false,
      defaultUri: current ? vscode.Uri.file(current) : vscode.workspace.workspaceFolders?.[0]?.uri
    });
    return selected?.[0]?.fsPath;
  }

  async project() {
    const root = await this.folder('Select the project folder containing inspect.py', this.state.root);
    if (!root) return;
    if (!fs.existsSync(path.join(root, 'inspect.py'))) throw new Error('The selected folder does not contain inspect.py. Select the Model project folder.');
    this.state = { ...this.state, root, a: undefined, b: undefined, reports: {} };
    await this.save();
  }

  async ensureRoot() {
    const roots = (vscode.workspace.workspaceFolders || []).map(f => f.uri.fsPath).filter(p => fs.existsSync(path.join(p, 'inspect.py')));
    if (roots.length === 1) {
      if (this.state.root !== roots[0]) {
        this.state = { ...this.state, root: roots[0], reports: {} };
        await this.save();
      }
    } else if (roots.length > 1) {
      const choice = await vscode.window.showQuickPick(roots.map(root => ({ label: path.basename(root), description: root })), { title: 'Select the workspace containing inspect.py' });
      if (!choice) return;
      this.state.root = choice.description;
      await this.save();
    } else if (!this.state.root || !fs.existsSync(path.join(this.state.root, 'inspect.py'))) {
      await this.project();
    }
    return this.state.root;
  }

  async python() {
    const root = this.state.root || vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
    const environments = await discoverEnvironments(root, this.state.python);
    const selection = await vscode.window.showQuickPick([
      ...environments,
      { label: 'Enter a Python path or command', mode: 'input', description: 'For example: /.../envs/mssp/bin/python' },
      { label: 'Browse for a Python executable', mode: 'browse' }
    ], { title: 'Inspect: Select Python Environment', matchOnDescription: true, placeHolder: 'Choose an environment; its Python will run Inspect' });
    if (!selection) return;
    let candidate;
    if (selection.executable) {
      candidate = selection.executable;
    } else if (selection.mode === 'browse') {
      candidate = (await vscode.window.showOpenDialog({ canSelectMany: false, canSelectFiles: true, canSelectFolders: false, title: 'Select python / python.exe' }))?.[0]?.fsPath;
    } else {
      candidate = await vscode.window.showInputBox({ title: 'Python path', value: this.state.python || (process.platform === 'win32' ? 'python' : 'python3'), prompt: 'Enter the executable path without quotes or arguments. For Conda use envs/<name>/bin/python (python.exe on Windows).', ignoreFocusOut: true });
    }
    if (!candidate?.trim()) return;
    candidate = expandHome(candidate.trim());
    const { stdout } = await execute(candidate, ['-c', 'import sys; print(sys.executable)'], { timeout: 15000 });
    const executable = stdout.trim();
    if (!path.isAbsolute(executable) || !fs.existsSync(executable)) throw new Error('Could not verify the Python executable path.');
    this.state.python = executable;
    await this.save();
    vscode.window.showInformationMessage(`Inspect Python: ${executable}`);
  }

  async prefix() {
    const prefix = await vscode.window.showInputBox({ title: 'Export name', value: this.state.prefix, prompt: 'Choose a name for your export, for example analysis_export or trial_01.', ignoreFocusOut: true });
    if (prefix?.trim()) { this.state.prefix = expandHome(prefix.trim()); await this.save(); }
  }

  async select(side) {
    const selected = await this.folder(`Select comparison directory ${side.toUpperCase()}`, this.state[side] || this.state.root);
    if (selected) {
      this.state[side] = selected;
      this.state.reports = {};
      await this.save();
    }
  }

  stop() {
    if (this.child) { this.cancelled = true; this.child.kill(); }
  }

  async run(kind) {
    if (!vscode.workspace.isTrusted) throw new Error('Trust this workspace before running Python scripts.');
    if (this.child || this.preparing) { vscode.window.showWarningMessage('An Inspect command is already running or being configured.'); return; }
    this.preparing = true;
    try {
      const root = await this.ensureRoot();
      if (!root) return;
      if (!this.state.python) await this.python();
      if (!this.state.python) return;
      if (kind !== 'build') {
        if (!this.state.a) await this.select('a');
        if (!this.state.b) await this.select('b');
        if (!this.state.a || !this.state.b) return;
        for (const p of [this.state.a, this.state.b]) if (!fs.statSync(p).isDirectory()) throw new Error(`Not a directory: ${p}`);
        delete this.state.reports[kind];
        await this.save();
      }
      const args = argumentsFor(root, kind, this.state);
      this.cancelled = false;
      this.runningKind = kind;
      // No shell interpolation: spaces and shell characters stay literal arguments.
      const child = spawn(this.state.python, args, { cwd: root, shell: false, env: { ...process.env, PYTHONUNBUFFERED: '1' }, windowsHide: true });
      this.child = child;
      void vscode.commands.executeCommand('setContext', 'modelInspect.running', true);
      this.changed.fire();
      let tail = '';
      let failure;
      const record = chunk => { const text = chunk.toString(); tail = (tail + text).slice(-2 * 1024 * 1024); };
      child.stdout.on('data', record);
      child.stderr.on('data', record);
      child.on('error', err => { failure = err; });
      await new Promise(resolve => child.on('close', async code => {
        try {
          if (this.cancelled) { vscode.window.showInformationMessage('Stopped. Outputs may be incomplete.'); return; }
          const result = failure ? { success: false, message: failure.message } : completion(kind, code, tail, root);
          if (result.report && fs.existsSync(result.report)) {
            this.state.reports[kind] = result.report;
            await this.save();
          }
          if (!result.success) vscode.window.showErrorMessage(result.message);
          else {
            void vscode.window.showInformationMessage(result.message, ...(result.report ? ['Open report'] : [])).then(async action => {
              if (action === 'Open report') await vscode.window.showTextDocument(await vscode.workspace.openTextDocument(result.report));
            }).catch(err => vscode.window.showErrorMessage(String(err.message || err)));
          }
        } catch (err) { vscode.window.showErrorMessage(String(err.message || err)); }
        finally { resolve(); }
      }));
    } finally {
      this.child = undefined;
      this.preparing = false;
      this.runningKind = undefined;
      await vscode.commands.executeCommand('setContext', 'modelInspect.running', false);
      this.changed.fire();
    }
  }
}

function activate(context) {
  const control = new InspectController(context);
  context.subscriptions.push(vscode.window.registerTreeDataProvider('modelInspect.controls', control));
  const actions = {
    project: () => control.project(), python: () => control.python(), prefix: () => control.prefix(),
    selectA: () => control.select('a'), selectB: () => control.select('b'), build: () => control.run('build'),
    stop: () => control.stop()
  };
  for (const kind of categories) {
    actions[kind] = () => control.run(kind);
  }
  for (const [name, callback] of Object.entries(actions)) {
    context.subscriptions.push(vscode.commands.registerCommand(`modelInspect.${name}`, async () => {
      if (!vscode.workspace.isTrusted) { vscode.window.showWarningMessage('Trust this workspace first.'); return; }
      if (control.preparing && name !== 'stop') { vscode.window.showWarningMessage('Wait for the current command to finish, or click Stop.'); return; }
      try { await callback(); } catch (err) { vscode.window.showErrorMessage(`Inspect: ${err.message || err}`); }
    }));
  }
  context.subscriptions.push({ dispose: () => control.stop() });
}

module.exports = { activate, argumentsFor, completion, discoverEnvironments };

'use strict';
const assert = require('node:assert/strict');
const test = require('node:test');
const Module = require('node:module');
const path = require('node:path');
const fs = require('node:fs');
const os = require('node:os');
const { spawnSync } = require('node:child_process');
// Unit-test command construction without needing an Extension Host.
const original = Module._load;
Module._load = function (name, ...rest) { return name === 'vscode' ? {} : original.call(this, name, ...rest); };
const { argumentsFor, completion } = require('./extension');
Module._load = original;

test('build and all comparison categories preserve exact paths', () => {
  const root = path.join(os.tmpdir(), 'Model with spaces');
  const state = { prefix: 'runs/my run/naiqi', a: '/tmp/Chrystal A', b: '/tmp/Naiqi B' };
  assert.deepEqual(argumentsFor(root, 'build', state), ['-u', path.join(root, 'inspect.py'), 'build', state.prefix]);
  for (const kind of ['errors', 'realizations', 'training', 'environment']) {
    assert.deepEqual(argumentsFor(root, kind, state), ['-u', path.join(root, 'inspect.py'), 'compare', kind, state.a, state.b]);
  }
  assert.throws(() => argumentsFor(root, 'heatmaps', state));
});

test('differences are normal results; traceback/missing data path errors are failures', () => {
  const output = 'real differences: 3\nComparison saved to runs/my run/compare_bridge_errors.txt\n';
  assert.equal(completion('errors', 1, output, '/project').success, true);
  assert.equal(completion('errors', 0, output, '/project').success, true);
  const failure = completion('errors', 1, 'Traceback: failed', '/project');
  assert.equal(failure.success, false);
  assert.match(failure.message, /Traceback: failed/);
  assert.equal(completion('errors', 2, output, '/project').success, false);
  assert.equal(completion('build', 1, output, '/project').success, false);
  assert.equal(completion('build', 0, '', '/project').success, true);
});

test('arguments with shell characters reach the process unchanged', () => {
  const special = 'dir with spaces; $(echo SHOULD_NOT_EXECUTE)';
  const result = spawnSync(process.execPath, ['-e', 'process.stdout.write(JSON.stringify(process.argv.slice(1)))', special], { shell: false, encoding: 'utf8' });
  assert.deepEqual(JSON.parse(result.stdout), [special]);
});

test('manifest has only requested actions and all runtime files', () => {
  const pkg = require('./package.json');
  assert.equal(pkg.contributes.commands.some(c => /heatmap/i.test(c.command)), false);
  assert.equal(pkg.contributes.commands.length, 11);
  assert.ok(fs.existsSync(path.join(__dirname, pkg.main)));
  assert.ok(fs.existsSync(path.join(__dirname, pkg.contributes.viewsContainers.activitybar[0].icon)));
});

test('sidebar runs real inspect comparisons and records the generated report', { skip: !process.env.INSPECT_TEST_PYTHON }, async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'inspect sidebar '));
  const commands = new Map();
  const errors = [];
  const messages = [];
  const opened = [];
  let state;
  const folderSelections = [];
  const mock = {
    EventEmitter: class { event = () => {}; fire() {} dispose() {} },
    TreeItem: class { constructor(label) { this.label = label; } },
    ThemeIcon: class {}, TreeItemCollapsibleState: { Expanded: 2 },
    Uri: { file: p => ({ fsPath: p }) },
    workspace: {
      isTrusted: true, workspaceFolders: [{ uri: { fsPath: root } }],
      openTextDocument: async uri => typeof uri === 'string' ? uri : uri.fsPath
    },
    window: {
      registerTreeDataProvider: () => ({ dispose() {} }),
      showQuickPick: async () => ({ mode: 'input' }),
      showInputBox: async () => process.env.INSPECT_TEST_PYTHON,
      showOpenDialog: async () => [{ fsPath: folderSelections.shift() }],
      showInformationMessage: async message => { messages.push(message); },
      showWarningMessage: async message => { errors.push(message); },
      showErrorMessage: async message => { errors.push(message); },
      showTextDocument: async document => { opened.push(document); }
    },
    commands: {
      executeCommand: async () => {},
      registerCommand: (name, callback) => { commands.set(name, callback); return { dispose() {} }; }
    }
  };
  try {
    fs.copyFileSync(path.join(__dirname, '..', 'inspect.py'), path.join(root, 'inspect.py'));
    const a = path.join(root, 'chrystal_bridge');
    const b = path.join(root, 'run with spaces', 'naiqi_bridge');
    for (const p of [a, b]) {
      const base = path.join(p, 'bridge', 'acceleration', 'field');
      fs.mkdirSync(path.join(base, 'systems', '1'), { recursive: true });
      fs.mkdirSync(path.join(base, 'training', 'ground'), { recursive: true });
      fs.writeFileSync(path.join(base, 'error.csv'), '0.1\n');
      fs.writeFileSync(path.join(base, 'systems', '1', 'A.csv'), p === a ? '1\n' : '2\n');
      fs.writeFileSync(path.join(base, 'training', 'ground', '1.csv'), '1 2\n');
    }
    delete require.cache[require.resolve('./extension')];
    Module._load = function (name, ...rest) { return name === 'vscode' ? mock : original.call(this, name, ...rest); };
    const extension = require('./extension');
    Module._load = original;
    extension.activate({ subscriptions: [], workspaceState: { get: (_key, fallback) => fallback, update: async (_key, value) => { state = value; } } });
    folderSelections.push(a, b);
    await commands.get('modelInspect.selectA')();
    await commands.get('modelInspect.selectB')();
    for (const kind of ['errors', 'realizations', 'training']) await commands.get(`modelInspect.${kind}`)();
    assert.deepEqual(errors, []);
    assert.ok(messages.some(s => s.includes('realizations: differences')));
    assert.ok(messages.some(s => s.includes('errors: match')));
    for (const kind of ['errors', 'realizations', 'training']) assert.ok(fs.existsSync(state.reports[kind]));
    assert.equal(commands.has('modelInspect.openErrors'), false);
    assert.equal(commands.has('modelInspect.output'), false);
    const envA = path.join(root, 'reference_environment');
    const envB = path.join(root, 'analysis_export_environment');
    for (const env of [envA, envB]) {
      fs.mkdirSync(env);
      fs.writeFileSync(path.join(env, 'environment.json'), JSON.stringify({ python_version: '3.12.0', platform: 'test-platform', python_executable: env, captured_at_utc: env }));
      fs.writeFileSync(path.join(env, 'packages.txt'), 'numpy==2.2.6\n');
    }
    folderSelections.push(envA, envB);
    await commands.get('modelInspect.selectA')();
    await commands.get('modelInspect.selectB')();
    await commands.get('modelInspect.environment')();
    assert.deepEqual(errors, []);
    assert.ok(messages.some(s => s.includes('environment: recorded Python version')));
    assert.equal(state.reports.environment, path.join(root, 'compare_environment.txt'));
    fs.writeFileSync(path.join(envB, 'packages.txt'), 'numpy==2.1.0\nscipy==1.14.0\n');
    await commands.get('modelInspect.environment')();
    assert.ok(messages.some(s => s.includes('environment: differences')));
    const report = fs.readFileSync(state.reports.environment, 'utf8');
    assert.match(report, /numpy: A=2.2.6, B=2.1.0/);
    assert.match(report, /only in B: scipy/);
    fs.unlinkSync(path.join(envB, 'packages.txt'));
    await commands.get('modelInspect.environment')();
    assert.match(fs.readFileSync(state.reports.environment, 'utf8'), /environment information not provided/);
    assert.deepEqual(errors, []);

  } finally {
    Module._load = original;
    fs.rmSync(root, { recursive: true, force: true });
  }
});

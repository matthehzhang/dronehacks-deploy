"""Exercise the copied ROS package payload without a checkout or ROS process."""
import ast
import os
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def test_copied_install_plans_with_its_bundled_new_map(tmp_path):
    result = subprocess.run([
        sys.executable, 'setup.py', 'build', '--build-base', str(tmp_path / 'build'),
        'bdist_wheel', '--bdist-dir', str(tmp_path / 'bdist'),
        '--dist-dir', str(tmp_path / 'dist'),
    ], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    wheel = next((tmp_path / 'dist').glob('*.whl'))
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(tmp_path / 'installed')
    share = next((tmp_path / 'installed').glob('*.data/data/share/kestrel_nav'))
    launch = share / 'launch/refnav_skybeta.launch.py'
    # Evaluate the actual ROOT selection with imports removed (ROS is not needed
    # for path selection). This catches source-only assumptions in installed launch.
    tree = ast.parse(launch.read_text())
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'ROOT' for t in node.targets):
            nodes.append(node)
        elif isinstance(node, ast.If) and 'ROOT' in ast.unparse(node.test):
            nodes.append(node)
    namespace = {'Path': Path, '__file__': str(launch)}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(launch), 'exec'), namespace)
    runtime = namespace['ROOT']
    assert runtime == share / 'runtime'
    for name in ('refnav/node.py', 'coursenav/loc_node.py', 'onboard/config/common.yaml',
                 'maps/course_new_origin.npz', 'course/new_course_origin.obj', 'flight.yaml'):
        assert (runtime / name).is_file(), name
    env = dict(os.environ)
    env.pop('COURSENAV_COURSE', None)
    env['PYTHONPATH'] = str(runtime)
    result = subprocess.run([
        sys.executable, str(runtime / 'fly_course.py'), '--dry-run', '--skip-preflight',
        '--event-start', '--start-from-mark', '--no-return', '--landed-z', '0.10',
    ], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'navigation: path mode' in result.stdout
    assert 'predicted false blocks []' in result.stdout

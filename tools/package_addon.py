"""Build the installable add-on ZIP from the reviewable source tree."""
from pathlib import Path
import ast
import zipfile

root = Path(__file__).resolve().parents[1]
package = root / 'HP_SimpleModelingTools'
info_node = next(n for n in ast.parse((package / '__init__.py').read_text()).body
                 if isinstance(n, ast.Assign) and any(isinstance(t,ast.Name) and t.id=='bl_info' for t in n.targets))
version = '.'.join(map(str,ast.literal_eval(info_node.value)['version']))
output = root / f'HP_SimpleModelingTools_PathPen_v{version}_SectionWindow.zip'
files = sorted(list(package.glob('*.py')) + [package / 'README.txt'])
with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
    for path in files:
        info = zipfile.ZipInfo(str(path.relative_to(root)), date_time=(2026,10,9,0,0,0))
        info.external_attr = 0o644 << 16
        info.compress_type = zipfile.ZIP_DEFLATED
        archive.writestr(info,path.read_bytes())
with zipfile.ZipFile(output) as archive:
    assert archive.testzip() is None
    for path in files:
        assert archive.read(str(path.relative_to(root))) == path.read_bytes()
print(output.name)

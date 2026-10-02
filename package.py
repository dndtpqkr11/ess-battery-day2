"""Create a local upload-ready archive; performs no upload or submission."""
from pathlib import Path
import zipfile,re

HERE=Path(__file__).resolve().parent

def run():
    docs=[HERE/name for name in ['README.md','FOLLOWUP.md','IMPROVEMENT.md','VALIDATION.md','requirements.txt','.gitignore']]
    files=list(HERE.glob('*.py'))+docs
    for name in ['data','models','results','figures']:
        files.extend(p for p in (HERE/name).rglob('*') if p.is_file())
    for md in [p for p in files if p.suffix=='.md']:
        for target in re.findall(r'\]\(([^)]+)\)',md.read_text()):
            if not target.startswith(('http://','https://')) and not (md.parent/target).is_file():
                raise ValueError(f'Broken link: {md.name}: {target}')
    out=HERE/'output/DS-MINI-Day2-울산_4반-박세웅.zip';out.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(set(files)):z.write(p,p.relative_to(HERE))
    with zipfile.ZipFile(out) as z:
        if z.testzip() is not None:raise ValueError('Corrupted archive')
        assert 'models/followup_model.joblib' in z.namelist()
        assert 'models/improved_model.joblib' in z.namelist()
        assert not any(x.endswith('.mat') or 'snapshot' in x for x in z.namelist())
    print(f'{len(set(files))} files, {out.stat().st_size} bytes -> {out}')

if __name__=='__main__':run()

"""Build web/app/index.html from the explorer template: empty sequence list,
meta (bin centres) baked in from the relspec constants, API loader appended.
Run at image build; also runnable directly for local dev."""
import json, pathlib, sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]                      # .../relspec
sys.path.insert(0, str(ROOT / 'src'))
from relspec.pipeline import CENTERS, ENV_CENTERS  # noqa: E402

LOADER_TAG = '\n<scr' + 'ipt src="app.js"></scr' + 'ipt>\n'

def build():
    tpl = (ROOT / 'viz' / 'explorer_template.html').read_text()
    data = dict(meta=dict(centers=[round(float(c), 4) for c in CENTERS],
                          env_centers=[round(float(c), 4) for c in ENV_CENTERS],
                          q_db=0.5, db_off=-128.0),
                sequences=[])
    html = tpl.replace('__DATA_JSON__', json.dumps(data, separators=(',', ':')))
    html += LOADER_TAG
    out = HERE / 'app' / 'index.html'
    out.write_text(html)
    print(f'wrote {out} ({len(html)//1024} KiB)')

    # ship the CMMS dashboard alongside the explorer; served under /app it
    # auto-detects same-origin API mode and offers the passphrase connect
    dash = ROOT / 'viz' / 'dashboard_mockup.html'
    if dash.exists():
        dst = HERE / 'app' / 'dashboard.html'
        dst.write_text(dash.read_text())
        print(f'wrote {dst} ({dst.stat().st_size//1024} KiB)')

if __name__ == '__main__':
    build()

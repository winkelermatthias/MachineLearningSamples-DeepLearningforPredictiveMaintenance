#!/usr/bin/env python3
"""Build viz/dashboard_mockup.html from dashboard_mockup_src.html by
inlining the brand fonts (Archivo for display/UI, IBM Plex Mono for data)
as base64 @font-face rules in place of the __FONTS_CSS__ placeholder.

Font blobs come from a fonts_b64.json produced by base64-encoding the
latin woff2 files of the @fontsource/archivo and @fontsource/ibm-plex-mono
npm packages (OFL-licensed). Pass its path as argv[1], or drop the file
next to this script.
"""
import json, pathlib, sys

HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE / 'dashboard_mockup_src.html'
OUT = HERE / 'dashboard_mockup.html'

FACES = [
    ('Space Grotesk', 400, 'space-grotesk-400'),
    ('Space Grotesk', 500, 'space-grotesk-500'),
    ('Space Grotesk', 700, 'space-grotesk-700'),
    ('IBM Plex Mono', 400, 'ibm-plex-mono-400'),
    ('IBM Plex Mono', 500, 'ibm-plex-mono-500'),
    ('IBM Plex Mono', 600, 'ibm-plex-mono-600'),
]


def fonts_css(blobs: dict) -> str:
    rules = []
    for family, weight, key in FACES:
        b64 = blobs[key]
        rules.append(
            "@font-face{font-family:'%s';font-style:normal;"
            "font-weight:%d;font-display:swap;"
            "src:url(data:font/woff2;base64,%s) format('woff2')}"
            % (family, weight, b64))
    return '\n'.join(rules)


def main() -> None:
    blob_path = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 \
        else HERE / 'fonts_b64.json'
    blobs = json.loads(blob_path.read_text())
    html = SRC.read_text()
    assert '__FONTS_CSS__' in html, 'placeholder missing from src'
    html = html.replace('__FONTS_CSS__', fonts_css(blobs), 1)
    OUT.write_text(html)
    print(f'wrote {OUT} ({len(html)/1024:.0f} KB, {len(FACES)} font faces)')


if __name__ == '__main__':
    main()

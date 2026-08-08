# Real datasets used by the v2 evaluation

All three are fetched from public GitHub mirrors with plain `git clone`; no
credentials, no registration walls, no manual downloads. The loaders in
`src/relspec/datasets.py` read the mirrors in place.

```bash
git clone --depth 1 https://github.com/s-whynot/cwru-dataset /workspace/s-whynot/cwru-dataset
git clone --depth 1 https://github.com/mathworks/rollingelementbearingfaultdiagnosis-data \
    /workspace/mathworks/rollingelementbearingfaultdiagnosis-data
git clone --depth 1 https://github.com/cathysiyu/mechanical-datasets /workspace/cathysiyu/mechanical-datasets
```

Override locations with `CWRU_ROOT`, `MFPT_ROOT`, `SEU_ROOT`.

## CWRU (Case Western Reserve University bearing data centre)

Mirror of the full dataset (~890 MB). v2 uses the complete 12 kHz drive-end
matrix - 60 files against the 16 v1 used:

| state | sizes (mil) | positions | loads |
|---|---|---|---|
| normal | - | - | 0-3 HP |
| inner race | 007 / 014 / 021 / 028 | - | 0-3 HP |
| ball | 007 / 014 / 021 / 028 | - | 0-3 HP |
| outer race | 007 / 014 / 021 | @3, @6, @12 o'clock | 0-3 HP |

12 kHz drive-end accelerometer, SKF 6205-2RS JEM, shaft 1721-1797 rpm with
measured RPM in most files. Fan-end and 48 kHz data are present in the mirror
but unused here.

## MFPT (Machinery Failure Prevention Technology society bearing set)

Via the MathWorks packaging (~82 MB, 20 files). Baseline at 97.656 kHz /
270 lbs; outer-race faults at 25-300 lbs and inner-race faults at 0-300 lbs,
48.828 kHz, 25 Hz shaft. Fault frequencies (BPFO 81.1 Hz etc.) ship inside
each file and the loader converts them to orders. The NiceBearing geometry
differs from CWRU's 6205 - a second bearing family for free.

## SEU (Southeast University gearbox / DDS rig)

~2.2 GB of 8-channel CSV at 5.12 kHz (DAQ frequency limit 2000 Hz x 2.56),
two conditions (20 Hz / 0 V and 30 Hz / 2 V). `bearingset`: healthy, inner,
outer, ball, combined. `gearset`: healthy, chipped, missing tooth, root
crack, surface wear. One file is comma- rather than tab-separated and header
lengths vary; the loader sniffs both per file.

SEU matters to v2 for a structural reason: Nyquist is 2.56 kHz, so v1's fixed
2-5 kHz demodulation band cannot be constructed at all. Every SEU number in
the results therefore exercises the kurtogram band selection path.

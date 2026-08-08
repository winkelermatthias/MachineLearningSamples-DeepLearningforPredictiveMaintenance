N ?= 16
D ?= 90
PY = python3
export N_ASSETS=$(N)
export DAYS=$(D)

demo:
	cd src && N_ASSETS=$(N) DAYS=$(D) $(PY) -u run_demo.py ../fleet.duckdb

db:
	cd src && $(PY) -c "import sys;sys.path.insert(0,'.');from relspec.db import DB;DB('../schema.duckdb').migrate();print('schema ok')"

query:
	cd src && $(PY) -u -c "import sys;sys.path.insert(0,'.');from relspec.db import DB;\
	db=DB('../fleet.duckdb');print(db.q('SELECT * FROM finding_candidate ORDER BY rise_db DESC LIMIT 20'))"

clean:
	rm -f fleet.duckdb* schema.duckdb* run.log

# ---- v2: entropy codec, multi-rig eval, DSP studies -------------------------
# Dataset roots default to the public GitHub mirrors cloned under /workspace;
# override with CWRU_ROOT / MFPT_ROOT / SEU_ROOT.
v2-test:
	cd src && $(PY) -u test_codec2.py

v2-real:
	cd src && $(PY) -u eval_real.py --out ../eval_real.json

v2-fleet:
	cd src && $(PY) -u fleet2.py

v2-dsp:
	cd src && $(PY) -u eval_dsp.py

v2-rd:
	cd src && $(PY) -u eval_rd.py

v2-all: v2-test v2-fleet v2-real v2-dsp v2-rd

.PHONY: demo db query clean v2-test v2-real v2-fleet v2-dsp v2-rd v2-all

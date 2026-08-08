import sys, time, os; sys.path.insert(0, os.path.dirname(__file__))
from relspec.db import DB, build_fleet
from relspec.fleet import run_fleet, diagnose_fleet, refresh_pyramid, refresh_order_trend

DBPATH = sys.argv[1] if len(sys.argv) > 1 else '../fleet.duckdb'
N_ASSETS = int(os.environ.get('N_ASSETS', 16))
DAYS     = int(os.environ.get('DAYS', 90))

if os.path.exists(DBPATH): os.remove(DBPATH)
t = time.time()
db = DB(DBPATH).migrate()
specs = build_fleet(db, N_ASSETS)
print(f'fleet: {N_ASSETS} assets, {len(specs)} instrumented channels')
r = run_fleet(db, specs, days=DAYS, wake_h=6, seconds=1.0)
print('run:', r)
print('pyramid:', refresh_pyramid(db))
print('order_trend rows:', refresh_order_trend(db))
print('diagnose:', diagnose_fleet(db))
print(f'total {time.time()-t:.0f}s -> {DBPATH}')
db.con.close()

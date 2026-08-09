"""Zero-setup development server: embedded Postgres (pgserver) + the API in
one process. Not for production — compose runs real TimescaleDB there.

    python3 scripts/dev_server.py [--port 8000] [--data ./dev-pg]
"""
import argparse, pathlib, sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / 'server'))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=8000)
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--data', default=str(HERE.parents[0] / 'dev-pg'))
    args = ap.parse_args()

    import pgserver
    srv = pgserver.get_server(args.data)
    from relspec_service import config, db
    config.DATABASE_URL = srv.get_uri()
    db.migrate()
    print(f'embedded postgres at {args.data}')

    import uvicorn
    from relspec_service.app import app
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level='info')
    finally:
        srv.cleanup()

if __name__ == '__main__':
    main()

"""Cloud uplink: drains the spool into POST /v1/waveforms.

Auth: the gateway holds the workspace passphrase (devices never do).
It trades the phrase for a bearer token on start and re-mints on 401.
Alternatively RS_CLOUD_KEY supplies a pre-issued token that is sent
as Bearer directly (no passphrase exchange); the passphrase path
stays as the fallback when no key is configured.
Failure policy: network/5xx -> exponential backoff, stays pending
(store-and-forward); 409 out-of-order or 4xx -> 'skipped' with the
error recorded (never blocks the queue).
"""
from __future__ import annotations
import asyncio, base64, logging
import httpx
from .spool import Spool

log = logging.getLogger('uplink')


class Uplink:
    def __init__(self, base_url: str, passphrase: str, spool: Spool,
                 registry=None, cloud_key: str = ''):
        self.base = base_url.rstrip('/')
        self.phrase = passphrase
        self.cloud_key = cloud_key      # pre-issued bearer, used verbatim
        self.spool = spool
        self.registry = registry
        self.token: str | None = None
        self.online = False
        self.sent_total = 0

    async def _auth(self, client: httpx.AsyncClient):
        if self.cloud_key:
            self.token = self.cloud_key
            return
        r = await client.post(self.base + '/v1/workspaces',
                              json={'passphrase': self.phrase,
                                    'name': 'edge-gateway'})
        r.raise_for_status()
        self.token = r.json()['token']

    async def submit_one(self, client: httpx.AsyncClient, item: dict) -> bool:
        """True = item is finished (sent or skipped); False = retry later."""
        if not item['sensor_path']:
            self.spool.mark(item['id'], 'skipped',
                            {'error': 'no sensor_path mapping'})
            return True
        body = dict(sensor_path=item['sensor_path'], ts=item['ts_iso'],
                    fs=item['fs'], encoding='int16', scale=item['scale'],
                    units='g',
                    data_b64=base64.b64encode(item['data']).decode(),
                    client_ref=f"{item['dev']}:{item['ts_iso']}")
        try:
            if not self.token:
                await self._auth(client)
            r = await client.post(
                self.base + '/v1/waveforms', json=body,
                headers={'authorization': f'Bearer {self.token}'})
            if r.status_code == 401:
                self.token = None
                return False
            if r.status_code == 200:
                self.online = True
                self.sent_total += 1
                self.spool.mark(item['id'], 'sent', r.json())
                log.info('sent %s %s -> %s', item['dev'], item['ts_iso'],
                         r.json().get('sig_reason'))
                if self.registry and self.registry.on_change:
                    # refresh OPC UA gauges promptly (not just on the
                    # 2 s timer) so SpoolSent tracks the drain live
                    self.registry.on_change(self.registry.get(item['dev']))
                return True
            if 400 <= r.status_code < 500:
                self.spool.mark(item['id'], 'skipped',
                                {'status': r.status_code, 'body': r.text[:300]})
                log.warning('skipped %s: %s %s', item['dev'],
                            r.status_code, r.text[:120])
                return True
            raise httpx.HTTPStatusError('server error', request=r.request,
                                        response=r)
        except (httpx.TransportError, httpx.HTTPStatusError) as e:
            self.online = False
            self.spool.bump_tries(item['id'])
            log.warning('cloud unreachable (%s); buffering', type(e).__name__)
            return False

    async def run(self, stop: asyncio.Event, poll_s: float = 0.5):
        backoff = poll_s
        async with httpx.AsyncClient(timeout=20.0) as client:
            while not stop.is_set():
                item = self.spool.next_pending()
                if item is None:
                    await _wait(stop, poll_s); continue
                done = await self.submit_one(client, item)
                if done:
                    backoff = poll_s
                else:
                    await _wait(stop, backoff)
                    backoff = min(backoff * 2, 30.0)


async def _wait(stop: asyncio.Event, t: float):
    try:
        await asyncio.wait_for(stop.wait(), timeout=t)
    except asyncio.TimeoutError:
        pass

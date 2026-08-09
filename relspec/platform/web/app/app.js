/* relspec explorer — API loader. The page ships with empty sequences; this
   connects to the service with a passphrase, lists the fleet, and streams
   explorer bundles into the same rendering path the embedded build uses. */
'use strict';
(function(){
  const API = (window.RELSPEC_API || '') + '/v1';
  let token = sessionStorage.getItem('relspec_token') || '';

  const ov = document.createElement('div');
  ov.id = 'connect';
  ov.innerHTML = `
    <style>
      #connect{position:fixed;inset:0;z-index:50;display:flex;align-items:center;
        justify-content:center;background:var(--bg)}
      #connect .card{background:var(--panel);border:1px solid var(--hair);
        border-radius:8px;padding:28px 30px;max-width:440px;width:92%}
      #connect h2{font-family:var(--mono);font-size:13px;letter-spacing:.12em;
        text-transform:uppercase;color:var(--accent);margin:0 0 10px}
      #connect p{color:var(--muted);font-size:12.5px;margin:0 0 16px;line-height:1.5}
      #connect input{width:100%;background:var(--well);border:1px solid var(--hair);
        border-radius:5px;color:var(--ink);font-family:var(--mono);font-size:13px;
        padding:9px 11px;margin-bottom:12px}
      #connect button{background:var(--accent);color:var(--bg);border:0;
        border-radius:5px;font-family:var(--mono);font-size:12px;
        letter-spacing:.08em;padding:9px 18px;cursor:pointer}
      #connect .err{color:var(--ev-change);font-family:var(--mono);
        font-size:11px;min-height:16px;margin-top:8px}
    </style>
    <div class="card">
      <h2>connect to workspace</h2>
      <p>Enter the workspace passphrase. It is the only key: it locates the
         workspace and unlocks it. Nothing is stored server-side but a
         verifier; lose the phrase and the data is gone.</p>
      <input id="phrase" type="password" autocomplete="current-password"
             placeholder="workspace passphrase">
      <button id="go">Connect</button>
      <div class="err" id="cerr"></div>
    </div>`;

  async function api(path, opts={}){
    const r = await fetch(API+path, {...opts, headers: {
      'Content-Type': 'application/json',
      ...(token? {'Authorization': 'Bearer '+token} : {}),
      ...(opts.headers||{})}});
    if (!r.ok) throw new Error((await r.json().catch(()=>({}))).detail || r.status);
    return r.json();
  }

  async function connect(phrase){
    const r = await fetch(API+'/workspaces', {method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({passphrase: phrase})});
    if (!r.ok) throw new Error((await r.json().catch(()=>({}))).detail || 'connect failed');
    const j = await r.json();
    token = j.token;
    sessionStorage.setItem('relspec_token', token);
    return j;
  }

  async function loadFleet(){
    const list = await api('/fleet/overview');
    if (!list.length){
      document.getElementById('cerr').textContent =
        'workspace is empty — submit waveforms first (see the MCP tools)';
      return false;
    }
    // newest-first bundles, one per sensor, first selected
    for (let i=0;i<list.length;i++){
      const s = list[i];
      const b = await api(`/sensors/${s.sensor_id}/explorer-bundle?limit=400`);
      b.cat = s.plant; b.note = `sensor ${s.path} · live from workspace`;
      window.relspecApp.addSequence(b, i===0);
    }
    return true;
  }

  async function boot(){
    document.body.appendChild(ov);
    const btn = ov.querySelector('#go'), inp = ov.querySelector('#phrase');
    const err = ov.querySelector('#cerr');
    if (token){
      try { await loadFleet(); ov.remove(); return } catch(e){ token=''; }
    }
    const go = async () => {
      err.textContent = '';
      try {
        await connect(inp.value);
        if (await loadFleet()) ov.remove();
      } catch(e){ err.textContent = String(e.message || e) }
    };
    btn.onclick = go;
    inp.onkeydown = e => { if (e.key === 'Enter') go() };
    // e2e / deep-link convenience: #p=<passphrase>
    const m = location.hash.match(/^#p=(.+)$/);
    if (m){ inp.value = decodeURIComponent(m[1]); go() }
  }
  if (document.readyState === 'loading')
    window.addEventListener('DOMContentLoaded', boot);
  else boot();
})();

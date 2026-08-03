"""Teacher labeling: prompts, smoke tests and concurrent generation with
retries + JSONL checkpointing (interrupt and re-run — finished calls are kept)."""
import asyncio
import json
import os
import random

from .synth import COND_BOUNDS, FAULTS

_C0, _C1, _C2 = COND_BOUNDS

SYSTEM_TEACHER = f"""You are a senior vibration analyst (ISO 18436-2 Category IV) doing condition
monitoring for rotating machinery. You will receive one measurement report: machine nameplate
data, ISO 20816-3 zone boundaries, overall levels, a velocity-spectrum peak table and an
envelope-spectrum peak table.

Work the case step by step, exactly in this order:
1. INFER THE TRUE RUNNING SPEED. The report only gives the nameplate range. Find the 1x shaft
   peak: it must lie inside the nameplate range and be consistent with the harmonic comb
   (2x, 3x, ...) and any sideband spacings. Report speed in RPM.
2. IDENTIFY THE FAULT — exactly one of: {", ".join(FAULTS)}.
   Convert candidate fault frequencies (ratio x shaft speed) and compare against the peak
   tables: dominant 1x -> imbalance; 2x >= 1x (with 1x, 3x) -> misalignment; long integer
   harmonic comb + 0.5x subharmonic -> looseness; envelope peaks at BPFO harmonics without
   1x sidebands -> outer race; at BPFI with +-1x sidebands -> inner race; at 2xBSF with FTF
   sidebands -> rolling element; gear-mesh frequency (teeth x 1x) with +-1x sidebands -> gear
   mesh wear; nothing beyond baseline -> healthy.
3. IF THE BEARING DESIGNATION IS UNKNOWN, identify the most likely bearing from the candidate
   catalogue by matching envelope peaks to ratio x inferred speed. Otherwise repeat the given
   designation.
4. GRADE THE ISO ZONE (A, B, C or D) from the reported overall velocity RMS against the zone
   boundaries stated in the report.
5. GRADE THE BEARING CONDITION from the largest envelope-spectrum peak:
   none < {_C0} g <= early < {_C1} g <= moderate < {_C2} g <= severe.
6. RECOMMEND ACTIONS proportionate to the findings.

Write your full reasoning under a heading 'Analysis:' (show the arithmetic — expected fault
frequencies in Hz, which peaks matched). Then output ONE fenced json block, nothing after it:

```json
{{"inferred_speed_rpm": <number>, "fault_type": "<one of the list>", "iso_zone": "A|B|C|D",
 "bearing_condition": "none|early|moderate|severe", "bearing_designation": "<name>",
 "key_evidence": ["...", "..."], "recommended_actions": ["...", "..."]}}
```"""


def smoke_test(teachers):
    """Verify model ids are live and one call works per provider, before spending anything."""
    import time

    from openai import OpenAI

    for name, cfg in teachers.items():
        client = OpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"], timeout=120)
        try:
            ids = [m.id for m in client.models.list()]
            near = [i for i in ids if any(k in i.lower() for k in ("glm", "kimi"))][:12]
            status = "OK" if cfg["model"] in ids else "NOT FOUND — pick a nearby id and update the registry"
            print(f"[{name}] {cfg['model']}: {status}\n         nearby: {near}")
        except Exception as e:
            print(f"[{name}] models.list unavailable ({type(e).__name__}) — not fatal on some gateways")
        for attempt in range(4):     # free tiers 429 easily; a rate limit is not a broken setup
            try:
                r = client.chat.completions.create(
                    model=cfg["model"],
                    messages=[{"role": "user", "content": "Reply with the single word: pong"}],
                    max_tokens=512, temperature=cfg.get("temperature", 1.0),
                )
                print(f"[{name}] smoke test → {(r.choices[0].message.content or '').strip()[:60]!r}")
                break
            except Exception as e:
                if "429" in str(e) and attempt < 3:
                    wait = 15 * (attempt + 1)
                    print(f"[{name}] rate-limited (429) — retrying in {wait}s")
                    time.sleep(wait)
                    continue
                print(f"[{name}] smoke test FAILED: {e}")
                break


async def _call_one(client, tcfg, sem, sample, max_tokens=4096):
    async with sem:
        last_err = "empty response"
        for attempt in range(6):
            try:
                r = await client.chat.completions.create(
                    model=tcfg["model"],
                    messages=[{"role": "system", "content": SYSTEM_TEACHER},
                              {"role": "user", "content": sample["report"]}],
                    temperature=tcfg.get("temperature", 1.0),   # kimi-k3 only accepts 1
                    max_tokens=max_tokens,
                )
                msg = r.choices[0].message
                # reasoning models may split output; prefer visible content
                text = msg.content or getattr(msg, "reasoning_content", None) or ""
                if text.strip():
                    return sample["id"], text, None
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
                # rate limits deserve patience, not just exponential jitter
                base = 15 if "429" in last_err else 2 ** attempt
                await asyncio.sleep(base * (attempt + 1) if "429" in last_err
                                    else base + random.uniform(0, 1))
        return sample["id"], None, last_err


async def _run_teacher(name, cfg, samples, out_dir):
    from openai import AsyncOpenAI
    from tqdm.auto import tqdm

    path = os.path.join(out_dir, f"teacher_{name}.jsonl")
    done = set()
    if os.path.exists(path):
        with open(path) as fh:
            done = {json.loads(l)["id"] for l in fh if l.strip()}
    todo = [s for s in samples if s["id"] not in done]
    print(f"[{name}] {len(done)} cached, {len(todo)} to go")
    if not todo:
        return
    client = AsyncOpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"], timeout=600)
    sem = asyncio.Semaphore(cfg["concurrency"])
    coros = [_call_one(client, cfg, sem, s) for s in todo]
    ok, fails = 0, []
    with open(path, "a") as fh:
        for fut in tqdm(asyncio.as_completed(coros), total=len(coros), desc=name):
            sid, text, err = await fut
            if text:
                fh.write(json.dumps({"id": sid, "teacher": name, "text": text}) + "\n")
                fh.flush()
                ok += 1
            else:
                fails.append(err)
    print(f"[{name}] wrote {ok}, FAILED {len(fails)}"
          + (f" — last error: {fails[-1]}" if fails else ""))
    if fails and not ok:
        print(f"[{name}] every call failed — check the API key and model id before re-running")


async def run_teachers(teachers, pool, out_dir):
    """Split the pool between the teachers and run all providers concurrently."""
    names = list(teachers)
    share = len(pool) // len(names)
    jobs = []
    for i, name in enumerate(names):
        chunk = pool[i * share:(i + 1) * share] if i < len(names) - 1 else pool[i * share:]
        jobs.append(_run_teacher(name, teachers[name], chunk, out_dir))
    await asyncio.gather(*jobs)
    print("teacher generation done")

"""Platform smoke test for the installed `kittenml` package.

Runs the README's user paths against whatever `kittenml` is installed and writes
one JSON report. Each check is independent: a failure is recorded and the run
moves on, so one report shows everything that works and everything that does not.
"""
import json
import os
import platform
import sys
import tempfile
import time
import traceback

import numpy as np

RESULTS = []
TEXT = "One day, a little girl named Lily found a needle in her room."


def check(name):
    def wrap(fn):
        def run(*a, **kw):
            t0 = time.time()
            try:
                detail = fn(*a, **kw) or {}
                RESULTS.append({"name": name, "status": "pass", "secs": round(time.time() - t0, 1), **detail})
                print(f"PASS {name} ({time.time() - t0:.1f}s) {detail}", flush=True)
                return detail
            except Exception as e:
                RESULTS.append({"name": name, "status": "fail", "secs": round(time.time() - t0, 1),
                                "error": f"{type(e).__name__}: {e}"[:600],
                                "trace": traceback.format_exc()[-2500:]})
                print(f"FAIL {name}: {type(e).__name__}: {e}", flush=True)
                traceback.print_exc()
                return None
        return run
    return wrap


def audio_ok(audio, sr, min_s=0.5, max_s=60):
    a = np.asarray(audio)
    assert a.ndim == 1, f"expected mono 1-D audio, got shape {a.shape}"
    assert np.isfinite(a).all(), "audio has NaN/inf"
    dur = len(a) / sr
    assert min_s <= dur <= max_s, f"implausible duration {dur:.2f}s"
    peak = float(np.abs(a).max())
    assert peak > 1e-3, f"audio is silent (peak {peak})"
    return {"audio_s": round(dur, 2), "peak": round(peak, 3)}


def env():
    info = {"python": sys.version.split()[0], "implementation": platform.python_implementation(),
            "os": platform.platform(), "machine": platform.machine(), "processor": platform.processor(),
            "cpu_count": os.cpu_count()}
    try:
        if sys.platform == "linux":
            info["cpu"] = os.popen("lscpu | grep -m1 'Model name'").read().split(":", 1)[-1].strip()
        elif sys.platform == "darwin":
            info["cpu"] = os.popen("sysctl -n machdep.cpu.brand_string").read().strip()
        else:
            info["cpu"] = os.environ.get("PROCESSOR_IDENTIFIER", "") or platform.processor()
    except Exception as e:
        info["cpu"] = f"? ({e})"
    from importlib.metadata import version
    for pkg in ("kittenml", "torch", "torchaudio", "transformers", "onnxruntime", "numpy",
                "librosa", "soundfile", "phonemizer", "espeakng_loader", "kitten-text-processing", "diffusers"):
        try:
            info[pkg] = version(pkg)
        except Exception:
            info[pkg] = None
    return info


@check("import")
def t_import():
    import kittenml
    return {"version": kittenml.__version__}


@check("normalize_text")
def t_normalize():
    from kittenml import normalize_text
    out = normalize_text("I paid $3.50 on 12/03/2024 for 2kg.")
    assert "$" not in out and any(w in out.lower() for w in ("dollar", "three")), out
    return {"out": out[:120]}


def legacy(repo):
    from kittenml import KittenTTS
    m = KittenTTS(repo)
    assert "Bruno" in m.available_voices, m.available_voices
    t0 = time.time()
    audio = m.generate(TEXT, voice="Bruno")
    gen = time.time() - t0
    d = audio_ok(audio, 24000)
    slow = m.generate(TEXT, voice="Luna", speed=0.8)
    audio_ok(slow, 24000)
    chunks = list(m.generate_stream(TEXT + " " + TEXT, voice="Kiki"))
    assert chunks, "stream yielded nothing"
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "o.wav")
        m.generate_to_file(TEXT, p, voice="Bruno")
        assert os.path.getsize(p) > 1000
    d.update({"rtf": round(gen / d["audio_s"], 3), "stream_chunks": len(chunks)})
    return d


@check("legacy_nano")
def t_nano():
    return legacy("KittenML/kitten-tts-nano-0.8")


@check("legacy_mini")
def t_mini():
    return legacy("KittenML/kitten-tts-mini-0.8")


M2 = {}


@check("tts2_load")
def t2_load():
    from kittenml import KittenTTS
    t0 = time.time()
    m = KittenTTS("KittenML/kitten-tts-2", weights=os.environ.get("TTS2_WEIGHTS") or None)
    M2["m"] = m
    nv = len(m.available_voices)
    assert nv >= 40, f"only {nv} voices"
    return {"load_s": round(time.time() - t0, 1), "voices": nv, "sample_rate": m.sample_rate,
            "device": str(getattr(m, "device", "?"))}


@check("tts2_generate")
def t2_gen():
    m = M2["m"]
    t0 = time.time()
    audio = m.generate(TEXT, voice="Bruno")
    gen = time.time() - t0
    d = audio_ok(audio, m.sample_rate)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        M2["ref"] = f.name
    import soundfile as sf
    sf.write(M2["ref"], audio, m.sample_rate)
    d["rtf"] = round(gen / d["audio_s"], 2)
    return d


@check("tts2_expression")
def t2_expr():
    m = M2["m"]
    audio = m.generate("[joyful] We actually won <laugh> I can (((hardly))) believe it!",
                       voice="Kiki", preset="expressive")
    return audio_ok(audio, m.sample_rate)


@check("tts2_stream")
def t2_stream():
    m = M2["m"]
    t0 = time.time()
    first = None
    chunks = []
    for c in m.generate_stream(TEXT + " Then she sewed a button on her shirt.", voice="Luna"):
        if first is None:
            first = time.time() - t0
        chunks.append(np.asarray(c))
    assert chunks, "stream yielded nothing"
    d = audio_ok(np.concatenate(chunks), m.sample_rate)
    d.update({"chunks": len(chunks), "first_chunk_s": round(first, 2)})
    return d


@check("tts2_clone_with_transcript")
def t2_clone():
    m = M2["m"]
    audio = m.generate("This is my own voice, cloned.", reference=M2["ref"], reference_text=TEXT)
    return audio_ok(audio, m.sample_rate)


@check("tts2_clone_whisper")
def t2_clone_whisper():
    m = M2["m"]
    audio = m.generate("The transcript is optional.", reference=M2["ref"])
    return audio_ok(audio, m.sample_rate)


PARTS = {
    "basic": [t_import, t_normalize],
    "legacy": [t_nano, t_mini],
}


def run_tts2():
    if t2_load() is None:
        return
    for t in (t2_gen, t2_expr, t2_stream):
        t()
    if "ref" in M2:
        t2_clone()
        t2_clone_whisper()


def child(part, out):
    if part == "tts2":
        run_tts2()
    else:
        for t in PARTS[part]:
            t()
    json.dump(RESULTS, open(out, "w"))


def main(out):
    # Each model family runs in its own process: espeak and native runtimes can
    # exit() or segfault, which must not take the other families' results with it.
    import subprocess
    info = env()
    print(json.dumps(info, indent=1), flush=True)
    parts = ["basic", "legacy"] + ([] if os.environ.get("SKIP_TTS2") == "1" else ["tts2"])
    for part in parts:
        part_out = f"{out}.{part}.json"
        if os.path.exists(part_out):
            os.remove(part_out)
        t0 = time.time()
        proc = subprocess.run([sys.executable, __file__, "--part", part, part_out],
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        sys.stdout.write(proc.stdout)
        sys.stdout.flush()
        if os.path.exists(part_out):
            RESULTS.extend(json.load(open(part_out)))
        if proc.returncode != 0 and not os.path.exists(part_out):
            RESULTS.append({"name": f"{part}_process", "status": "crash", "secs": round(time.time() - t0, 1),
                            "error": f"process exited with code {proc.returncode}",
                            "trace": proc.stdout[-2500:]})
            print(f"CRASH {part}: exit code {proc.returncode}", flush=True)
    json.dump({"env": info, "results": RESULTS}, open(out, "w"), indent=1)
    failed = [r["name"] for r in RESULTS if r["status"] != "pass"]
    print("FAILED:", failed or "none")
    return 1 if failed else 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--part":
        child(sys.argv[2], sys.argv[3])
    else:
        sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "qa-report.json"))

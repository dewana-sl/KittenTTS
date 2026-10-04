"""Turn the per-job qa-*.json reports into one Markdown matrix."""
import glob
import json
import os
import sys

CHECKS = ["import", "normalize_text", "legacy_nano", "legacy_mini", "tts2_load", "tts2_generate",
          "tts2_expression", "tts2_stream", "tts2_clone_with_transcript", "tts2_clone_whisper"]
MARK = {"pass": "✅", "fail": "❌", "crash": "💥"}


def cell(rep):
    inst = rep["install"]
    if not inst.get("python_available"):
        return "n/a (no Python build)"
    if inst.get("rc") != "0":
        tail = inst.get("log_tail", "")
        if "Requires-Python" in tail or "requires a different Python" in tail:
            return "⛔ refused (Requires-Python)"
        return "❌ install failed"
    res = {r["name"]: r for r in rep["results"]}
    crashed = [r for r in rep["results"] if r["status"] == "crash"]
    bad = [n for n in CHECKS if n in res and res[n]["status"] != "pass"]
    if not bad and not crashed:
        rtf = res.get("tts2_generate", {}).get("rtf")
        return "✅" + (f" RTF {rtf}" if rtf is not None else "")
    return "❌ " + ", ".join(bad + [r["name"] for r in crashed])


def main(d):
    reps = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(d, "qa-*.json")))]
    runners = sorted({r["runner"] for r in reps})
    pys = sorted({r["python_requested"] for r in reps}, key=lambda v: tuple(map(int, v.split("."))))
    by = {(r["runner"], r["python_requested"]): r for r in reps}
    print(f"## kittenml platform QA — {reps[0]['source'] if reps else '?'}\n")
    print("| Runner | CPU | " + " | ".join(f"py{p}" for p in pys) + " |")
    print("|---|---|" + "---|" * len(pys))
    for run in runners:
        cpu = next((by[(run, p)]["env"].get("cpu") for p in pys if (run, p) in by and by[(run, p)]["env"].get("cpu")), "")
        cells = [cell(by[(run, p)]) if (run, p) in by else "—" for p in pys]
        print(f"| {run} | {cpu} | " + " | ".join(cells) + " |")
    print("\n### Failures\n")
    for r in reps:
        label = f"{r['runner']} / py{r['python_requested']}"
        inst = r["install"]
        if inst.get("python_available") and inst.get("rc") != "0":
            print(f"<details><summary>{label}: install failed</summary>\n\n```\n{inst.get('log_tail', '')[-1500:]}\n```\n</details>\n")
        for res in r["results"]:
            if res["status"] != "pass":
                print(f"<details><summary>{label}: {res['name']} — {res.get('error', '')[:200]}</summary>\n\n"
                      f"```\n{res.get('trace', '')[-1500:]}\n```\n</details>\n")


if __name__ == "__main__":
    main(sys.argv[1])

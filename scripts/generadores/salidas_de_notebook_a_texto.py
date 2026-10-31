"""Guarda en texto las salidas de un notebook (sin imágenes) para comparar antes y después."""
import json, sys
nb = json.load(open(sys.argv[1], encoding="utf-8"))
out = []
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code":
        continue
    for o in c.get("outputs", []):
        if o["output_type"] == "stream":
            out.append(f"[{i}] stream\n" + "".join(o["text"]))
        elif o["output_type"] in ("execute_result", "display_data"):
            out.append(f"[{i}] data\n" + "".join(o["data"].get("text/plain", "")))
        elif o["output_type"] == "error":
            out.append(f"[{i}] ERROR {o['ename']}: {o['evalue']}")
open(sys.argv[2], "w", encoding="utf-8").write("\n".join(out))
print(sys.argv[1], "->", len(out), "bloques de salida")

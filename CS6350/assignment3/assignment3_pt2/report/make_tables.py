"""Turn the JSON files in outputs/ into LaTeX tables in tables/.

Usage: python make_tables.py <outputs_dir> <tables_dir>
"""

import json
import sys
from pathlib import Path


def esc(s) -> str:
    s = str(s)
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"),
                 ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}"),
                 ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")]:
        s = s.replace(a, b)
    return s


def table(header, rows, align):
    lines = [rf"\begin{{tabular}}{{{align}}}", r"\toprule",
             " & ".join(header) + r" \\", r"\midrule"]
    lines += [" & ".join(esc(c) for c in r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def main():
    out, tab = Path(sys.argv[1]), Path(sys.argv[2])
    tab.mkdir(parents=True, exist_ok=True)

    g = json.loads((out / "graph_results.json").read_text())
    rank = lambda xs: [(i + 1, *x) for i, x in enumerate(xs)]
    specs = {
        "outdegree": (["Rank", "Node id", "Out-degree"],
                      [(r["id"], f'{r["outDegree"]:,}') for r in g["outdegree"]]),
        "indegree": (["Rank", "Node id", "In-degree"],
                     [(r["id"], f'{r["inDegree"]:,}') for r in g["indegree"]]),
        "pagerank": (["Rank", "Node id", "PageRank"],
                     [(r["id"], f'{r["pagerank"]:.2f}') for r in g["pagerank"]]),
        "components": (["Rank", "Component id", "Nodes"],
                       [(r["component"], f'{r["count"]:,}') for r in g["components"]]),
        "triangles": (["Rank", "Node id", "Triangles"],
                      [(r["id"], f'{r["count"]:,}') for r in g["triangles"]]),
    }
    for name, (hdr, rows) in specs.items():
        (tab / f"{name}.tex").write_text(table(hdr, rank(rows), "rrr"))
    (tab / "graph_stats.tex").write_text(
        f"\\newcommand{{\\nVertices}}{{{g['n_vertices']:,}}}\n"
        f"\\newcommand{{\\nEdges}}{{{g['n_edges']:,}}}\n"
        f"\\newcommand{{\\nComponents}}{{{g['n_components']:,}}}\n")

    snaps = sorted(out.glob("entities_*min.json"))
    for p in snaps:
        s = json.loads(p.read_text())
        rows = [(e, c) for e, c in s["top10"]]
        (tab / f"{p.stem}.tex").write_text(table(["Rank", "Entity", "Count"], rank(rows), "rlr"))


if __name__ == "__main__":
    main()

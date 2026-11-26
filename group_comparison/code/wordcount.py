#!/usr/bin/env python3
"""Political Analysis word count: abstract + text + figure legends + footnotes, references
excluded (Cambridge Core, 'Preparing your materials'). Mathematics counts as one word per
inline expression; display equations are not counted. Table captions are reported separately."""
import os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def words(t):
    t = re.sub(r"(?<!\\)%.*", "", t)
    t = re.sub(r"\\begin\{(equation|align)\*?\}.*?\\end\{\1\*?\}", " ", t, flags=re.S)
    t = re.sub(r"\$[^$]*\$", " X ", t)
    t = re.sub(r"\\(cite\w*|ref|eqref|label|input|url|includegraphics)(\[[^]]*\])*\{[^}]*\}", " ", t)
    t = re.sub(r"\\[a-zA-Z]+\*?", " ", t)
    t = re.sub(r"[{}~\[\]]", " ", t)
    return len(re.findall(r"[A-Za-z0-9][A-Za-z0-9'.\-]*", t))


def main(path):
    s = open(path, encoding="utf-8").read()
    ab = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", s, re.S).group(1)
    end = min(s.index(m) for m in (r"\section*{Funding}", r"\section*{Data Availability") if m in s)
    body = s[s.index(r"\section{Introduction}"):end]
    body_nofig = re.sub(r"\\begin\{figure\}.*?\\end\{figure\}", " ", body, flags=re.S)   # legends counted once, below
    secs = re.split(r"(\\(?:sub)?section\*?\{[^}]*\})", body_nofig)
    caps = re.findall(r"\\begin\{figure\}.*?\\caption\{(.*?)\}\s*\\label", s, re.S)
    fns = re.findall(r"\\footnote\{(.*?)\}", body, re.S)
    tcaps = sum(words(c) for c in re.findall(r"\\caption\{(.*?)\}\s*\\label\{tab", s, re.S))
    cur = None
    for p in secs:
        if p.startswith("\\"):
            cur = p
        elif cur:
            print(f"{words(p):5d}  {cur[:70]}")
    a, b, c, f = words(ab), words(body_nofig), sum(words(x) for x in caps), sum(words(x) for x in fns)
    print(f"abstract {a} | text {b} | figure legends {c} | footnotes {f} | TOTAL {a + b + c + f} (limit 6000)")
    print(f"(table captions, not in the stated count: {tcaps})")
    return a + b + c + f


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "manuscript", "manuscript.tex"))

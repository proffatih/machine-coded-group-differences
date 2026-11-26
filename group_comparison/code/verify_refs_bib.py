#!/usr/bin/env python3
"""Every cited reference (manuscript + appendix): DOI present and Crossref title matches the
.bib title (title and subtitle joined). Entries without a DOI are listed for a manual check.
Writes results/refs_check.txt."""
import difflib, json, os, re, time, unicodedata, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def norm(t):
    t = unicodedata.normalize("NFKD", re.sub(r"\\[a-zA-Z]+|[{}\\'\"`^~]", "", t))
    return re.sub(r"[^a-z0-9 ]", " ", t.lower()).split()


def main():
    keys = set()
    for f in ("manuscript.aux", "appendix.aux"):
        for c in re.findall(r"\\citation\{([^}]+)\}", open(os.path.join(ROOT, "manuscript", f)).read()):
            keys |= set(k.strip() for k in c.split(","))
    bib = open(os.path.join(ROOT, "refs", "refs.bib"), encoding="utf-8").read()
    entries = {m.group(2): m.group(3) for m in re.finditer(r"@(\w+)\{([^,]+),(.*?)\n\}", bib, re.S)}
    out, bad = [], 0
    for k in sorted(keys):
        e = entries[k]
        title = re.search(r"\btitle\s*=\s*\{(.*?)\},?\s*\n", e, re.S).group(1)
        doi = re.search(r"\bdoi\s*=\s*\{([^}]*)\}", e)
        if not doi:
            out.append(f"NO-DOI  {k}: {title[:80]}"); continue
        req = urllib.request.Request(f"https://api.crossref.org/works/{doi.group(1)}",
                                     headers={"User-Agent": "reference-check/1.0"})
        try:
            m = json.load(urllib.request.urlopen(req, timeout=30))["message"]
        except Exception as ex:
            out.append(f"FAIL    {k}: {doi.group(1)} ({ex})"); bad += 1; continue
        ct = " ".join((m.get("title") or [""]) + (m.get("subtitle") or []))
        r = difflib.SequenceMatcher(None, norm(title), norm(ct)).ratio()
        tag = "OK" if r >= .85 else "CHECK"
        bad += tag != "OK"
        out.append(f"{tag:7s} {k}: {r:.2f} | {ct[:90]}")
        time.sleep(0.2)
    txt = "\n".join(out) + f"\n{len(keys)} cited, {sum(l.startswith('OK') for l in out)} OK, " \
          f"{sum(l.startswith('NO-DOI') for l in out)} without DOI, {bad} to check\n"
    open(os.path.join(ROOT, "results", "refs_check.txt"), "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()

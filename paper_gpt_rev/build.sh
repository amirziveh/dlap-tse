#!/bin/bash
# Build GPT-revision manuscript: placeholder fill + xelatex-free pdflatex chain
set -e
cd /home/ubuntu/research/dlap-tse/paper_gpt_rev

# fill figure placeholder
/home/ubuntu/venvs/dlap-tse/bin/python - <<'PYEOF'
p = 'manuscript.tex'
t = open(p, encoding='utf-8').read()
if '@@SIGNWIN_BEAT@@' in t:
    t = t.replace('@@SIGNWIN_BEAT@@', '16')
    open(p, 'w', encoding='utf-8').write(t)
    print('placeholder filled -> 16')
else:
    print('no placeholder (already filled)')
PYEOF

pdflatex -interaction=nonstopmode manuscript.tex > /dev/null 2>&1 || true
bibtex manuscript > bibtex.log 2>&1 || true
pdflatex -interaction=nonstopmode manuscript.tex > /dev/null 2>&1 || true
pdflatex -interaction=nonstopmode manuscript.tex > /dev/null 2>&1 || true

echo "=== errors: $(grep -c '^!' manuscript.log || true)"
grep '^!' manuscript.log | head -5 || true
echo "=== overfull: $(grep -c 'Overfull' manuscript.log || true)"
echo "=== undefined refs: $(grep -c 'undefined' manuscript.log || true)"
echo "=== bibtex issues:"; grep -iE 'warning|error' bibtex.log | head -5 || echo "(clean)"
pdfinfo manuscript.pdf | grep Pages

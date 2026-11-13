"""The research graph: typed nodes in graph/<type>/<name>.md, a YAML header for facts and links, prose below.

Node types and their header fields (id is always <type>/<file name>):
  environment  title
  ability      title, env, measure, target (null if the user has set none)
  model        title;                      links: uses → principle, has → ability (value, n, source)
  principle    title, status (adopted | tested | untested | rejected);
                                           links: addresses → problem (basis: shown | claimed, evidence)
  paper        title, cite;                links: proposes → principle, addresses → problem (basis: claimed)
  problem      title, parent (a problem id or null)
  result       title, card, model, env, measure, value, against, n, outcome (pass | fail),
               lost_points (benchmark points of success lost to this failure; 0 for a diagnostic)
                                           links: instance → problem (weight, weight_basis: counted | estimated),
                                                  tested → principle, measures → ability
A link is written once, on the node it leaves; the script derives the other direction.

  python tools/graph.py check            every id resolves, every field is present, every link has the right types
  python tools/graph.py rank             problems ranked by cost, breadth, neglect and blocked abilities
  python tools/graph.py show <id>        one node with its links in both directions
  python tools/graph.py gaps             principles never tested here; problems nothing addresses; one-environment principles
  python tools/graph.py json             the whole graph as JSON on stdout (for rendering)
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent / 'graph'
FIELDS = dict(environment=[], ability=['env', 'measure', 'target'], model=[], principle=['status'], paper=['cite'],
              problem=['parent'], result=['card', 'model', 'env', 'measure', 'value', 'against', 'n', 'outcome', 'lost_points'])
RELS = {('model', 'uses'): 'principle', ('model', 'has'): 'ability', ('principle', 'addresses'): 'problem',
        ('paper', 'proposes'): 'principle', ('paper', 'addresses'): 'problem', ('result', 'instance'): 'problem',
        ('result', 'tested'): 'principle', ('result', 'measures'): 'ability'}
REFS = dict(env='environment', model='model', parent='problem')


def load():
    nodes = {}
    for f in sorted(ROOT.glob('*/*.md')):
        _, head, body = f.read_text().split('---\n', 2)
        n = yaml.safe_load(head)
        n['body'], n['file'] = body.strip(), str(f.relative_to(ROOT.parent))
        n.setdefault('links', [])
        nodes[n['id']] = n
    return nodes


def check(nodes):
    bad = []
    for i, n in nodes.items():
        t = n.get('type')
        if n['file'] != f'graph/{i}.md' or i.split('/')[0] != t:
            bad.append(f'{n["file"]}: id {i} or type {t} does not match the path')
        for k in ['title'] + FIELDS.get(t, []):
            if k not in n:
                bad.append(f'{i}: field {k} missing')
        for k, want in REFS.items():
            v = n.get(k)
            if v is not None and (v not in nodes or nodes[v]['type'] != want):
                bad.append(f'{i}: {k} = {v} is not a known {want}')
        for l in n['links']:
            want = RELS.get((t, l.get('rel')))
            if want is None:
                bad.append(f'{i}: a {t} cannot have a link "{l.get("rel")}"')
            elif l.get('to') not in nodes or nodes[l['to']]['type'] != want:
                bad.append(f'{i}: {l["rel"]} → {l.get("to")} is not a known {want}')
            if l.get('rel') == 'instance' and ('weight' not in l or l.get('weight_basis') not in ('counted', 'estimated')):
                bad.append(f'{i}: instance → {l.get("to")} needs weight and weight_basis')
            if l.get('rel') == 'addresses' and l.get('basis') not in ('shown', 'claimed'):
                bad.append(f'{i}: addresses → {l.get("to")} needs basis shown or claimed')
    return bad


def back(nodes):
    b = defaultdict(list)
    for i, n in nodes.items():
        for l in n['links']:
            b[l['to']].append(dict(l, to=i, rel='← ' + l['rel']))
    return b


def family(nodes, p):
    """A problem and every problem under it."""
    out, grew = {p}, True
    while grew:
        more = {i for i, n in nodes.items() if n['type'] == 'problem' and n.get('parent') in out} - out
        out, grew = out | more, bool(more)
    return out


def rank(nodes):
    best = defaultdict(float)                                               # ability → best value any of our models has
    for n in nodes.values():
        if n['type'] == 'model' and not n.get('external'):
            for l in n['links']:
                if l['rel'] == 'has':
                    best[l['to']] = max(best[l['to']], l.get('value', 0))
    rows = []
    for p, n in nodes.items():
        if n['type'] != 'problem':
            continue
        fam = family(nodes, p)
        fails = [(r, l) for r in nodes.values() if r['type'] == 'result' and r['outcome'] == 'fail'
                 for l in r['links'] if l['rel'] == 'instance' and l['to'] in fam]
        shown = [q['id'] for q in nodes.values() if q['type'] == 'principle' for l in q['links']
                 if l['rel'] == 'addresses' and l['to'] in fam and l['basis'] == 'shown']
        claimed = [q['id'] for q in nodes.values() if q['type'] in ('principle', 'paper') for l in q['links']
                   if l['rel'] == 'addresses' and l['to'] in fam and l['basis'] == 'claimed']
        blocked = sorted({m['to'] for r, _ in fails for m in r['links'] if m['rel'] == 'measures' and best[m['to']] == 0})
        rows.append(dict(problem=p, parent=n.get('parent'), failures=len({r['id'] for r, _ in fails}),
                         cost=round(sum(l['weight'] * r['lost_points'] for r, l in fails), 1),
                         estimated=any(l['weight_basis'] == 'estimated' for _, l in fails),
                         cards=len({str(r['card']) for r, _ in fails}), envs=len({r['env'] for r, _ in fails}),
                         shown=len(set(shown)), claimed=len(set(claimed)), blocked=blocked))
    return sorted(rows, key=lambda r: (-r['cost'], -r['failures']))


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'check'
    nodes = load()
    bad = check(nodes)
    if cmd == 'check' or bad:
        print('\n'.join(bad) if bad else f'{len(nodes)} nodes, {sum(len(n["links"]) for n in nodes.values())} links, no faults')
        sys.exit(bool(bad))
    if cmd == 'rank':
        print('Cost = benchmark points of success lost to failures in the problem or the problems under it, by the links\' weights')
        print('(~ marks a cost resting on estimated weights). Shown / claimed = principles addressing it, by evidence here / by a paper or untested.\n')
        print(f'{"problem":44}{"cost":>7} {"fails":>5} {"cards":>5} {"shown":>5} {"claimed":>7}  abilities at zero behind it')
        for r in rank(nodes):
            name = ('  ' if r['parent'] else '') + r['problem'].split('/')[1]
            print(f'{name:44}{("~" if r["estimated"] else "") + str(r["cost"]):>7} {r["failures"]:>5} {r["cards"]:>5} {r["shown"]:>5} {r["claimed"]:>7}  '
                  + ', '.join(a.split('/')[1] for a in r['blocked']))
    elif cmd == 'show':
        n, b = nodes[sys.argv[2]], back(nodes)
        print(yaml.safe_dump({k: v for k, v in n.items() if k not in ('body', 'links')}, sort_keys=False, allow_unicode=True).strip())
        for l in n['links'] + b[n['id']]:
            print(f'  {l["rel"]:14} {l["to"]:48}', {k: v for k, v in l.items() if k not in ('rel', 'to')} or '')
        print('\n' + n['body'])
    elif cmd == 'gaps':
        b = back(nodes)
        print('Principles with no result here:')
        for i, n in nodes.items():
            if n['type'] == 'principle' and not any(l['rel'] == '← tested' for l in b[i]):
                print('  ', i, f'({n["status"]})')
        print('Principles tested in one environment only:')
        for i, n in nodes.items():
            envs = {nodes[l['to']]['env'] for l in b[i] if l['rel'] == '← tested'}
            if n['type'] == 'principle' and len(envs) == 1:
                print('  ', i, sorted(envs))
        print('Problems with failures and no principle shown to address them:')
        for r in rank(nodes):
            if r['failures'] and not r['shown']:
                print('  ', r['problem'], f'({r["failures"]} failures, {r["claimed"]} claimed)')
    elif cmd == 'json':
        json.dump(list(nodes.values()), sys.stdout, indent=1, default=str)


if __name__ == '__main__':
    main()

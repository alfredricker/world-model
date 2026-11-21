---
id: result/153-no-word-classes-from-100-sentences
type: result
title: with one or two records per word the rule finds no word classes
card: '153'
model: model/og4
env: environment/word-stream
measure: percent of the gap closed, mean of 10 worlds at 100 sentences
value: 38.2
against: 90.0
n: 10
outcome: fail
lost_points: 0
links:
- rel: tested
  to: principle/kinds-by-one-energy
---
From [card 153](../../experiments/153-kinds-found-by-one-energy-in-two-domains/card.md): at 100 sentences four kinds are found in 1 world of 10 and none in 2; at 200 sentences the gap closed is 84.9% with 4 to 6 kinds of purity 62 to 79%. The true classes are worth 0.36 nats at 100 sentences, so the information is there. Reasoned, not separately tested: one record favours the right class by about a nat, less than joining a small new kind costs under the prior, and no single word has enough records to propose a good kind. Items that share a looks code would have to propose together.

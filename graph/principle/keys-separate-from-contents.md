---
id: principle/keys-separate-from-contents
type: principle
title: Keys separate from contents
status: untested
links:
- rel: addresses
  to: problem/recall-key
  basis: claimed
---
Memory holds two parts: a key used to find an entry and a content returned from it, and the key is learned separately from the content. In the Emergent Symbol Binding Network (Webb et al. 2020) the controller writes keys that act as variable names while the stored contents stay as raw embeddings; Neural Episodic Control also keeps keys and values apart. For recall, the key for an episode need not be a copy of its content.

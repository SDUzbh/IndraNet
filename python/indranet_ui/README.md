# indranet_ui: look at a model in the browser

A local web page on the tfi engine. Type a prompt and see the model's attention, the logit lens,
attribution and patching, with every number on hover.

```bash
make python                                            # once: the tfi bindings (CMake + pip install nanobind)
PYTHONPATH=python python3 -m indranet_ui model.safetensors --fp16 --open
PYTHONPATH=python python3 -m indranet_ui models/ other.safetensors --fp16   # several models to choose from
```

Give it files, folders, or both. In a folder, every `.safetensors` file that holds a model config is
offered; other files, such as saved activations, are skipped. The first model loads at start. The
page's model button lists them all, and picking one loads it in place of the current one, so only one
model is in memory at a time.

Any file the engine loads works: models from `indranet.export` and checkpoints converted with
[`indranet_hf`](../indranet_hf/). With a tokenizer (converted models) or a vocabulary, prompts are
text. Otherwise they are token ids, `5,17,3`. Target tokens are text, such as ` Paris` with its
leading space, or ids written `#123`.

| Option | |
|---|---|
| `--port 8765` | the port; `0` picks a free one |
| `--host 127.0.0.1` | the address to listen on. The page has no login, so keep it on this computer unless you trust the network |
| `--fp16` | keep the weights in half precision (half the memory) |
| `--device cpu\|both\|gpu` | where the engine computes (`both`, the default, uses the GPU for large batched work on macOS) |
| `--open` | open the page in the default browser |
| `--verbose` | log every request |

## The page

The page opens on the prompt, over a picture of Indra's net. Type a prompt and press Enter
(Shift+Enter starts a new line), or click an example (indirect object, induction, fact recall; each
also fills in the patching fields that go with it). The round button at the left of the prompt (or
the model's name under it) opens the models menu. It groups the models by family (Llama, Qwen, ...),
with a search box above them, and opens the chosen family's models in a panel beside it. **Custom
model…** opens a model file, or a folder of them, by its path on this computer. This works only while
the page listens on this computer alone (the default `--host`). Running a prompt switches to the results
below. **Edit prompt**, the IndraNet mark, or the browser's Back button returns to the prompt.

- **Tokens**: the prompt as tokens, each coloured by its surprisal (−log p, full red at 10 nats).
  Hover a token for its probability and the model's top next-token predictions. Click it to select
  its position, which the other views use as the query and measurement position.
- **Attention**: every head of a layer as a thumbnail. Click one to enlarge it with values on hover.
  While a head is open, bars under the tokens show where the selected token attends.
- **Logit lens**: the residual stream after every sublayer (rows) at every position (columns),
  decoded with the final norm and unembedding. Each cell shows the top token, coloured by its
  probability or by the probability of the actual next token. Hover a cell for the top 5.
- **Attribution**: direct logit attribution of `logit(target) − logit(minus)` at a position, as a
  layer × (heads + MLP) map and the largest parts. The target defaults to the top prediction.
- **Patching**: activation patching from the prompt (clean) into a corrupted prompt of the same
  length. Choose the site (`resid_pre` ... `mlp_out` by layer and position, or `heads`) and the
  method (exact, attribution, or mixed; mixed marks the exactly measured cells with a dot).
- **Head scores**: induction, previous-token, duplicate-token, first-token and self scores of every
  head on random repeated sequences. Click a head to open its pattern.

## How it works

[server.py](server.py) serves the page and a JSON API using Python's standard library
(`http.server`). It loads one model at start and calls the engine through the `tfi` bindings,
one call at a time. The engine releases the GIL, so the server keeps answering meanwhile.
Attention patterns go over the wire as raw float32. For one prompt, every layer's patterns are
kept while they fit in 256 MB, so moving between layers costs nothing. The page itself
([static/](static/)) is plain JavaScript drawing on canvas, with no framework and no build step,
and follows the system's light or dark mode.

| Route (POST, JSON) | Returns |
|---|---|
| `info` | the model's shape, config and file |
| `models` / `select` `{index}` | the model files offered, and loading one of them (returns its `info`) |
| `add` `{path}` | offer a model file, or a folder's model files, by path (refused unless the page listens on this computer alone) |
| `tokenize` `{text, special}` / `texts` `{tokens}` | token ids and their text |
| `run` `{tokens, k}` | top-k next-token predictions per position, the probability of each actual next token, the loss |
| `attention` `{tokens, layer}` | the (heads, T, T) pattern, as little-endian float32 (shape in `X-Shape`) |
| `lens` `{tokens, k}` | the logit lens at every position (`tfi.lens_grid`) |
| `dla` `{tokens, pos, target, other}` | direct logit attribution |
| `patch` `{clean, corrupt, pos, target, other, site, method, steps, verify}` | a patching table |
| `heads` `{segment_len, n_samples}` | head pattern scores |

`python/tests/test_ui.py` checks every route over HTTP against the engine.

# MiniMax-M3.1-Flash EXL3 on one M5 Ultra (TensorFold, Metal)

MiniMaxAI's [MiniMax-M3.1-Flash](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash) (428B-A23B MoE, block-sparse
attention with the lightning indexer, a vision tower) served on one 256 GB Mac Studio (M5 Ultra) by
[TensorFold](https://github.com/ashhart/TensorFold)'s native Metal engine, from Mia-AiLab's published EXL3 pack
[`Mia-AiLab/MiniMax-M3.1-Flash-EXL3-EAQ-v1-3.06-bpw`](https://huggingface.co/Mia-AiLab/MiniMax-M3.1-Flash-EXL3-EAQ-v1-3.06-bpw)
**as published** (revision `c33e7a66`): no conversion; the served weights are the pack's bytes.

The pack is a lossy third-party quantization of the release (EAQ v1, 3.06 bits a weight on average: attention 6-bit,
dense and shared MLPs 5-bit, routed experts 3-bit, head 8-bit, `mul1` codebook; the fp16 index projections, the bf16
embedding and norms, the fp32 router, the DSpark draft head and the vision tower as in the release). It stays much
closer to the release than our MLX conversions: teacher-forced KL **0.074**, top-1 agreement **90.6%** (below).
The price is speed: EXL3's trellis codes take arithmetic to decode on a GPU, so this recipe decodes at about 19.5
tokens a second where [the mix256k recipe](https://github.com/BobClawblaw/minimax-m31-flash-mlx-m5-ultra) does 43 (drafted),
and reads prompts at about 500 tokens a second against its 1,480.

| | this recipe (EXL3 3.06 bpw) | mix256k (MLX affine, ours) |
|---|---|---|
| KL against the release, teacher-forced | **0.074** (90.6% top-1; 2,406 tokens) | 0.295 (78.4%; 1,024 tokens, other text) |
| Decode, one stream (prose) | 19.5 tok/s | 43.1 tok/s (DFlash2 drafts) |
| Prompts, cold, 8k / 32k / 128k | 486 / 509 / 501 tok/s | 1,483 / 1,491 / 1,379 tok/s |
| Weights held (with the tower) | 158.9 GiB | see its README |
| Window served | 262,144 tokens (383,360 fit) | 147,456 tokens |

Pick this one when answers matter more than speed; the mix256k recipe when speed does.

## Run it

```sh
git clone https://github.com/BobClawblaw/minimax-m31-flash-exl3-m5-ultra && cd minimax-m31-flash-exl3-m5-ultra
cp local.example.toml local.toml      # point tensorfold_src at a clone of github.com/BobClawblaw/TensorFold
python3 -m mac download               # the pack at the pinned revision (171 GB)
python3 -m mac build                  # the pinned engine, libtfgrammar and the image decoders
python3 -m mac up                     # http://127.0.0.1:8897/v1 (OpenAI-compatible)
python3 -m mac chat "Hello"
python3 -m mac down
```

`python3 -m mac command` prints the serve command; `python3 -m mac launchd` writes a LaunchAgent (not loaded).
Settings live in `recipe.toml` (defaults), `local.toml` (this Mac) and upper-case environment variables.

## The engine

Branch [`minimax-m3-metal-exl3`](https://github.com/BobClawblaw/TensorFold/tree/minimax-m3-metal-exl3) of TensorFold
(the recipe pins `afcc366c1`: the drafter's commit `ac35cd022` and a test fix), on top of the MiniMax-M3 Metal family (ashhart/TensorFold#1235) and the
CUDA backend that serves the same pack on two DGX Sparks (#1388). What the Metal path adds:

- **EXL3 rows** (`zig/kernels/metal/core/exl3_rows.metal`): a decode window's rows (up to 32) through EXL3 matrices
  in one dispatch. A threadgroup owns one 128-column Hadamard block of one K range with both rotations folded in;
  the last range to arrive sums the partials in range order, so a row's bits never depend on what else the window
  holds. q | k | v share a dispatch, and so do each token's four routed experts and the shared expert.
- **EXL3 prompt chunks** (`exl3_mm.metal`): 64 rows by 128 columns a threadgroup on the tensor units, each simdgroup
  decoding its own weight tiles straight into the fp16 operand (exact fp32 products and sums), dense and expert-sorted.
- The pack's fp16 index pair and bf16 embedding, its DSpark head (the release's MXFP8, widened exactly to bf16 at
  load; optional, below), and the vision tower from the pack's own shard.

Both kernels are checked against TensorFold's reference EXL3 projection (`tf-exl3-rows-bench`): every output within
one bf16 rounding of it.

## Checked

On the pinned engine, through [recipe-harness](https://github.com/BobClawblaw/recipe-harness) (target
`minimax-m31-exl3-m5`); the runs are in `evidence/`.

- **Fidelity** (`evidence/kl`). 2,406 tokens of a three-turn chat (prose, code, math, German, Chinese, JSON), teacher-forced in
  16-row windows, against the release's own logits (`tools/zig/minimax_m3_ref.py` in TensorFold, its MXFP8 and NVFP4
  decoded exactly): KL(release || served) mean 0.0741, median 0.0072, top-1 agreement 90.57%. The CUDA backend on the
  same pack measures 0.0788 / 90.36% on the same tokens, and the two engines differ from each other by KL 0.0125. Prompt chunks (the tensor-unit path) match the windows: below the sparse-selection threshold
  (rows 1,000-1,405 after a 1,000-token prompt) KL 0.0135 against the release, 0.0012 against the windows.
- **Serving** (`quick` plan, `evidence/plain-quick-dbcb9d791`, and with drafts `dspark-quick-ac35cd022`): structured output 18/18 (json_object, json_schema, regex, choice, grammar; drafted ==
  plain 12/12), tools 5/5, seeded sampling 9/9 equal on repeat, soak 60/60, concurrent 12/12, streaming 4/4,
  multi-turn 8/8, stop 3/3, thinking 10/10, max-tokens 3/3.
- **Prompt cache.** A turn resumed from a kept state equals the same request sent cold and after a restart; the
  resumed turn's first token in 1.27 s against 11.2 s cold (4,645 of 4,725 tokens kept).
- **Images** (`glm-vision` plan, `evidence/plain-vision-dbcb9d791`): shapes, text and counting right in PNG, JPEG and WebP; concurrent image requests
  12/16. The two-image case answers "red and green" for red then yellow, as mix256k and a transformers reference on
  the release do (the model's own near-tie). Video is not served (400), and image URLs are off unless the server is
  started with `--vision-urls` (send data URLs).
- **Long context.** A needle (`the secret launch code is PURPLE-HARBOR-7349`, two fifths into 858,865 characters of
  TensorFold's docs and sources) came back from a 227,727-token prompt, 504 s cold, 206.3 GiB wired at the peak
  (`evidence/needle-256k`).
- **Memory.** 158.9 GiB wired with the tower, 44-49 GiB left for caches under
  `load_limit_gb = 226`: 383,360 tokens fit; the recipe serves 262,144, two streams sharing it.

## Measured

Greedy, 256-token replies, the bf16 key/value cache, the harness's ruler (prose / code / structured / list):

**Decode**, tokens a second, all streams together:

| | streams | prose / code / structured / list | first token |
|---|---:|---|---|
| plain | 1 | 19.5 / 19.5 / 19.5 / 19.5 | 1.48-1.57 s |
| plain | 2 | 24.2 / 23.8 / 24.2 / 23.6 | 2.22-2.34 s |
| DSpark drafts | 1 | 17.6-18.6 / 19.8-20.3 / 17.7-17.8 / 18.5-18.8 | 1.51-1.62 s |
| DSpark drafts | 2 | 22.9 / 21.3 / 21.8 / 23.6 | 2.28-2.41 s |

Plain: `evidence/plain-quick-dbcb9d791` and `plain-speed-dbcb9d791`. Drafts: `evidence/dspark-quick-ac35cd022` and
`dspark-speed-ac35cd022` (three one-stream rulers; 74-78% of drafts accepted, 1.38-1.46 tokens a round). The
plain runs are on the commit before the drafter's, whose plain path it leaves unchanged.

**Prompts**, cold (a fresh server, each prompt starting with words no earlier one did; `evidence/plain-prefill-dbcb9d791`):

| | 2k | 8k | 32k | 128k |
|---|---:|---:|---:|---:|
| thinking off | 434 tok/s (5.1 s) | 486 (17.2 s) | 509 (64.7 s) | 501 (261.8 s) |
| thinking on | 433 (5.1 s) | 486 (17.2 s) | 505 (65.2 s) | 483 (271.5 s) |

**Why it is slower than the MLX recipes.** A 3-bit EXL3 weight costs about 17 integer and half-precision operations
to decode (its 16-bit trellis window, the `mul1` hash, one fp16 fused multiply-add); a token touches about 24 billion
weights (routed experts 16 billion), so decode is bound by the GPU's arithmetic, not its memory: the 8-bit head runs
at 667 million weights a second. That is also why the pack's DSpark head does not pay here: its drafts are exact
(drafted == plain, 12/12 in the suite) and three in four are accepted, but each extra row in a verify window picks
its own four experts, whose 3-bit weights must be decoded too, so a round costs 74-79 ms against a plain step's 51.
The recipe serves plain (`drafts = false`); `drafts = true` turns the head on (about 5.3 GiB more: its 2.1 GiB shard and 3.2 GiB widened).

**Quality suites were not run** on this build (the harness's quality plans run only when asked); the fidelity
numbers above are the quality evidence so far.

## Not supported

- Video (the tower reads images only on Metal).
- Contexts past 262,144 tokens are not offered by default (more fit, but the needle was checked at 227,727).
- Image URLs unless started with `--vision-urls`.

## Licence and credits

The weights are Mia-AiLab's quantization of MiniMaxAI's MiniMax-M3.1-Flash: follow
[MiniMaxAI/MiniMax-M3.1-Flash](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash)'s licence. The EXL3 format and
codebooks are [exllamav3](https://github.com/turboderp-org/exllamav3)'s. This recipe's tooling is MIT (LICENSE).

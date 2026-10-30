# MiniMax-M3.1-Flash EXL3 on one M5 Ultra (TensorFold, Metal)

MiniMaxAI's [MiniMax-M3.1-Flash](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash) (428B-A23B MoE, block-sparse
attention with the lightning indexer, a vision tower) served on one 256 GB Mac Studio (M5 Ultra) by
[TensorFold](https://github.com/ashhart/TensorFold)'s native Metal engine, from Mia-AiLab's published EXL3 pack
[`Mia-AiLab/MiniMax-M3.1-Flash-EXL3-EAQ-v1-3.06-bpw`](https://huggingface.co/Mia-AiLab/MiniMax-M3.1-Flash-EXL3-EAQ-v1-3.06-bpw)
**as published** (revision `c33e7a66`): no conversion; the served weights are the pack's bytes.

The pack is a lossy third-party quantization of the release (EAQ v1, 3.06 bits a weight on average: attention 6-bit,
dense and shared MLPs 5-bit, routed experts 3-bit, head 8-bit, `mul1` codebook; the fp16 index projections, the bf16
embedding and norms, the fp32 router, the DSpark draft head and the vision tower as in the release). It stays much
closer to the release than our MLX conversions: teacher-forced KL **0.079**, top-1 agreement **89.9%** (below).
The price is speed: EXL3's trellis codes take arithmetic to decode on a GPU, so this recipe decodes at about 33
tokens a second where [the mix256k recipe](https://github.com/BobClawblaw/minimax-m31-flash-mlx-m5-ultra) does 43 (drafted),
and reads prompts at about 500 tokens a second against its 1,480.

| | this recipe (EXL3 3.06 bpw) | mix256k (MLX affine, ours) |
|---|---|---|
| KL against the release, teacher-forced | **0.079** (89.9% top-1; 2,406 tokens) | 0.295 (78.4%; 1,024 tokens, other text) |
| Decode, one stream (prose) | 33.3-33.4 tok/s | 43.1 tok/s (DFlash2 drafts) |
| Prompts, cold, 8k / 32k / 128k | 487 / 508-509 / 503-504 tok/s | 1,483 / 1,491 / 1,379 tok/s |
| Weights held (with the tower) | 158.9 GiB | see its README |
| Window served | 262,144 tokens (383,360 fit) | 147,456 tokens |

Which to pick: this one stays much closer to the release (the KL above) and follows instructions better (IFEval
110 of 150 against 88); mix256k is faster and scored higher on MATH-500 (459 of 500 against 422). The other quality
suites land within a few questions of each other (Quality, below).

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
(the recipe pins `af3573193`), on top of the MiniMax-M3 Metal family (ashhart/TensorFold#1235) and the
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
  decoded exactly; on `c11fb68d6`, whose logits `af3573193` matches byte for byte): KL(release || served) mean 0.0787, median 0.0077, top-1 agreement 89.94%. The CUDA backend on the
  same pack measures 0.0788 / 90.36% on the same tokens. Prompt chunks (the tensor-unit path) match the windows: below the sparse-selection threshold
  (rows 1,000-1,405 after a 1,000-token prompt) KL 0.0135 against the release, 0.0012 against the windows (on
  `dbcb9d791`, whose prompt kernel is unchanged).
- **Serving** (`quick` plan, `evidence/plain-quick-af3573193`, and with drafts `dspark-quick-ac35cd022`): structured output 18/18 (json_object, json_schema, regex, choice, grammar; drafted ==
  plain 12/12), tools 5/5, seeded sampling 9/9 equal on repeat, soak 60/60, concurrent 12/12, streaming 4/4,
  multi-turn 8/8, stop 3/3, thinking 10/10, max-tokens 3/3.
- **Prompt cache.** A turn resumed from a kept state equals the same request sent cold and after a restart; the
  resumed turn's first token in 1.26 s against 11.2-11.4 s cold (4,645 of 4,725 tokens kept).
- **Images** (`glm-vision` plan, `evidence/plain-vision-af3573193`): shapes, text and counting right in PNG, JPEG and WebP; concurrent image requests
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
| plain | 1 | 33.3-33.4 / 33.3-33.4 / 33.3-33.5 / 33.3-33.5 | 1.45-1.56 s |
| plain | 2 | 43.3 / 43.1 / 42.1 / 42.2 | 2.18-2.31 s |
| DSpark drafts | 1 | 28.2-29.6 / 40.1 / 31.1-31.4 / 32.2-32.4 | 1.49-1.58 s |
| DSpark drafts | 2 | 37.0 / 38.1 / 34.1 / 41.5 | 2.24-2.36 s |

Plain: `evidence/plain-quick-af3573193` and `plain-speed-af3573193`. Drafts: `evidence/dspark-speed-af3573193`
(63-74% of drafts accepted, 1.61-1.88 tokens a round) and, for exactness, `dspark-quick-ac35cd022` (the drafter's
commit, before the faster decode kernel).

**Prompts**, cold (a fresh server, each prompt starting with words no earlier one did; `evidence/plain-prefill-af3573193`):

| | 2k | 8k | 32k | 128k |
|---|---:|---:|---:|---:|
| thinking off | 442 tok/s (5.0 s) | 487 (17.2 s) | 509 (64.7 s) | 504 (260.6 s) |
| thinking on | 439 (5.1 s) | 488 (17.1 s) | 508 (64.9 s) | 503 (260.8 s) |

**Why it is slower than the MLX recipes.** Every EXL3 weight is decoded on the fly: its 16-bit trellis window (a lane
takes eight consecutive weights from one run of the stream, bitfield ops fixed per width), the `mul1` hash and a
lookup in a threadgroup table of its 1,021 values, about a dozen operations; a threadgroup rotates its input rows once
and shares them. A token touches about 24 billion weights (routed experts 16 billion), so decode is bound by the
GPU's arithmetic more than its memory: the experts decode about 1.08 billion weights a millisecond. Of a 30.6 ms step
the experts take about 15 ms and the rest (attention projections, attention, the head) about 15.5. The first kernel ran 52 ms
(19.5 tokens a second); the engine branch's commit messages say what each change bought.

**Drafts.** The pack's DSpark head is exact (drafted == plain, 12/12 in the suite) and helps code (40.1 against 33.4
tokens a second, one stream), but costs a little on prose (28.2-29.6 against 33.4): each extra row in a verify window
picks its own four experts, whose 3-bit weights must be decoded too, so a round costs 54-57 ms against a plain step's
30. The recipe serves plain (`drafts = false`); `drafts = true` turns the head on (about 5.3 GiB more: its 2.1 GiB
shard and 3.2 GiB widened), worth it for code.

**TensorFold leaderboard** (community-v1, the board's own suite on the public pack, engine `af3573193` with its
DSpark head, which the benchmark turns on): [result d853e54d](https://tensorfold.dev/benchmarks?result=d853e54d-6f51-4ff5-9bd8-5bed22c6605d).

**Quality**, thinking off (the harness's `dsv41-quality-off` suite, the same suite code as mix256k's run), on the
pinned engine without drafts. The run was stopped before MATH-500 to free the Mac; MATH-500 then ran on its own on
the same binary (`evidence/quality-off-af3573193`, `evidence/math500-af3573193`). mix256k's column is its pinned
engine's run, from [its README](https://github.com/BobClawblaw/minimax-m31-flash-mlx-m5-ultra):

| | this recipe | mix256k |
|---|---:|---:|
| IFEval, loose (150) | **110** (0.733; strict 104) | 88 (0.587; strict 77) |
| tools (60) | 57 | 55 |
| JSON (30) | 30 | 30 |
| MMLU (342) | 283 (0.827) | 280 (0.819) |
| GSM8K (250) | 241 (0.964) | 246 (0.984) |
| MGSM (240) | 197 (0.821) | 202 (0.842) |
| MATH-500 | 422 (0.844) | **459** (0.918) |

IFEval is the clear gain. MATH-500 is the clear loss, and it is mostly replies without a final answer: of this
recipe's 78 misses, 57 give no answer the grader can read (mix256k: 18 of its 41), while wrong answers are about
even (21 against 23). The gap grows with difficulty (level 5: 98 of 134 against 112). GSM8K, MGSM, MMLU, tools and
JSON differ by five questions or fewer, within what sets this size swing. No thinking-on quality run was made.

## Not supported

- Video (the tower reads images only on Metal).
- Contexts past 262,144 tokens are not offered by default (more fit, but the needle was checked at 227,727).
- Image URLs unless started with `--vision-urls`.

## Licence and credits

The weights are Mia-AiLab's quantization of MiniMaxAI's MiniMax-M3.1-Flash: follow
[MiniMaxAI/MiniMax-M3.1-Flash](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash)'s licence. The EXL3 format and
codebooks are [exllamav3](https://github.com/turboderp-org/exllamav3)'s. This recipe's tooling is MIT (LICENSE).

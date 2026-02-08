# BoardDiffusion: Generative Diffusion Models for LED Climbing Board Problems

## Project Spec v2.1

---

## Overview

Build a discrete masked diffusion model that learns the distribution of boulder problems on LED climbing training boards and generates novel climbs conditioned on difficulty grade, wall angle, board size, quality, and classic status. The system supports **multiple boards** — Tension Board 2, Kilter Board, and MoonBoard — via a shared model architecture with board-specific vocabularies and embeddings. Each board uses the same Aurora Climbing SQLite schema, the same `frames` encoding, and the same training pipeline, differing only in hold placements, coordinate geometry, role definitions, and available board sizes.

### Supported Boards

| Board | BoardLib name | Adjustable angle | Board sizes | Grade range | Database size (est.) | Notes |
|---|---|---|---|---|---|---|
| **Tension Board 2** | `tension` | 0–65° | 10×8, 12×8, 12×10, 12×12 | V0–V14+ | ~20K–80K climbs | Mirror + spray layouts; wood + dual-tex holds |
| **Kilter Board** | `kilter` | 0–70° | 7×10, 8×12, 12×12, 16×12 | V0–V14+ | ~160K+ climbs | Largest database; Original + HW Fullride layouts |
| **MoonBoard** | `moonboard` | Fixed 25° or 40° | 8×11 (standard) | V3/V4–V14+ | ~50K–100K climbs | Multiple hold sets (2016, 2017, 2019, 2024); smallest/most crimpy holds |

All three boards use Aurora Climbing's software infrastructure and the same SQLite database schema, making a unified pipeline possible.

### Why Tokens Over Matrices

Climbs on all boards are sparse unordered sets of 4–20 holds placed on a grid of ~150–350 valid positions (2–10% occupancy). A matrix representation wastes ~95% of model capacity predicting "empty." A token sequence representation only represents active holds, giving the model direct control over hold count, naturally handling the set-like structure, and fitting well-established discrete diffusion methods (MDLM, MaskGIT). Spatial relationships are captured through coordinate embeddings in the transformer's attention rather than convolutions on a tiny grid.

### Architecture: Per-Board Models with Shared Code

Each board gets its **own trained model** with its own vocabulary (placement set, coordinate geometry, role definitions). The codebase is fully shared — only the config file changes between boards. This is the right tradeoff because:

- Board vocabularies are incompatible (different hold positions, different placement IDs)
- Grade distributions and climbing styles differ substantially between boards
- Individual models are small enough (~1–10M params) that training all three is cheap
- A shared codebase with board-parameterized configs avoids code duplication

---

## Stage 1: Data Pipeline

### 1.1 Database Acquisition

Use BoardLib to download and sync the SQLite database for any supported board:

```bash
pip install boardlib

# Download each board's database
boardlib database tension data/tension.sqlite
boardlib database kilter data/kilter.sqlite
boardlib database moonboard data/moonboard.sqlite
```

Each command extracts the database from the board's Android APK and syncs shared tables via the Aurora Climbing API. The databases share an identical schema but contain different climbs, placements, and board geometry.

### 1.2 Key Tables (Shared Schema)

All Aurora-based boards use the same database schema:

| Table | Purpose | Key Columns |
|---|---|---|
| `climbs` | All user-created boulder problems | `uuid`, `name`, `frames`, `layout_id`, `setter_id`, `is_listed`, `is_draft` |
| `climb_stats` | Aggregate stats per climb × angle | `climb_uuid`, `angle`, `ascent_count`, `display_difficulty`, `difficulty_average`, `quality_average`, `benchmark_difficulty` |
| `placements` | Maps placement IDs → physical holes | `id`, `hole_id`, `set_id`, `hold_id` |
| `holes` | X/Y coordinates for each hole on the board | `id`, `x`, `y`, `mirrored_hole_id` |
| `holds` | Hold metadata | `id`, `name`, `set_id` |
| `placement_roles` | Role types for hold usage | `id`, `full_name` |
| `difficulty_grades` | Numeric difficulty → V-grade | `difficulty`, `boulder_name` |
| `layouts` | Board layout configurations | `id`, `product_id`, `name`, `is_mirrored` |
| `product_sizes` | Board dimensions | `id`, `edge_left`, `edge_right`, `edge_bottom`, `edge_top` |
| `product_sizes_layouts_sets` | Valid layout-set combinations | `id`, `product_size_id`, `layout_id`, `set_id` |

### 1.3 Frame Encoding (Shared Format)

All Aurora boards store holds in the same `frames` string format:

```
p{placement_id}r{role_id}p{placement_id}r{role_id}...
```

For example: `p1083r15p1117r15p1164r12p1185r12p1233r13p1282r13` encodes 6 holds with their roles. The actual `placement_id` and `role_id` values differ between boards, but the format is identical.

### 1.4 Role Definitions Per Board

Each board uses the same role semantics but with different LED colors:

| Role | Tension TB2 | Kilter | MoonBoard | Semantic |
|---|---|---|---|---|
| Start | Green | Green | Green | Starting hand holds |
| Middle | Blue | Cyan | Blue | Intermediate holds |
| Finish | Red | Magenta | Red | Top-out / finish holds |
| Foot-only | Magenta | Yellow | — | Feet only, no hands |

> **Note:** MoonBoard uses a fixed "kickboard" of footholds that are always available rather than explicitly assigned foot-only holds. The number of roles `V_role` may therefore be 3 for MoonBoard and 4 for Tension/Kilter. The pipeline discovers `V_role` dynamically from the `placement_roles` table.

### 1.5 Token Sequence Representation

#### 1.5.1 Vocabulary Construction

Each hold in a climb is a (placement_id, role_id) pair. We use a **factorized token** representation where each hold position in the sequence carries two components:

1. **Placement token:** which physical position on the board the hold occupies. The vocabulary size `V_pos` equals the number of valid placements for the target board + layout + hold set. This varies by board (expect ~150–350).
2. **Role token:** the hold's role. The vocabulary size `V_role` is discovered from the database (typically 3–4).

These are represented as a **joint token** by flattening into a single vocabulary:

```
token_id = placement_index * V_role + role_index
```

Total vocabulary size: `V = V_pos * V_role + 2` (add MASK and PAD special tokens).

| Token | ID | Meaning |
|---|---|---|
| `(placement_i, role_j)` | `i * V_role + j` | Hold at placement i with role j |
| MASK | `V_pos * V_role` | Masked / unknown (used during diffusion) |
| PAD | `V_pos * V_role + 1` | Padding for variable-length sequences |

#### 1.5.2 Coordinate Features

Each placement has associated spatial coordinates from the `holes` table. These are used as continuous features — not as discrete tokens — so the model can reason about spatial relationships:

1. Query `placements → holes` to get `(x, y)` for each placement.
2. Normalize coordinates to `[0, 1]` relative to the board extents (from `product_sizes`).
3. For each token in the sequence, look up its `(x_norm, y_norm)` and inject as a positional feature (see §2.4).

MASK and PAD tokens receive a learned coordinate embedding (or zeros).

#### 1.5.3 Canonical Ordering

Climbs are unordered sets of holds, but the transformer operates on sequences. Impose a deterministic canonical ordering:

1. **Primary sort:** by `y` coordinate, ascending (bottom to top — the climbing direction).
2. **Secondary sort:** by `x` coordinate, ascending (left to right) for ties.

This ordering is consistent across all climbs and gives the model a weak spatial prior.

> **Note on permutation invariance:** While the canonical ordering provides a fixed sequence, the masked diffusion process naturally handles order-invariance. At any noise level, the model sees a partially revealed set of holds and must predict the masked ones regardless of which positions were revealed first.

#### 1.5.4 Sequence Format

Each climb becomes a fixed-length sequence of `L_max` tokens (pad shorter climbs):

```
[hold_0, hold_1, ..., hold_k, PAD, PAD, ..., PAD]
 |___________________________|  |_______________|
        k active holds           L_max - k pads
```

Set `L_max` per board to accommodate the longest climbs with headroom (e.g., `L_max = 24` or `32` — determine empirically from each board's hold count distribution).

#### 1.5.5 Layout, Hold Set, and Board Size

Each board supports multiple layouts, hold sets, and **physical board sizes**. Smaller sizes are strict spatial crops of the largest size — a 10×8 Tension board has a subset of the placements available on the 12×12.

**Available sizes:**

| Board | Available sizes | Notes |
|---|---|---|
| Tension TB2 | 10×8, 12×8, 12×10, 12×12 | All same hold set; smaller boards crop outer placements |
| Kilter | 7×10, 8×12, 12×12, 16×12 | Original layout; HW Fullride is a separate layout |
| MoonBoard | 8×11 (standard), 8×12 | Essentially one size with minor variants |

**Vocabulary strategy:** Build the placement vocabulary from the **largest available board size** for the chosen layout. This is a superset of all smaller sizes' placements. Climbs from any size can be represented in this vocabulary. At generation time, a **size-specific logit mask** prevents the model from placing holds outside the target size's boundaries.

**Size determination per climb:** Each climb is set on a specific board installation, but the `climbs` table does not directly store board size. Instead, determine which sizes a climb is compatible with by checking whether all its placements fall within each size's boundaries:

```python
def get_compatible_sizes(climb_placements, size_placement_sets):
    """Return list of board sizes this climb fits on."""
    return [size for size, valid_placements in size_placement_sets.items()
            if climb_placements.issubset(valid_placements)]
```

A climb using only central placements is compatible with all sizes. A climb using edge placements may only be compatible with the largest size. Tag each climb with its **minimum compatible size** (the smallest board it fits on).

**Size-specific placement masks:** For each board size, precompute a binary mask over the full vocabulary indicating which tokens (placement-role pairs) are valid:

```python
# size_logit_mask[size] = BoolTensor of shape (V - 1,)
# True for tokens whose placement falls within that size's boundaries
for size_id, (edge_left, edge_right, edge_bottom, edge_top) in sizes.items():
    valid = []
    for placement_idx in range(V_pos):
        x, y = placement_coords[placement_idx]
        valid.append(edge_left <= x <= edge_right and edge_bottom <= y <= edge_top)
    # Expand to cover all roles for each placement, plus MASK token
    size_logit_mask[size_id] = expand_to_vocab(valid, V_role)
```

The config file specifies which `layout_id` and `set_id` to use. The pipeline automatically resolves valid placements for the chosen combination via `product_sizes_layouts_sets` and builds the size masks from `product_sizes`.

#### 1.5.6 Filtering Criteria

- Only include climbs where `is_listed = 1` and `is_draft = 0`
- Only include climbs with at least 1 ascent (`ascent_count >= 1` in `climb_stats`)
- Filter to the configured `layout_id` and hold set for consistent placement vocabulary
- Parse the `frames` string and resolve each `placement_id`; discard climbs with placements outside the target layout/set
- Discard climbs with more holds than `L_max`
- Tag each climb with its minimum compatible board size (see §1.5.5)

### 1.6 Conditioning Labels

For each climb × angle pair, extract the following scalar conditioning signals:

| Label | Source | Type | Notes |
|---|---|---|---|
| `difficulty` | `climb_stats.display_difficulty` | Continuous (float) | Numeric difficulty; map to V-grade via `difficulty_grades` |
| `angle` | `climb_stats.angle` | Discrete (int) | Wall angle in degrees |
| `board_size` | Inferred from placements (§1.5.5) | Discrete (categorical) | Target board size (e.g., "12x12", "10x8") |
| `is_classic` | `climb_stats.benchmark_difficulty IS NOT NULL` | Binary | Benchmark/classic status |
| `quality` | `climb_stats.quality_average` | Continuous | Average star rating |

`ascent_count` is stored per sample but used only for sample weighting — not as a conditioning label.

**Board-specific angle handling:**

- **Tension TB2:** Adjustable, many angles (0–65° in 5° increments). Angle is a meaningful conditioning signal.
- **Kilter:** Adjustable, wide range (0–70°). Angle is a meaningful conditioning signal.
- **MoonBoard:** Fixed angle (25° or 40° depending on setup). If the database contains only one angle, the angle conditioning label is effectively constant and can be omitted from guidance. The pipeline handles this automatically — if only one unique angle exists, angle conditioning is disabled.

**Board-specific size handling:**

- **Tension TB2:** Multiple sizes (10×8 through 12×12). Size is a meaningful conditioning signal that controls which placements are available.
- **Kilter:** Multiple sizes (7×10 through 16×12). Size is a meaningful conditioning signal.
- **MoonBoard:** Effectively one size. If only one unique size exists, size conditioning is disabled (same logic as angle).

### 1.7 Sample Weighting

Weight each training sample to upweight high-quality, popular climbs:

```
w_i = α · is_classic_i + β · log(1 + ascent_count_i) + γ · quality_i
```

where `α, β, γ` are tunable hyperparameters (suggested starting values: `α=2.0, β=1.0, γ=0.5`).

### 1.8 Output Format

Save the processed dataset as a single file per board (e.g., PyTorch `.pt`):

```python
{
    "board": str,                          # "tension", "kilter", or "moonboard"
    "layout_id": int,                      # which layout this dataset targets
    "set_ids": List[int],                  # which hold set(s) are included
    "sequences": Tensor[N, L_max],         # int64, token IDs (joint placement-role tokens, PAD-filled)
    "seq_lengths": Tensor[N],              # int, number of active holds per climb
    "difficulty": Tensor[N],               # float32 — conditioning label
    "angle": Tensor[N],                    # int — conditioning label
    "board_size": Tensor[N],              # int — conditioning label (index into size list)
    "is_classic": Tensor[N],              # bool — conditioning label
    "quality": Tensor[N],                 # float32 — conditioning label
    "ascent_count": Tensor[N],            # int — used for sample weighting only, NOT conditioning
    "sample_weight": Tensor[N],           # float32
    "climb_uuids": List[str],             # for traceability
    "vocab_meta": {                        # for reconstruction (board-specific)
        "board": str,
        "placement_index_to_id": Dict,     # model index → original placement_id
        "id_to_placement_index": Dict,     # original placement_id → model index
        "role_index_to_id": Dict,          # model role index → original role_id
        "id_to_role_index": Dict,          # original role_id → model role index
        "role_names": List[str],           # human-readable role names in index order
        "placement_coords": Tensor[V_pos, 2],  # normalized (x, y) per placement
        "V_pos": int,                      # number of valid placements (superset from largest size)
        "V_role": int,                     # number of roles (3 or 4)
        "V_total": int,                    # total vocab size including MASK/PAD
        "MASK_TOKEN": int,
        "PAD_TOKEN": int,
        "L_max": int,
        "board_sizes": List[str],          # available sizes in index order (e.g., ["10x8", "12x8", "12x12"])
        "size_logit_masks": Dict[int, Tensor],  # size_index → BoolTensor(V-1) of valid tokens
    }
}
```

---

## Stage 2: Discrete Masked Diffusion Model

### 2.1 Motivation

The climb is a sparse, variable-length, unordered set of (placement, role) tokens — a natural fit for **masked discrete diffusion** (MDLM / absorbing-state diffusion). The forward process replaces tokens with MASK, and the reverse process iteratively predicts and reveals tokens. A transformer backbone handles variable-length sets, pairwise spatial reasoning via attention over coordinate embeddings, and conditioning through cross-attention or prepended tokens.

The model architecture is **identical across boards** — only the vocabulary size, coordinate geometry, and number of roles change, all of which are parameterized via the board config and `vocab_meta`.

### 2.2 Forward Process: Masking

At each noise level `t ∈ [0, 1]`, corrupt the clean sequence `x_0` by independently replacing each active (non-PAD) token with MASK:

```
For each position i where x_0[i] ≠ PAD:
    x_t[i] = MASK    with probability γ(t)
    x_t[i] = x_0[i]  with probability 1 - γ(t)
```

PAD tokens are never masked — they remain PAD throughout. The noise schedule `γ(t)` is monotonically increasing:

```
γ(t) = 1 - cos(πt / 2)²    # cosine schedule (recommended)
γ(t) = t                     # linear schedule
```

At `t = 0`, the sequence is clean. At `t = 1`, all active tokens are MASK.

### 2.3 State Space

Each position in the sequence is in one of three categories:

| State | Meaning |
|---|---|
| Token `0 ... V-3` | Active hold: a specific (placement, role) pair |
| MASK | Unknown — to be predicted by the model |
| PAD | Inactive padding — not part of the climb |

The model predicts a distribution over the `V - 1` non-PAD tokens (all placement-role pairs + MASK) at each masked position. PAD is never predicted — it is structurally determined by the sequence length.

### 2.4 Model Architecture: Transformer

```
Input:  sequence of L_max tokens, each with token + coordinate + timestep features
Output: logits of shape (L_max, V - 1) over non-PAD vocabulary at each position
```

**Token embedding layer:**

```python
class TokenEmbedding(nn.Module):
    def __init__(self, V_total, d_model, placement_coords, L_max):
        self.token_embed = nn.Embedding(V_total, d_model)
        self.coord_proj = nn.Linear(2, d_model)   # project (x, y) → d_model
        self.pos_embed = nn.Embedding(L_max, d_model)  # sequence position

    def forward(self, token_ids, t):
        tok = self.token_embed(token_ids)           # (B, L, d_model)
        coords = self.get_coords(token_ids)         # (B, L, 2) — lookup from placement_coords
        coord = self.coord_proj(coords)             # (B, L, d_model)
        pos = self.pos_embed(positions)             # (B, L, d_model)
        return tok + coord + pos
```

`V_total`, `placement_coords`, and `L_max` are all read from the board's `vocab_meta` at model construction time. The same `TokenEmbedding` class works for every board.

For each token, the model receives:
- **Token embedding:** learned embedding for each (placement, role) pair, MASK, and PAD
- **Coordinate embedding:** the normalized `(x, y)` board position projected to model dimension. MASK tokens get a learned default coordinate. PAD tokens get zeros.
- **Sequence position embedding:** learned embedding for position 0...L_max-1 in the canonical ordering

**Backbone:**

A standard transformer encoder (bidirectional self-attention):

| Parameter | Small (start here) | Full |
|---|---|---|
| `d_model` | 128 | 256 |
| `n_heads` | 4 | 8 |
| `n_layers` | 4 | 6–8 |
| `d_ff` | 512 | 1024 |
| Dropout | 0.1 | 0.1 |
| Parameters | ~1–2M | ~5–10M |

Start with the small variant for fast iteration (trains in minutes on MacBook Pro). Scale to full only if the small model underfits. Kilter's larger dataset may benefit from the full model sooner than Tension or MoonBoard.

Bidirectional attention is essential — unlike autoregressive generation, masked diffusion needs each position to attend to all other revealed tokens to make coherent predictions.

**Timestep conditioning:**

Embed `t` via sinusoidal encoding → MLP → inject via adaptive layer norm (AdaLN) at each transformer layer:

```python
t_embed = MLP(sinusoidal_encoding(t))   # (B, d_model)
# AdaLN (preferred, following DiT)
scale, shift = t_embed.chunk(2, dim=-1)
h = layer_norm(h) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
```

**Label conditioning:**

Embed each scalar label via a small MLP and combine:

```python
diff_embed = MLP_diff(difficulty)         # (B, d_cond)
angle_embed = MLP_angle(angle)            # (B, d_cond)  — disabled if single-angle board
size_embed = MLP_size(board_size)         # (B, d_cond)  — disabled if single-size board
classic_embed = MLP_classic(is_classic)   # (B, d_cond)
quality_embed = MLP_quality(quality)      # (B, d_cond)

y = MLP_combine(cat([diff_embed, angle_embed, size_embed, classic_embed, quality_embed]))  # (B, d_model)
```

Inject `y` the same way as `t` (add to AdaLN, or prepend as a conditioning token at position 0).

For boards with a fixed angle (e.g., MoonBoard at 40°), the angle MLP is omitted and its slot in the concatenation is filled with zeros. Same for boards with a single size. This is controlled by the `angle_conditioning` and `size_conditioning` flags in the board config.

**Attention mask:**

```python
attn_mask[i, j] = True   if token_ids[i] ≠ PAD and token_ids[j] ≠ PAD
attn_mask[i, j] = False  if token_ids[i] == PAD or token_ids[j] == PAD
```

**Output head:**

```python
logits = Linear(d_model, V - 1)  # predict over all non-PAD tokens
# Shape: (B, L_max, V - 1)
```

The output dimension `V - 1` is board-specific (determined by `vocab_meta["V_total"] - 1`).

### 2.5 Training Objective

At each training step:

1. Sample a clean climb `x_0` of length `k` with weight `w_i` (weighted sampling from the dataset).
2. Sample noise level `t ~ Uniform(0, 1)`.
3. Construct `x_t` by replacing each active token with MASK independently with probability `γ(t)`.
4. Run the model: `logits = f_θ(x_t, t, y)`.
5. Compute cross-entropy loss **only at masked positions**:

```
L = -Σ_{i : x_t[i] = MASK} log p_θ(x_0[i] | x_t, t, y)
```

PAD positions and unmasked positions are excluded from the loss.

### 2.6 Classifier-Free Guidance (CFG)

During training, randomly drop the conditioning vector `y` with probability `p_uncond = 0.1` (replace with a learned null embedding). This enables CFG at inference.

Each conditioning label can be independently masked:

| Label | Drop probability | Null value | Notes |
|---|---|---|---|
| `difficulty` | 0.1 | Learned null embedding | All boards |
| `angle` | 0.1 | Learned null embedding | Omitted for fixed-angle boards |
| `board_size` | 0.1 | Learned null embedding | Omitted for single-size boards |
| `is_classic` | 0.1 | Learned null embedding | All boards |
| `quality` | 0.1 | Learned null embedding | All boards |
| All labels jointly | 0.1 | All null | All boards |

> **Important:** Dropping `board_size` during CFG training means the model sometimes trains without knowing which placements are valid for the target size. This is intentional — it forces the model to learn spatial patterns that generalize across sizes. At inference, size is always specified and the logit mask (§2.7) enforces hard constraints.

At inference, apply guidance scale `s`:

```
logits_guided = (1 + s) · logits_cond - s · logits_uncond
```

Typical `s ∈ [1.0, 5.0]`.

### 2.7 Sampling (Reverse Process)

Use iterative unmasking over `T` steps (e.g., `T = 20–50`, sufficient for short sequences; generation takes <1 second on MacBook Pro):

1. **Initialize:** Choose a target sequence length `k` (sample from the training length distribution, or specify directly). Create a sequence of `k` MASK tokens followed by `L_max - k` PAD tokens.
2. **Load the size logit mask** for the target board size from `vocab_meta["size_logit_masks"]`. This is a boolean tensor of shape `(V - 1,)` that is `True` for tokens whose placement falls within the target size's physical boundaries (and `True` for the MASK token).
3. At each step, going from `t` to `t - 1/T`:
   a. Run the model to get logits at all MASK positions.
   b. Apply CFG.
   c. **Apply size logit mask:** set `logits[:, ~size_mask] = -inf` to zero out placements outside the target board size. This is a hard constraint — the model cannot place holds outside the board.
   d. Compute the number of tokens to unmask at this step: `n_unmask = round(k · (γ(t) - γ(t - 1/T)))`.
   e. **Confidence-based selection:** rank MASK positions by their max predicted probability (after masking). Unmask the top `n_unmask` positions.
   f. For each selected position: sample a token from the categorical distribution (with optional temperature).
   g. Replace the MASK at those positions with the sampled tokens. Fix them for all subsequent steps.
4. At `t = 0`, all `k` positions have been assigned tokens — all guaranteed to be valid for the target board size.

**Alternative sampling strategies:**

- **Random selection:** unmask a random subset instead of highest-confidence. Simpler but lower quality.
- **Annealed temperature:** start with high temperature (more diverse) and decay toward low temperature (more confident) over the diffusion steps.
- **Remasking (à la MaskGIT):** after unmasking, allow low-confidence tokens to be re-masked and re-predicted. Adds compute but improves coherence.

**Sequence length selection:**

The target length `k` can be:
- Sampled from the empirical length distribution (optionally conditioned on difficulty — harder climbs tend to have more holds)
- Specified by the user
- Predicted by a small auxiliary model conditioned on difficulty/angle

### 2.8 Post-processing & Validity Checks

After sampling, apply hard constraints:

1. **No duplicate placements:** if the same placement_id appears twice (with any role), keep the one predicted with higher confidence and re-sample the other.
2. **At least 1 start hold:** if none, reassign the role of the lowest activated hold to start.
3. **At least 1 finish hold:** if none, reassign the role of the highest activated hold to finish.
4. **Minimum hold count:** reject samples with fewer than 4 total holds.
5. **Maximum hold count:** reject samples with more than ~20 holds (configurable per board).
6. **Valid placements only:** guaranteed by the vocabulary (each token maps to a valid placement for the target board/layout). **Valid for target board size:** guaranteed by the size logit mask applied during sampling (§2.7).

### 2.9 Training Details

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 3e-4 with cosine decay |
| Warmup | 1000 steps |
| Batch size | 64–128 |
| Epochs | Until convergence (~100–300 depending on dataset size) |
| EMA | Exponential moving average of weights, decay 0.9999 |
| Precision | FP32 (MPS backend does not fully support BF16 mixed precision) |
| Weighted sampling | Use `sample_weight` for `WeightedRandomSampler` |
| Training hardware | MacBook Pro (Apple Silicon) via PyTorch MPS backend |

**MacBook Pro training notes:**

- Use `device = torch.device("mps")` for GPU acceleration on Apple Silicon.
- The model is ~5–10M parameters on sequences of length 24–32 — this fits comfortably in unified memory. Expect convergence in 1–2 hours per board.
- Start with the smaller architecture variant (`d_model=128, n_layers=4, n_heads=4, d_ff=512`, ~1–2M params) and scale up only if underfitting. The small variant trains in minutes and provides fast iteration on the data pipeline and diffusion logic.
- Stick with FP32 — the model is small enough that mixed precision provides negligible speedup and avoids MPS compatibility issues.
- If MPS causes issues (kernel gaps, numerical instability), fall back to CPU — at this model size, CPU training on Apple Silicon is still feasible (~5–10× slower, still under a day).
- The entire dataset fits in RAM; use `pin_memory=False` with MPS (unlike CUDA, MPS uses unified memory and does not benefit from pinned memory).
- Training all three boards sequentially takes ~3–6 hours total on MacBook Pro.

---

## Stage 3: Guided Generation

### 3.1 Generation Interface

Provide a Python API and CLI for generating climbs, parameterized by board:

```python
from board_diffusion import BoardDiffusionModel

# Load a board-specific model
model = BoardDiffusionModel.load("checkpoints/tension/best.pt")

climbs = model.generate(
    n_samples=10,
    difficulty=14,          # numeric difficulty (V5 ~ 14)
    angle=40,               # wall angle in degrees (ignored for fixed-angle boards)
    board_size="12x12",     # target board size — constrains valid placements
    is_classic=True,        # bias toward classic-quality
    quality=3.5,            # target quality rating
    guidance_scale=3.0,     # CFG strength
    temperature=0.8,        # sampling temperature
    n_holds=None,           # None = sample from distribution; int = fixed count
    n_steps=32,             # diffusion steps (fast on MacBook Pro)
)
# Returns: List of dicts with {tokens, frames_string, hold_list, metadata}
```

**CLI usage:**

```bash
# Generate 10 Kilter Board problems at V6, 40 degrees, on a 12x12 board
python generate.py --board kilter --difficulty 16 --angle 40 --size 12x12 --n 10

# Generate Tension problems for a 10x8 home wall
python generate.py --board tension --difficulty 14 --angle 40 --size 10x8 --n 10

# Generate MoonBoard problems (angle and size auto-detected from config)
python generate.py --board moonboard --difficulty 18 --n 10

# Generate Tension Board classics on the full-size board
python generate.py --board tension --difficulty 14 --angle 40 --size 12x12 --classic --n 10
```

### 3.2 Partial Conditioning

Users can specify any subset of conditions. Unspecified conditions use the null embedding (unconditional on that axis).

```python
# Only specify angle and difficulty
climbs = model.generate(difficulty=16, angle=45)

# Only specify angle — explore the grade distribution at that angle
climbs = model.generate(angle=40, n_samples=50)
```

### 3.3 Inpainting / Partial Specification

Allow users to fix certain holds and generate the rest:

```python
# Fix start holds, generate the remaining holds
fixed = [
    {"placement_id": 1083, "role": "start"},
    {"placement_id": 1117, "role": "start"},
]

climbs = model.generate(
    difficulty=14,
    angle=40,
    fixed_holds=fixed,     # these tokens are never masked during sampling
    n_holds=8,             # total holds including the 2 fixed ones
)
```

The fixed holds are placed into the sequence as unmasked tokens before sampling begins. The remaining `n_holds - len(fixed)` positions start as MASK and are iteratively revealed.

### 3.4 Output Conversion

Convert the generated token sequence back to the `frames` string format (shared across all Aurora boards):

```python
def tokens_to_frames(token_ids, vocab_meta):
    """Convert token ID sequence → Aurora frames string. Works for any board."""
    V_role = vocab_meta["V_role"]
    parts = []
    for tok in token_ids:
        if tok == vocab_meta["MASK_TOKEN"] or tok == vocab_meta["PAD_TOKEN"]:
            continue
        placement_index = tok // V_role
        role_index = tok % V_role
        placement_id = vocab_meta["placement_index_to_id"][placement_index]
        role_id = vocab_meta["role_index_to_id"][role_index]
        parts.append(f"p{placement_id}r{role_id}")
    return "".join(parts)
```

### 3.5 Evaluation Metrics

| Metric | Description |
|---|---|
| **Validity rate** | Fraction of samples passing all post-processing checks |
| **Hold count distribution** | Compare generated vs. real hold count histograms per grade |
| **Spatial density** | Compare heatmaps of hold positions: generated vs. real |
| **Grade calibration** | Train a grade predictor on real climbs; evaluate predicted grade of generated climbs vs. target |
| **Diversity** | Average pairwise Jaccard distance between generated climbs at same difficulty |
| **Novelty** | Fraction of generated climbs that don't exactly match any training climb |
| **Placement co-occurrence** | Compare pairwise hold co-occurrence matrices: generated vs. real |

---

## Stage 4: Visualization & Submission Interface

### 4.1 Web Interface

Build a single-page app (React or plain HTML/JS) that:

1. **Board selector:** choose between Tension, Kilter, or MoonBoard. This loads the corresponding board geometry, hold positions, and role color scheme.
2. **Displays the board** with hold positions rendered from the database geometry.
3. **Renders generated climbs** with color-coded holds matching each board's LED colors (see §1.4).
4. **Provides generation controls:**
   - Difficulty slider (grade range adapts to selected board)
   - Angle selector (hidden or fixed for MoonBoard; adjustable for Tension/Kilter)
   - Board size selector (dropdown of available sizes for the selected board; hidden for single-size boards)
   - Classic toggle
   - Quality slider
   - Guidance scale slider
   - Temperature slider
   - Hold count slider (or "auto")
   - "Generate" button → calls backend API
5. **Shows climb metadata:** predicted grade, hold count, generation parameters, board name.
6. **Allows manual editing:** click holds to toggle on/off or change role, then re-evaluate or submit.

### 4.2 Board Visualization

Render the board using SVG or Canvas, parameterized by board geometry:

```
For each placement in the target board/layout:
  - Look up (x, y) from holes table
  - Draw a circle at (x_scaled, y_scaled)
  - If this placement is active in the generated climb:
    - Color = board-specific role color (from role_colors config)
    - Size = larger
    - Label = optional hold name
  - Else:
    - Color = grey
    - Size = smaller
```

Overlay hold shape images if available (via `boardlib images {board} ...`). The SVG approach is preferred for spatial accuracy and interactivity.

### 4.3 Submission to Board Apps

Use the BoardLib Aurora API to submit generated climbs to any board's system:

```python
import boardlib.api.aurora as aurora

def submit_climb(board_name, username, password, frames_string, layout_id, angle, name):
    """Submit a generated climb. Works for tension, kilter, or moonboard."""
    session = aurora.login(board_name, username, password)
    token = session["token"]
    user_id = session["user_id"]

    result = aurora.save_climb(
        board=board_name,
        token=token,
        layout_id=layout_id,
        setter_id=user_id,
        name=name,
        description=f"Generated by BoardDiffusion",
        is_draft=False,
        frames=frames_string,
        angle=angle,
    )
    return result["climb_uuid"]
```

The saved climb will appear in the corresponding board app and can be sent to the LED board via Bluetooth.

### 4.4 Deep Links

After submission, construct a deep link to open the climb directly in the board's mobile app:

| Board | Deep link format |
|---|---|
| Tension TB2 | `tensionboard2://climb/{climb_uuid}` |
| Kilter | `kilterboard://climb/{climb_uuid}` |
| MoonBoard | `moonboard://climb/{climb_uuid}` |

Display as a QR code or clickable link in the web interface. Verify exact URL schemes from each app's documentation.

### 4.5 Backend API

A lightweight FastAPI server running locally on the MacBook:

| Endpoint | Method | Description |
|---|---|---|
| `/boards` | GET | List available boards with loaded models |
| `/generate` | POST | Generate climbs for a specific board (<1s per batch on MPS) |
| `/board-config/{board}` | GET | Return board geometry, valid positions, role definitions, color scheme |
| `/submit` | POST | Submit a climb to the board's API (requires auth) |
| `/evaluate` | POST | Run validity checks + grade prediction on a climb |

---

## Board Configuration

Each board is configured via a YAML file that parameterizes the entire pipeline:

```yaml
# configs/tension.yaml
board: tension
db_path: data/tension.sqlite
layout_id: 1                     # primary mirror layout — discover from database
set_ids: [1, 2, 3]              # hold sets to include — discover from database
angle_conditioning: true
angle_range: [0, 65]
size_conditioning: true
available_sizes: ["10x8", "12x8", "12x10", "12x12"]  # discovered from product_sizes
default_size: "12x12"            # largest size — used for vocabulary construction
L_max: 24
model_size: small                # "small" or "full"

# Board-specific visualization
role_colors:
  start: "#00FF00"
  middle: "#0000FF"
  finish: "#FF0000"
  foot_only: "#FF00FF"

deep_link_scheme: "tensionboard2://climb/"
```

```yaml
# configs/kilter.yaml
board: kilter
db_path: data/kilter.sqlite
layout_id: 1                     # original layout — discover from database
set_ids: [1]
angle_conditioning: true
angle_range: [0, 70]
size_conditioning: true
available_sizes: ["7x10", "8x12", "12x12", "16x12"]
default_size: "16x12"
L_max: 32                       # Kilter climbs can be longer

role_colors:
  start: "#00FF00"
  middle: "#00FFFF"
  finish: "#FF00FF"
  foot_only: "#FFFF00"

deep_link_scheme: "kilterboard://climb/"
```

```yaml
# configs/moonboard.yaml
board: moonboard
db_path: data/moonboard.sqlite
layout_id: 1                     # 2024 set at 40° — discover from database
set_ids: [1]
angle_conditioning: false        # fixed angle
default_angle: 40
size_conditioning: false         # effectively one size
available_sizes: ["8x11"]
default_size: "8x11"
L_max: 24

role_colors:
  start: "#00FF00"
  middle: "#0000FF"
  finish: "#FF0000"

deep_link_scheme: "moonboard://climb/"
```

> **Note on layout/set IDs:** The exact `layout_id` and `set_id` values must be discovered from each board's database by querying the `layouts`, `product_sizes_layouts_sets`, and `sets` tables. The config values above are placeholders. The `data/process.py` script prints all available layout/set combinations to help you choose.

---

## Project Structure

```
board-diffusion/
├── README.md
├── spec.md                          # This document
├── requirements.txt
├── configs/
│   ├── tension.yaml                 # Tension Board 2 config
│   ├── kilter.yaml                  # Kilter Board config
│   └── moonboard.yaml              # MoonBoard config
├── data/
│   ├── download.py                  # Download databases for all boards
│   ├── process.py                   # SQLite → token sequences (board-parameterized)
│   ├── dataset.py                   # PyTorch Dataset + WeightedRandomSampler
│   ├── tension.sqlite               # Downloaded databases (gitignored)
│   ├── kilter.sqlite
│   └── moonboard.sqlite
├── model/
│   ├── transformer.py               # Transformer encoder architecture (board-agnostic)
│   ├── embeddings.py                # Token, coordinate, timestep, label embeddings
│   ├── diffusion.py                 # Masking schedule, forward/reverse process
│   ├── conditioning.py              # Label embeddings, CFG logic
│   └── sampler.py                   # Confidence-based unmasking sampler
├── training/
│   ├── train.py                     # Training loop (takes --config flag)
│   ├── train_all.sh                 # Train all boards sequentially
│   └── evaluate.py                  # Metrics computation
├── generation/
│   ├── generate.py                  # CLI generation script (takes --board flag)
│   ├── postprocess.py               # Validity checks, constraint enforcement
│   └── convert.py                   # Token sequence ↔ frames string conversion
├── interface/
│   ├── server.py                    # FastAPI backend (serves all boards)
│   ├── submit.py                    # Aurora API submission logic (board-parameterized)
│   └── frontend/                    # React app
│       ├── App.jsx                  # Board selector + routing
│       ├── BoardView.jsx            # SVG board rendering (board-parameterized)
│       ├── Controls.jsx             # Generation controls (adapts to board)
│       └── ClimbCard.jsx            # Generated climb display
├── checkpoints/                     # Saved model weights (gitignored)
│   ├── tension/
│   ├── kilter/
│   └── moonboard/
└── processed/                       # Processed datasets (gitignored)
    ├── tension.pt
    ├── kilter.pt
    └── moonboard.pt
```

---

## Dependencies

```
# Core
torch>=2.0              # MPS backend requires torch >= 2.0; install via pip install torch
numpy
pandas
scipy

# Data
boardlib
sqlite3 (stdlib)

# Model
einops

# Interface
fastapi
uvicorn
jinja2

# Visualization
matplotlib
seaborn

# Config
pyyaml
wandb  # experiment tracking
```

> **macOS note:** Install PyTorch with `pip install torch torchvision` — the default pip package includes MPS support on Apple Silicon. No special CUDA builds needed.

---

## Usage Summary

```bash
# 1. Download all databases
python data/download.py --boards tension kilter moonboard

# 2. Process each board's data into token sequences
python data/process.py --config configs/tension.yaml
python data/process.py --config configs/kilter.yaml
python data/process.py --config configs/moonboard.yaml

# 3. Train models (or use train_all.sh)
python training/train.py --config configs/tension.yaml
python training/train.py --config configs/kilter.yaml
python training/train.py --config configs/moonboard.yaml

# 4. Generate climbs
python generation/generate.py --board tension --difficulty 14 --angle 40 --size 12x12 --n 10
python generation/generate.py --board kilter --difficulty 16 --angle 45 --size 8x12 --n 10
python generation/generate.py --board moonboard --difficulty 18 --n 10

# 5. Launch web interface (serves all boards)
python interface/server.py
```

---

## Open Questions & Extensions

1. **Incorporate hold shape information.** Each placement has a specific hold shape (crimp, pinch, sloper, jug). This can be added as a third factor in the token (placement × role × shape), as an auxiliary input feature per token, or as a separate prediction head.

2. **Multi-layout support per board.** Each board supports multiple layouts (e.g., Tension mirror vs. spray, Kilter Original vs. HW Fullride, MoonBoard 2017 vs. 2024). Currently each (board, layout) pair gets its own model. An extension could use layout-conditional models with layout-specific vocabularies and shared transformer weights.

3. **Cross-board foundation model.** Train a single model on all boards jointly. The transformer weights are shared; each board gets its own embedding and output layers. Board identity becomes a conditioning signal. This could enable transfer learning — a board with fewer climbs (Tension) could benefit from patterns learned on a board with more data (Kilter).

4. **Autoregressive hybrid.** Use the diffusion model to generate the set of holds, then optionally train a second autoregressive model to predict the intended climbing sequence (move order) given the set.

5. **Style transfer.** Condition on setter identity to generate climbs "in the style of" specific setters. Setter ID becomes an additional conditioning label with its own CFG dropout.

6. **Active learning with human feedback.** After generating and climbing problems, users rate them. Use these ratings to fine-tune the model (RLHF or DPO on climb quality).

7. **Symmetry augmentation.** For mirror layouts (Tension, MoonBoard), every climb has a mirrored counterpart. Use this as free data augmentation: swap each placement_id with its mirrored counterpart (from `holes.mirrored_hole_id`) and reverse the x-coordinate. Not applicable to asymmetric boards (Kilter Original).

8. **Relative spatial attention.** Instead of absolute coordinate embeddings, use pairwise distance features in the attention mechanism (relative position bias). Compute `Δx, Δy, dist` between each pair of holds and project into attention bias terms.

9. **Length prediction model.** Train a small auxiliary model per board to predict the appropriate number of holds given (difficulty, angle), so that at generation time the sequence length is automatically set.

10. **Decoy and Grasshopper boards.** BoardLib also supports Decoy and Grasshopper boards via the same Aurora API. Adding these requires only a new config YAML — no code changes.

# Optimization Notes - Persistent Memory

## Current State
- **Cycles**: 3845 (down from ~5800 at earlier session start)
- **Target**: ~1300 cycles (best known: 1363)
- **Gap**: Need ~2.9x reduction

## Architecture Constraints (Critical)
```
SLOT_LIMITS per cycle:
- alu: 12 (scalar ALU)
- valu: 6 (vector ALU)
- load: 2 (THE BOTTLENECK)
- store: 2
- flow: 1 (vselect lives here!)
```

## Cycle Breakdown Analysis
| Section | Bundles | Cycles | Notes |
|---------|---------|--------|-------|
| Pre-loop (rounds 0-2) | 759 | 759 | Special rounds, no gathers |
| Main loop (rounds 3-9) | 233 | 1631 | 233 × 7 iterations |
| Post-loop (rounds 10-15 + stores) | ~1455 | ~1455 | Mixed special + gather rounds |
| **Total** | 2447 | **3845** | |

## Key Findings

### 1. Load Bottleneck Math
- Gather rounds (3-10, 14-15): 10 rounds × 256 loads = 2560 loads
- At 2 loads/cycle = **1280 cycles minimum** just for gathers
- Current total is 3845, so ~2565 cycles are non-gather overhead

### 2. VALU Slot Underutilization
- Main loop has **69 VALU-only bundles** (no loads)
- These cycles waste 2 load slots each = 138 wasted load ops
- Large 34-bundle VALU block averages only 3.1 ops/bundle (max 6)

### 3. Successful Optimizations Applied
1. **multiply_add for hash**: Stages 0, 2, 4 use pattern `(a + const) + (a << shift)` = `a * multiplier + const`
   - Multipliers: 4097 (stage 0), 33 (stage 2), 9 (stage 4)
2. **Forest vector reuse**: Rounds 11-13 reuse v_node_shared, v_node1/2, v_f3-6 from rounds 0-2
3. **Cross-round overlap**: Round 13's idx finish overlaps with round 14's batch 0 loading

### 4. vselect is Flow-Bound
- Rounds 2 and 13 use vselect tree (6 vselects per pair)
- vselect is a FLOW operation (1/cycle limit!)
- For 16 pairs: 96 vselect cycles per round
- This makes vselect rounds slower than expected

## What the Next Breakthrough Requires

### Option A: Full Unrolling (Most Promising)
Remove the main loop entirely and generate all 16 rounds as straight-line code.

**Benefits:**
- Cross-round pipelining: Start round N+1's gathers during round N's hash
- No loop control overhead
- Can specialize each round's code

**Estimate:** Could reduce main loop from 1631 to ~1000 cycles by overlapping rounds.

### Option B: Deeper Pre-loading (3-way instead of 2-way)
Currently pre-load 1 group ahead. Could pre-load 2 groups ahead.

**Challenge:** Need more scratch space for node buffers.

### Option C: vselect Elimination for Rounds 2/13
Instead of vselect tree, use arithmetic to compute forest index directly.

**Idea:** `forest_ptr + idx` is just pointer arithmetic. Load all 4 values, then use masking?

### Option D: Restructure for Better Packing
The 69 VALU-only cycles in main loop represent ~300 wasted load ops.
If we could restructure to add pre-pre-loading, we might save significant cycles.

## Code Structure Notes

### Round Types
| Rounds | Type | Indices | Load Strategy |
|--------|------|---------|---------------|
| 0, 11 | Single node | All 0 | 1 load + broadcast |
| 1, 12 | Two nodes | {1,2} | 2 loads + vselect |
| 2, 13 | Four nodes | {3,4,5,6} | 4 loads + vselect tree |
| 3-10, 14-15 | Full gather | Any | 256 scalar loads |

### Key Variables
- `v_idx[0..31]`: 32 vector index registers
- `v_val[0..31]`: 32 vector value registers
- `node_set_A/B/C`: Triple buffering for gather pre-loading
- `v_hash_consts[0..5]`: Vector hash constants
- `v_mult_4097/33/9`: multiply_add multipliers

## Quick Test Commands
```bash
# Correctness + cycles
python tests/submission_tests.py CorrectnessTests.test_kernel_correctness

# All tests
python tests/submission_tests.py

# Single performance test
python perf_takehome.py Tests.test_kernel_cycles
```

## Session History
- Session 1: Baseline optimizations, reached ~4300 cycles
- Session 2: multiply_add, forest reuse, round overlap → 3845 cycles
- Session 3: Loop unrolling implementation + analysis → 3817 cycles
  - Implemented full loop unrolling (rounds 3-9) in perf_takehome_optimized.py
  - Applied multiply_add for idx computations throughout
  - **Finding**: Loop unrolling alone saves only ~28 cycles (0.7%)
  - **Finding**: multiply_add optimization saves 0 cycles (already well-packed)
  - **Key insight**: Real bottleneck is lack of cross-round pipelining and bundle packing
  - **Path forward**: Need aggressive cross-round pipelining (est. 500-800 cycle savings)
  - Created OPTIMIZATION_SUMMARY.md with comprehensive optimization roadmap

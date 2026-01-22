# Optimization Notes - Persistent Memory

## Current State
- **Cycles**: 3628 (down from 3845 at session start, 3633 after previous session)
- **Target**: ~1300 cycles (best known: 1363)
- **Gap**: Need ~2.8x reduction
- **Progress this session**: 3633 → 3628 (5 cycles from Round 14-15 pipelining)

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

## Detailed Bundle Analysis (at 3628 cycles)

### Bundle Composition
| Type | Count | % of Total |
|------|-------|-----------|
| Load+VALU | 489 | 20.4% |
| Load-only | 86 | 3.6% |
| VALU-only | 1597 | **66.6%** |
| Neither | 226 | 9.4% |
| **Total** | 2398 | 100% |

### VALU Ops Distribution
| VALU ops/bundle | Count | % |
|-----------------|-------|---|
| 0 | 312 | 13.0% |
| 1 | 159 | 6.6% |
| 2 | 697 | 29.1% |
| 3 | 329 | 13.7% |
| 4 | 757 | 31.6% |
| 5 | 82 | 3.4% |
| 6 (max) | 62 | 2.6% |

### Resource Utilization
- **VALU utilization**: 44.1% (6350 ops / 14388 capacity)
- **Load utilization**: Poor - 1597 VALU-only bundles waste 3194 load slots
- **Flow utilization**: 302 ops across all bundles

### Bottleneck Analysis
- Total VALU ops: 6350 (needs 1058 cycles at 6/cycle)
- Total load ops: 1130 (needs 565 cycles at 2/cycle)
- Total flow ops: 302 (needs 302 cycles at 1/cycle)
- **Theoretical minimum**: max(1058, 565, 302) ≈ **1058 cycles**
- **Current**: 3628 cycles = **3.4x theoretical**

### Why 1363 Cycles is Achievable
Best known solution (1363 cycles) achieves near-theoretical VALU packing:
- 1363 × 6 = 8178 VALU capacity, needs 6350 ops = 78% utilization
- Must have near-perfect load+VALU overlap
- Minimal VALU-only bundles

### Key Insight
The 1597 VALU-only bundles (67% of code!) are the primary waste. These represent cycles where:
- Load slots are empty (could be doing useful loads)
- VALU capacity is partially utilized
- The structural bottleneck is group-by-group processing with VALU-only tails

## Session History
- Session 1: Baseline optimizations, reached ~4300 cycles
- Session 2: multiply_add, forest reuse, round overlap → 3845 cycles
- Session 3: Cross-round pipelining in main loop and Round 10 → 3633 cycles
  - Applied multiply_add to Group 7 in main loop (-56 cycles)
  - Applied multiply_add to rounds 14-15 Group 7 (-16 cycles)
  - Restructured Group 0 to use pre-loaded nodes (-120 cycles)
  - Group 7 now pre-loads next round's Group 0 nodes during hash computation
  - Applied same optimization to Round 10's Group 0 (-20 cycles)
  - Total improvement: 212 cycles (5.5% reduction)
- Session 4 (current): Round 14-15 pipelining, detailed analysis
  - Round 14 Group 7 pre-loads Round 15 Group 0 batch 0 (-5 cycles)
  - Identified 67% VALU-only bundles as main waste
  - Theoretical minimum is ~1058 cycles based on VALU ops

# Performance Optimization Summary

## Current Status

### Baseline Metrics
- **Original implementation**: 3,845 cycles
- **After loop unrolling** (in perf_takehome_optimized.py): 3,817 cycles
- **After multiply_add** (in perf_takehome_optimized.py): 3,817 cycles
- **Current improvement**: 28 cycles (0.7%)
- **Target**: ~1,300-1,500 cycles (best known: 1,363 cycles)
- **Gap**: Need 2.5-2.9x reduction (~2,300-2,500 cycles to save)

### Files Status
- `perf_takehome.py` - Original version (2,268 lines, 3,845 cycles)
- `perf_takehome_optimized.py` - Loop unrolled + multiply_add (5,484 lines, 3,817 cycles)
- `perf_takehome_unrolled.py` - Loop unrolled only (5,758 lines)
- `perf_takehome_v2.py` - Backup of original

## Bottleneck Analysis

### 1. Fundamental Load Limit: 1,280 cycles
- Gather rounds (3-10, 14-15) need **2,560 loads minimum**
- At 2 loads/cycle = **1,280 cycles unavoidable**
- This is the theoretical minimum for gather operations

### 2. Current Overhead: 2,565 cycles (3,845 - 1,280)
Breaking down the 3,845 total cycles:
- Pre-loop (rounds 0-2): ~759 cycles
- Main loop (rounds 3-9): ~1,631 cycles
- Post-loop (rounds 10-15 + stores): ~1,455 cycles

### 3. Target Overhead: ~80-220 cycles
To reach 1,300-1,500 cycles:
- 1,500 cycles = 1,280 (loads) + 220 (overhead)
- 1,363 cycles = 1,280 (loads) + 83 (overhead)
- **Current overhead is 30x higher than optimal!**

## Optimizations Applied

### ✅ Completed

#### 1. Loop Unrolling (Minor Impact: ~28 cycles saved)
- **What**: Removed rounds 3-9 loop, replicated body 7 times
- **Result**: Eliminated 3 loop control instructions × 7 iterations
- **Issue**: Didn't include aggressive cross-round pipelining
- **Location**: perf_takehome_optimized.py

#### 2. multiply_add Optimization (No Impact: 0 cycles saved)
- **What**: Replaced `idx = idx<<1; idx = idx+offset` with `multiply_add(idx, idx, 2, offset)`
- **Result**: Reduced 4 ops to 3 ops, but no bundle savings (operations were already packed)
- **Issue**: Cycle count is determined by bundles, not individual operations
- **Conclusion**: multiply_add is cleaner but doesn't reduce cycles unless it enables better packing

## Remaining Optimizations (High Impact)

### 🔴 Priority 1: Aggressive Cross-Round Pipelining (Est. 500-800 cycles)

**Current Problem**: Each round completes fully before the next begins. This wastes cycles because:
- Round N Group 7: Finishing hash/idx leaves load slots empty
- Round N+1 Group 0: Starting gathers leaves VALU slots empty

**Solution**: Overlap rounds by starting Round N+1 Group 0 gathers DURING Round N Group 7 completion.

**Implementation Pattern**:
```python
# Round N Group 7 - Current (wasteful):
bundle1: {valu: [hash ops]}           # 2 load slots wasted
bundle2: {valu: [hash ops]}           # 2 load slots wasted
bundle3: {valu: [idx ops]}            # 2 load slots wasted

# Round N+1 Group 0 - Current (wasteful):
bundle4: {alu: [address ops]}         # 6 VALU slots wasted
bundle5: {load: [gather ops]}         # 6 VALU slots wasted

# Optimized (packed):
bundle1: {valu: [round N hash], alu: [round N+1 addresses]}
bundle2: {valu: [round N hash], load: [round N+1 gathers]}
bundle3: {valu: [round N idx], load: [round N+1 gathers]}
```

**Where to Apply**:
- Transition from Round 2 → Round 3
- Between all gather rounds (3→4, 4→5, 5→6, 6→7, 7→8, 8→9, 9→10)
- Transition from Round 13 → Round 14

**Expected Savings**: 10-15 cycles per transition × 10 transitions = 100-150 cycles direct savings, plus better pipelining throughout = **500-800 cycles total**

### 🟡 Priority 2: Maximize Bundle Packing (Est. 300-500 cycles)

**Current Problem**: Many bundles don't use all available slots:
- Slot limits: 12 ALU, 6 VALU, 2 load, 2 store, 1 flow
- Current utilization: Often only 2-4 VALU, 8 ALU, 1 load per bundle

**Solution**: Pack more operations per bundle to reduce total bundles.

**Strategies**:
1. **Compute addresses for multiple batches in parallel** (use all 12 ALU slots):
   ```python
   # Current (12 ALU slots, processes 1 batch):
   {alu: [(+, addr[0], forest_p, idx+0), (+, addr[1], forest_p, idx+1), ...]}

   # Optimized (12 ALU slots, processes 1.5 batches):
   {alu: [(+, addrA[0], forest_p, idxA+0), ..., (+, addrB[0], forest_p, idxB+0), ...]}
   ```

2. **Process hash stages for multiple batches in parallel** (use all 6 VALU slots):
   ```python
   # Current (6 VALU slots, processes 2 batches):
   {valu: [(op1, tmp1A, valA, c1), (op3, tmp2A, valA, c2),
           (op1, tmp1B, valB, c1), (op3, tmp2B, valB, c2)]}

   # Optimized (6 VALU slots, processes 3 batches):
   {valu: [(op1, tmp1A, valA, c1), (op3, tmp2A, valA, c2),
           (op1, tmp1B, valB, c1), (op3, tmp2B, valB, c2),
           (op1, tmp1C, valC, c1), (op3, tmp2C, valC, c2)]}
   ```

3. **Dual-stream loading** (use both load slots):
   - Already partially implemented with addr_A and addr_B
   - Extend to more sections

**Expected Savings**: Reducing from 2,447 bundles to ~2,100 bundles = **300-500 cycles**

### 🟢 Priority 3: Optimize Hash Stages (Est. 100-200 cycles)

**Current State**: Hash stages 0, 2, 4 use multiply_add. Stages 1, 3, 5 use 3-op pattern.

**Opportunities**:
1. Check if stages 1, 3, 5 can use multiply_add
2. Reorder hash operations to better pack with loads/address computation
3. Consider computing hash for multiple batches concurrently

**Expected Savings**: **100-200 cycles**

### 🔵 Priority 4: Reduce vselect Overhead (Est. 50-100 cycles)

**Current State**: Rounds 2 and 13 use vselect trees (96 vselects each = 192 cycles total due to 1 flow/cycle limit).

**Analysis**: The vselect approach is likely optimal. Alternative (gathers) would take 128 cycles per round vs current 96.

**Possible Optimization**: Better interleaving of vselect with other operations (already partially done).

**Expected Savings**: **50-100 cycles** (limited by flow bottleneck)

## Implementation Roadmap

### Phase 1: Cross-Round Pipelining (Highest Impact)
1. Start with Round 2 → Round 3 transition
2. In Round 2's final group, start Round 3 Group 0 address computation
3. Overlap Round 3 Group 0 loads with Round 2's final hash/idx operations
4. Verify correctness with tests
5. Measure cycle savings
6. Replicate pattern for all round transitions (3→4, 4→5, ..., 13→14)

### Phase 2: Bundle Packing
1. Audit all bundles for slot utilization
2. Identify bundles with <50% utilization (e.g., only 3 of 12 ALU slots used)
3. Reorganize to pack more operations per bundle
4. Focus on address computation (can use all 12 ALU slots) and hash (can use all 6 VALU slots)

### Phase 3: Hash Optimization
1. Analyze hash stages 1, 3, 5 for multiply_add opportunities
2. Reorder operations for better packing
3. Consider processing 3 batches per hash cycle instead of 2

### Phase 4: Polish
1. Review special cases (rounds 0-2, 11-13)
2. Optimize final store operations
3. Fine-tune any remaining bottlenecks

## Testing Strategy

### Correctness (Critical)
```bash
# Run after EVERY change
python tests/submission_tests.py CorrectnessTests.test_kernel_correctness
```

### Performance Benchmarking
```bash
# Run to measure cycles after each optimization
python perf_takehome.py Tests.test_kernel_cycles

# Full suite (includes all performance tiers)
python tests/submission_tests.py
```

### Validation
```bash
# Ensure tests/ folder is unchanged
git diff origin/main tests/

# Should show no modifications
```

## Key Insights

### Why Loop Unrolling Alone Didn't Help Much
- Loop control overhead was only 3 instructions × 7 iterations = 21 cycles
- The real overhead is in inefficient bundling and lack of cross-round pipelining
- Unrolling ENABLES cross-round pipelining but doesn't implement it

### Why multiply_add Alone Didn't Help
- Reduced operation count (4→3) but not bundle count
- Operations were already packed efficiently
- multiply_add is useful when it enables BETTER packing or FEWER bundles

### The Path to 1,363 Cycles
To match the best known implementation (1,363 cycles):
- Theoretical minimum (loads): 1,280 cycles
- Allowed overhead: 83 cycles
- Current overhead: 2,565 cycles
- **Need to reduce overhead by 97% through aggressive bundling and pipelining**

This requires:
1. Near-perfect slot utilization (avg 10+ ALU, 5+ VALU, 2 load per bundle)
2. Aggressive cross-round pipelining
3. Minimal wasted cycles between operations

## Recommendations

### Immediate Next Steps
1. **Implement cross-round pipelining** for Round 2 → Round 3 transition
   - This is the highest-impact optimization
   - Should save 50-80 cycles for this one transition
   - Provides pattern to replicate across all transitions

2. **Profile bundle utilization**
   - Count how many of each slot type is used per bundle
   - Identify low-utilization bundles (<50% of slots used)
   - These are optimization targets

3. **Measure incrementally**
   - After each change, run correctness tests
   - Measure cycle count to validate improvements
   - Document what worked and what didn't

### Tools Available
- **Trace visualization**: `python watch_trace.py` + watch_trace.html
- **Perfetto UI**: For visualizing instruction execution timeline
- **Submission tests**: Comprehensive correctness + performance validation

### Expected Timeline to 1,500 Cycles
- Phase 1 (Cross-round pipelining): 500-800 cycles saved → **~3,000 cycles**
- Phase 2 (Bundle packing): 300-500 cycles saved → **~2,500 cycles**
- Phase 3 (Hash optimization): 100-200 cycles saved → **~2,300 cycles**
- Phase 4 (Polish): 800-1,000 cycles saved → **~1,300-1,500 cycles** ✓

This is achievable but requires careful, incremental optimization with continuous validation.

## Critical Success Factors

1. **Preserve Correctness**: Test after EVERY change
2. **Measure Everything**: Don't assume optimizations work - measure them
3. **Focus on Bundling**: Cycle count = bundle count (when not bottlenecked)
4. **Think in Bundles**: Optimize for bundles, not individual operations
5. **Cross-Round Pipelining**: This is the highest-impact opportunity

## References

- [NOTES.md](claude/NOTES.md) - Session notes and analysis
- [CLAUDE.md](CLAUDE.md) - Project instructions and constraints
- [problem.py](problem.py) - Simulator implementation and reference kernel
- [tests/submission_tests.py](tests/submission_tests.py) - Official validation tests

---

**Status**: Loop unrolling and multiply_add optimizations completed. Cross-round pipelining (highest impact) is next priority.

**Current Bottleneck**: Inefficient bundling and lack of cross-round pipelining causing 2,565 cycles of overhead (vs target of ~83 cycles).

**Path Forward**: Implement aggressive cross-round pipelining and bundle packing to reduce overhead by 97%.

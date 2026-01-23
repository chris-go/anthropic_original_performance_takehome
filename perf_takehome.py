"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

Validate your results using `python tests/submission_tests.py` without modifying
anything in the tests/ folder.

We recommend you look through problem.py next.
"""

from collections import defaultdict
import random
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        # Simple slot packing that just uses one slot per instruction bundle
        instrs = []
        for engine, slot in slots:
            instrs.append({engine: [slot]})
        return instrs

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def add_bundle(self, bundle):
        """Add a pre-built instruction bundle (dict of engine -> list of slots)"""
        self.instrs.append(bundle)

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def preload_const(self, val, name=None):
        """Allocate and return address for a constant (load happens in init phase)"""
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.const_map[val] = addr
        return self.const_map[val]

    def build_hash(self, val_hash_addr, tmp1, tmp2, round, i):
        slots = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            slots.append(("alu", (op1, tmp1, val_hash_addr, self.scratch_const(val1))))
            slots.append(("alu", (op3, tmp2, val_hash_addr, self.scratch_const(val3))))
            slots.append(("alu", (op2, val_hash_addr, tmp1, tmp2)))
            slots.append(("debug", ("compare", val_hash_addr, (round, i, "hash_stage", hi))))

        return slots

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Optimized 2-way pipeline with maximum VLIW packing.
        Minimize cycles through aggressive operation combining.
        """
        num_batches = batch_size // VLEN  # 32 vector batches

        # Pre-allocate hash constants
        hash_consts = []
        for (op1, val1, op2, op3, val3) in HASH_STAGES:
            c1 = self.preload_const(val1)
            c2 = self.preload_const(val3)
            hash_consts.append((c1, c2))

        zero_const = self.preload_const(0)
        one_const = self.preload_const(1)
        two_const = self.preload_const(2)

        # Multiplier constants for multiply_add optimization
        # Stage 0: (a + const) + (a << 12) = a * (1 + 2^12) + const = a * 4097 + const
        # Stage 2: (a + const) + (a << 5) = a * (1 + 2^5) + const = a * 33 + const
        # Stage 4: (a + const) + (a << 3) = a * (1 + 2^3) + const = a * 9 + const
        mult_4097 = self.preload_const(4097)
        mult_33 = self.preload_const(33)
        mult_9 = self.preload_const(9)

        # Memory layout
        init_vars = ["rounds", "n_nodes", "batch_size", "forest_height",
                    "forest_values_p", "inp_indices_p", "inp_values_p"]
        for v in init_vars:
            self.alloc_scratch(v, 1)

        # Keep ALL idx and val vectors in scratch
        v_idx = [self.alloc_scratch(f"v_idx_{i}", VLEN) for i in range(num_batches)]
        v_val = [self.alloc_scratch(f"v_val_{i}", VLEN) for i in range(num_batches)]

        # Working space for 2 batches (used in special rounds)
        v_node_A = self.alloc_scratch("v_node_A", VLEN)
        v_node_B = self.alloc_scratch("v_node_B", VLEN)
        v_tmp1_A = self.alloc_scratch("v_tmp1_A", VLEN)
        v_tmp2_A = self.alloc_scratch("v_tmp2_A", VLEN)
        v_tmp1_B = self.alloc_scratch("v_tmp1_B", VLEN)
        v_tmp2_B = self.alloc_scratch("v_tmp2_B", VLEN)

        # All node values for gather-first approach (256 values = 32 batches * 8)
        v_nodes = [self.alloc_scratch(f"v_nodes_{i}", VLEN) for i in range(num_batches)]

        # Extra temps for 3-way hash parallelism
        v_tmp1_C = self.alloc_scratch("v_tmp1_C", VLEN)
        v_tmp2_C = self.alloc_scratch("v_tmp2_C", VLEN)

        v_one = self.alloc_scratch("v_one", VLEN)
        v_two = self.alloc_scratch("v_two", VLEN)
        v_zero = self.alloc_scratch("v_zero", VLEN)
        v_n_nodes = self.alloc_scratch("v_n_nodes", VLEN)

        # Scalar temps
        tmp1 = self.alloc_scratch("tmp1")
        round_counter = self.alloc_scratch("round_counter")
        num_rounds_s = self.alloc_scratch("num_rounds_s")
        ptr = self.alloc_scratch("ptr")

        # Addresses for gather loads
        addr_A = [self.alloc_scratch(f"addrA{i}") for i in range(VLEN)]
        addr_B = [self.alloc_scratch(f"addrB{i}") for i in range(VLEN)]

        # Vector hash constants
        v_hash_consts = []
        for i in range(len(HASH_STAGES)):
            vc1 = self.alloc_scratch(f"vhc1_{i}", VLEN)
            vc2 = self.alloc_scratch(f"vhc2_{i}", VLEN)
            v_hash_consts.append((vc1, vc2))

        # Vector multiplier constants for multiply_add optimization
        v_mult_4097 = self.alloc_scratch("v_mult_4097", VLEN)
        v_mult_33 = self.alloc_scratch("v_mult_33", VLEN)
        v_mult_9 = self.alloc_scratch("v_mult_9", VLEN)

        # ===== INIT PHASE =====
        tmp_addr = self.alloc_scratch("tmp_addr")
        for i, v in enumerate(init_vars):
            self.add_bundle({"load": [("const", tmp_addr, i)]})
            self.add_bundle({"load": [("load", self.scratch[v], tmp_addr)]})

        const_loads = [("const", addr_c, val) for val, addr_c in self.const_map.items()]
        for i in range(0, len(const_loads), 2):
            self.add_bundle({"load": const_loads[i:i+2]})

        self.add_bundle({"valu": [
            ("vbroadcast", v_one, one_const),
            ("vbroadcast", v_two, two_const),
            ("vbroadcast", v_zero, zero_const),
            ("vbroadcast", v_n_nodes, self.scratch["n_nodes"]),
        ]})

        # Broadcast multiplier constants
        self.add_bundle({"valu": [
            ("vbroadcast", v_mult_4097, mult_4097),
            ("vbroadcast", v_mult_33, mult_33),
            ("vbroadcast", v_mult_9, mult_9),
        ]})

        for i in range(0, len(hash_consts), 3):
            valu_ops = []
            for j in range(3):
                if i + j < len(hash_consts):
                    c1, c2 = hash_consts[i + j]
                    vc1, vc2 = v_hash_consts[i + j]
                    valu_ops.extend([("vbroadcast", vc1, c1), ("vbroadcast", vc2, c2)])
            self.add_bundle({"valu": valu_ops})

        # Load all indices and values - optimized with 2 pointers for parallel loads
        ptr2 = self.alloc_scratch("ptr2_init")
        vlen2_const = self.scratch_const(VLEN * 2)

        # Load indices: 2 vectors per cycle using 2 pointers
        self.add_bundle({"flow": [("add_imm", ptr, self.scratch["inp_indices_p"], 0)]})
        self.add_bundle({"flow": [("add_imm", ptr2, self.scratch["inp_indices_p"], VLEN)]})
        for i in range(0, num_batches, 2):
            self.add_bundle({"load": [("vload", v_idx[i], ptr), ("vload", v_idx[i + 1], ptr2)]})
            if i + 2 < num_batches:
                self.add_bundle({"alu": [("+", ptr, ptr, vlen2_const), ("+", ptr2, ptr2, vlen2_const)]})

        # Load values: same approach
        self.add_bundle({"flow": [("add_imm", ptr, self.scratch["inp_values_p"], 0)]})
        self.add_bundle({"flow": [("add_imm", ptr2, self.scratch["inp_values_p"], VLEN)]})
        for i in range(0, num_batches, 2):
            self.add_bundle({"load": [("vload", v_val[i], ptr), ("vload", v_val[i + 1], ptr2)]})
            if i + 2 < num_batches:
                self.add_bundle({"alu": [("+", ptr, ptr, vlen2_const), ("+", ptr2, ptr2, vlen2_const)]})

        self.add_bundle({"load": [("const", num_rounds_s, rounds)]})
        self.add("flow", ("pause",))

        # ===== ROUND 0 SPECIAL CASE =====
        # All indices start at 0, so we only need ONE forest load!
        node_scalar = self.alloc_scratch("node_scalar")
        v_node_shared = self.alloc_scratch("v_node_shared", VLEN)

        # Extra temps for 3-batch processing
        v_tmp1_E = self.alloc_scratch("v_tmp1_E", VLEN)
        v_tmp2_E = self.alloc_scratch("v_tmp2_E", VLEN)

        # Pre-allocate Round 1's forest value scratch (will load during Round 0 processing)
        node1_scalar = self.alloc_scratch("node1_scalar")
        node2_scalar = self.alloc_scratch("node2_scalar")
        v_node1 = self.alloc_scratch("v_node1", VLEN)
        v_node2 = self.alloc_scratch("v_node2", VLEN)
        addr1 = self.alloc_scratch("addr1")
        addr2 = self.alloc_scratch("addr2")

        # Load forest[0] once and broadcast + compute addr1 for Round 1
        self.add_bundle({
            "flow": [("add_imm", addr1, self.scratch["forest_values_p"], 1)],
            "load": [("load", node_scalar, self.scratch["forest_values_p"])],
        })
        # Broadcast + compute addr2 for Round 1
        self.add_bundle({
            "flow": [("add_imm", addr2, self.scratch["forest_values_p"], 2)],
            "valu": [("vbroadcast", v_node_shared, node_scalar)],
        })

        # Process 3 batches at a time (uses all 6 VALU slots for hash ops)
        # 32 batches = 10 groups of 3 + 1 group of 2
        # OPTIMIZED: Group 0 adds Round 1's forest loading
        for group in range(11):
            if group < 10:
                # 3 batches at a time
                b = group * 3
                val_A, val_B, val_C = v_val[b], v_val[b + 1], v_val[b + 2]
                idx_A, idx_B, idx_C = v_idx[b], v_idx[b + 1], v_idx[b + 2]

                if group == 0:
                    # Group 0: XOR + load Round 1's forest values
                    self.add_bundle({
                        "load": [("load", node1_scalar, addr1), ("load", node2_scalar, addr2)],
                        "valu": [
                            ("^", val_A, val_A, v_node_shared), ("^", val_B, val_B, v_node_shared),
                            ("^", val_C, val_C, v_node_shared),
                        ],
                    })
                else:
                    # Groups 1-9: XOR only
                    self.add_bundle({"valu": [
                        ("^", val_A, val_A, v_node_shared), ("^", val_B, val_B, v_node_shared),
                        ("^", val_C, val_C, v_node_shared),
                    ]})

                # Hash stages 0-5 for all 3 batches
                # Stages 0, 2, 4: use multiply_add (saves 1 cycle each)
                # Stages 1, 3, 5: use original 2-cycle approach
                mult_consts = [v_mult_4097, None, v_mult_33, None, v_mult_9, None]
                for hi in range(6):
                    vc1, vc2 = v_hash_consts[hi]
                    op1, _, op2, op3, _ = HASH_STAGES[hi]
                    if hi in [0, 2, 4]:
                        if group == 0 and hi == 0:
                            # Group 0, Stage 0: multiply_add + vbroadcast Round 1's nodes
                            self.add_bundle({"valu": [
                                ("multiply_add", val_A, val_A, mult_consts[hi], vc1),
                                ("multiply_add", val_B, val_B, mult_consts[hi], vc1),
                                ("multiply_add", val_C, val_C, mult_consts[hi], vc1),
                                ("vbroadcast", v_node1, node1_scalar),
                                ("vbroadcast", v_node2, node2_scalar),
                            ]})
                        else:
                            # multiply_add: val = val * mult + const
                            self.add_bundle({"valu": [
                                ("multiply_add", val_A, val_A, mult_consts[hi], vc1),
                                ("multiply_add", val_B, val_B, mult_consts[hi], vc1),
                                ("multiply_add", val_C, val_C, mult_consts[hi], vc1),
                            ]})
                    else:
                        # Original 2-cycle approach
                        self.add_bundle({"valu": [
                            (op1, v_tmp1_A, val_A, vc1), (op3, v_tmp2_A, val_A, vc2),
                            (op1, v_tmp1_B, val_B, vc1), (op3, v_tmp2_B, val_B, vc2),
                            (op1, v_tmp1_E, val_C, vc1), (op3, v_tmp2_E, val_C, vc2),
                        ]})
                        self.add_bundle({"valu": [
                            (op2, val_A, v_tmp1_A, v_tmp2_A),
                            (op2, val_B, v_tmp1_B, v_tmp2_B),
                            (op2, val_C, v_tmp1_E, v_tmp2_E),
                        ]})

                # idx = (val & 1) + 1
                self.add_bundle({"valu": [
                    ("&", idx_A, val_A, v_one), ("&", idx_B, val_B, v_one), ("&", idx_C, val_C, v_one),
                ]})
                self.add_bundle({"valu": [
                    ("+", idx_A, idx_A, v_one), ("+", idx_B, idx_B, v_one), ("+", idx_C, idx_C, v_one),
                ]})
            else:
                # Last 2 batches (batches 30-31)
                b = 30
                val_A, val_B = v_val[b], v_val[b + 1]
                idx_A, idx_B = v_idx[b], v_idx[b + 1]

                self.add_bundle({"valu": [("^", val_A, val_A, v_node_shared), ("^", val_B, val_B, v_node_shared)]})

                mult_consts = [v_mult_4097, None, v_mult_33, None, v_mult_9, None]
                for hi in range(6):
                    vc1, vc2 = v_hash_consts[hi]
                    op1, _, op2, op3, _ = HASH_STAGES[hi]
                    if hi in [0, 2, 4]:
                        self.add_bundle({"valu": [
                            ("multiply_add", val_A, val_A, mult_consts[hi], vc1),
                            ("multiply_add", val_B, val_B, mult_consts[hi], vc1),
                        ]})
                    else:
                        self.add_bundle({"valu": [
                            (op1, v_tmp1_A, val_A, vc1), (op3, v_tmp2_A, val_A, vc2),
                            (op1, v_tmp1_B, val_B, vc1), (op3, v_tmp2_B, val_B, vc2),
                        ]})
                        self.add_bundle({"valu": [(op2, val_A, v_tmp1_A, v_tmp2_A), (op2, val_B, v_tmp1_B, v_tmp2_B)]})

                self.add_bundle({"valu": [("&", idx_A, val_A, v_one), ("&", idx_B, val_B, v_one)]})
                self.add_bundle({"valu": [("+", idx_A, idx_A, v_one), ("+", idx_B, idx_B, v_one)]})

        # ===== ROUND 1 SPECIAL CASE =====
        # All indices are in {1, 2}, so we only need 2 forest loads!
        # PIPELINED: overlap vselect (flow) with idx computation (VALU) from previous pair
        # NOTE: node1_scalar, node2_scalar, v_node1, v_node2, addr1, addr2
        # were pre-allocated and loaded during Round 0's processing (optimization)

        # Extra temps for pipelining (C batch for overlap)
        v_tmp1_F = self.alloc_scratch("v_tmp1_F", VLEN)
        v_tmp2_F = self.alloc_scratch("v_tmp2_F", VLEN)

        # Pre-allocate Round 2's forest value scratch (will load during Round 1 processing)
        v_f3 = self.alloc_scratch("v_f3", VLEN)
        v_f4 = self.alloc_scratch("v_f4", VLEN)
        v_f5 = self.alloc_scratch("v_f5", VLEN)
        v_f6 = self.alloc_scratch("v_f6", VLEN)
        addr3 = self.alloc_scratch("addr3")
        addr4 = self.alloc_scratch("addr4")
        addr5 = self.alloc_scratch("addr5")
        addr6 = self.alloc_scratch("addr6")
        fs3 = self.alloc_scratch("fs3")
        fs4 = self.alloc_scratch("fs4")
        fs5 = self.alloc_scratch("fs5")
        fs6 = self.alloc_scratch("fs6")
        v_r_odd = self.alloc_scratch("v_r_odd", VLEN)
        v_r_even = self.alloc_scratch("v_r_even", VLEN)

        # Process pairs with pipelining - overlap vselect with previous idx computation
        # Pair 0: full processing without overlap
        # OPTIMIZED: Add Round 2 forest loading during hash stages
        val_A, val_B = v_val[0], v_val[1]
        idx_A, idx_B = v_idx[0], v_idx[1]

        # Compare + compute addr3
        self.add_bundle({
            "flow": [("add_imm", addr3, self.scratch["forest_values_p"], 3)],
            "valu": [("==", v_tmp1_A, idx_A, v_one), ("==", v_tmp1_B, idx_B, v_one)],
        })
        self.add_bundle({"flow": [("vselect", v_node_A, v_tmp1_A, v_node1, v_node2)]})
        self.add_bundle({"flow": [("vselect", v_node_B, v_tmp1_B, v_node1, v_node2)]})
        # XOR + compute addr4
        self.add_bundle({
            "flow": [("add_imm", addr4, self.scratch["forest_values_p"], 4)],
            "valu": [("^", val_A, val_A, v_node_A), ("^", val_B, val_B, v_node_B)],
        })

        # Hash stages with Round 2 forest loading interleaved
        mult_consts = [v_mult_4097, None, v_mult_33, None, v_mult_9, None]
        # Stage 0: multiply_add + load fs3, fs4
        vc1, vc2 = v_hash_consts[0]
        self.add_bundle({
            "load": [("load", fs3, addr3), ("load", fs4, addr4)],
            "valu": [
                ("multiply_add", val_A, val_A, mult_consts[0], vc1),
                ("multiply_add", val_B, val_B, mult_consts[0], vc1),
            ],
        })
        # Stage 1 part 1 + compute addr5
        vc1, vc2 = v_hash_consts[1]
        op1, _, op2, op3, _ = HASH_STAGES[1]
        self.add_bundle({
            "flow": [("add_imm", addr5, self.scratch["forest_values_p"], 5)],
            "valu": [
                (op1, v_tmp1_A, val_A, vc1), (op3, v_tmp2_A, val_A, vc2),
                (op1, v_tmp1_B, val_B, vc1), (op3, v_tmp2_B, val_B, vc2),
            ],
        })
        # Stage 1 part 2 + compute addr6
        self.add_bundle({
            "flow": [("add_imm", addr6, self.scratch["forest_values_p"], 6)],
            "valu": [(op2, val_A, v_tmp1_A, v_tmp2_A), (op2, val_B, v_tmp1_B, v_tmp2_B)],
        })
        # Stage 2: multiply_add + load fs5, fs6
        vc1, vc2 = v_hash_consts[2]
        self.add_bundle({
            "load": [("load", fs5, addr5), ("load", fs6, addr6)],
            "valu": [
                ("multiply_add", val_A, val_A, mult_consts[2], vc1),
                ("multiply_add", val_B, val_B, mult_consts[2], vc1),
            ],
        })
        # Stage 3 part 1 + broadcast v_f3, v_f4
        vc1, vc2 = v_hash_consts[3]
        op1, _, op2, op3, _ = HASH_STAGES[3]
        self.add_bundle({"valu": [
            (op1, v_tmp1_A, val_A, vc1), (op3, v_tmp2_A, val_A, vc2),
            (op1, v_tmp1_B, val_B, vc1), (op3, v_tmp2_B, val_B, vc2),
            ("vbroadcast", v_f3, fs3), ("vbroadcast", v_f4, fs4),
        ]})
        # Stage 3 part 2 + broadcast v_f5, v_f6
        self.add_bundle({"valu": [
            (op2, val_A, v_tmp1_A, v_tmp2_A), (op2, val_B, v_tmp1_B, v_tmp2_B),
            ("vbroadcast", v_f5, fs5), ("vbroadcast", v_f6, fs6),
        ]})
        # Stage 4: multiply_add (no more Round 2 loading)
        vc1, vc2 = v_hash_consts[4]
        self.add_bundle({"valu": [
            ("multiply_add", val_A, val_A, mult_consts[4], vc1),
            ("multiply_add", val_B, val_B, mult_consts[4], vc1),
        ]})
        # Stage 5 part 1
        vc1, vc2 = v_hash_consts[5]
        op1, _, op2, op3, _ = HASH_STAGES[5]
        self.add_bundle({"valu": [
            (op1, v_tmp1_A, val_A, vc1), (op3, v_tmp2_A, val_A, vc2),
            (op1, v_tmp1_B, val_B, vc1), (op3, v_tmp2_B, val_B, vc2),
        ]})
        # Stage 5 part 2
        self.add_bundle({"valu": [(op2, val_A, v_tmp1_A, v_tmp2_A), (op2, val_B, v_tmp1_B, v_tmp2_B)]})

        # idx = 2*idx + (val%2 + 1) - prepare for next iteration overlap
        self.add_bundle({"valu": [
            ("&", v_tmp1_A, val_A, v_one), ("<<", idx_A, idx_A, v_one),
            ("&", v_tmp1_B, val_B, v_one), ("<<", idx_B, idx_B, v_one),
        ]})
        self.add_bundle({"valu": [("+", v_tmp1_A, v_tmp1_A, v_one), ("+", v_tmp1_B, v_tmp1_B, v_one)]})
        # Save prev values for overlap
        prev_idx_A, prev_idx_B = idx_A, idx_B
        prev_tmp1_A, prev_tmp1_B = v_tmp1_A, v_tmp1_B

        # Pairs 1-15: overlapped processing
        for b in range(2, num_batches, 2):
            val_A, val_B = v_val[b], v_val[b + 1]
            idx_A, idx_B = v_idx[b], v_idx[b + 1]

            # Compare for current + idx add for previous
            self.add_bundle({"valu": [
                ("==", v_tmp1_F, idx_A, v_one), ("==", v_tmp2_F, idx_B, v_one),
                ("+", prev_idx_A, prev_idx_A, prev_tmp1_A), ("+", prev_idx_B, prev_idx_B, prev_tmp1_B),
            ]})
            # vselect A for current (no bounds check needed - idx always < n_nodes)
            self.add_bundle({"flow": [("vselect", v_node_A, v_tmp1_F, v_node1, v_node2)]})
            # vselect B for current
            self.add_bundle({"flow": [("vselect", v_node_B, v_tmp2_F, v_node1, v_node2)]})
            # XOR for current
            self.add_bundle({"valu": [("^", val_A, val_A, v_node_A), ("^", val_B, val_B, v_node_B)]})

            # Hash stages with multiply_add for stages 0, 2, 4
            mult_consts = [v_mult_4097, None, v_mult_33, None, v_mult_9, None]
            for hi in range(6):
                vc1, vc2 = v_hash_consts[hi]
                op1, _, op2, op3, _ = HASH_STAGES[hi]
                if hi in [0, 2, 4]:
                    self.add_bundle({"valu": [
                        ("multiply_add", val_A, val_A, mult_consts[hi], vc1),
                        ("multiply_add", val_B, val_B, mult_consts[hi], vc1),
                    ]})
                else:
                    self.add_bundle({"valu": [
                        (op1, v_tmp1_A, val_A, vc1), (op3, v_tmp2_A, val_A, vc2),
                        (op1, v_tmp1_B, val_B, vc1), (op3, v_tmp2_B, val_B, vc2),
                    ]})
                    self.add_bundle({"valu": [(op2, val_A, v_tmp1_A, v_tmp2_A), (op2, val_B, v_tmp1_B, v_tmp2_B)]})

            # idx computation - prepare for next overlap
            self.add_bundle({"valu": [
                ("&", v_tmp1_A, val_A, v_one), ("<<", idx_A, idx_A, v_one),
                ("&", v_tmp1_B, val_B, v_one), ("<<", idx_B, idx_B, v_one),
            ]})
            self.add_bundle({"valu": [("+", v_tmp1_A, v_tmp1_A, v_one), ("+", v_tmp1_B, v_tmp1_B, v_one)]})
            prev_idx_A, prev_idx_B = idx_A, idx_B
            prev_tmp1_A, prev_tmp1_B = v_tmp1_A, v_tmp1_B

        # Finish last pair's idx (no bounds check needed - idx always < n_nodes)
        self.add_bundle({"valu": [("+", prev_idx_A, prev_idx_A, prev_tmp1_A), ("+", prev_idx_B, prev_idx_B, prev_tmp1_B)]})

        # ===== ROUND 2 SPECIAL CASE =====
        # Indices are in {3,4,5,6}, so only 4 forest values needed
        # vselect tree: 3 vselects per batch (cheaper than 4 gather cycles)
        # NOTE: v_f3, v_f4, v_f5, v_f6, addr3-6, fs3-6, v_r_odd, v_r_even
        # were pre-allocated and loaded during Round 1's processing (optimization)

        # Process pairs with pipelining - overlap vselect (flow) with hash (VALU) from previous pair
        # Extra temps for pipelining
        v_bit0_C = self.alloc_scratch("v_bit0_C", VLEN)
        v_bit1_C = self.alloc_scratch("v_bit1_C", VLEN)
        v_bit0_D = self.alloc_scratch("v_bit0_D", VLEN)
        v_bit1_D = self.alloc_scratch("v_bit1_D", VLEN)

        # Pair 0: full processing without overlap
        val_A, val_B = v_val[0], v_val[1]
        idx_A, idx_B = v_idx[0], v_idx[1]

        self.add_bundle({"valu": [
            ("&", v_tmp1_A, idx_A, v_one), (">>", v_tmp2_A, idx_A, v_one),
            ("&", v_tmp1_B, idx_B, v_one), (">>", v_tmp2_B, idx_B, v_one),
        ]})
        self.add_bundle({"valu": [("&", v_tmp2_A, v_tmp2_A, v_one), ("&", v_tmp2_B, v_tmp2_B, v_one)]})
        self.add_bundle({"flow": [("vselect", v_r_odd, v_tmp2_A, v_f3, v_f5)]})
        self.add_bundle({"flow": [("vselect", v_r_even, v_tmp2_A, v_f6, v_f4)]})
        self.add_bundle({"flow": [("vselect", v_node_A, v_tmp1_A, v_r_odd, v_r_even)]})
        self.add_bundle({"flow": [("vselect", v_r_odd, v_tmp2_B, v_f3, v_f5)]})
        self.add_bundle({"flow": [("vselect", v_r_even, v_tmp2_B, v_f6, v_f4)]})
        self.add_bundle({"flow": [("vselect", v_node_B, v_tmp1_B, v_r_odd, v_r_even)]})
        self.add_bundle({"valu": [("^", val_A, val_A, v_node_A), ("^", val_B, val_B, v_node_B)]})

        # Hash stages 0-3 with multiply_add for stages 0, 2
        mult_consts = [v_mult_4097, None, v_mult_33, None]
        for hi in range(4):
            vc1, vc2 = v_hash_consts[hi]
            op1, _, op2, op3, _ = HASH_STAGES[hi]
            if hi in [0, 2]:
                self.add_bundle({"valu": [
                    ("multiply_add", val_A, val_A, mult_consts[hi], vc1),
                    ("multiply_add", val_B, val_B, mult_consts[hi], vc1),
                ]})
            else:
                self.add_bundle({"valu": [
                    (op1, v_tmp1_A, val_A, vc1), (op3, v_tmp2_A, val_A, vc2),
                    (op1, v_tmp1_B, val_B, vc1), (op3, v_tmp2_B, val_B, vc2),
                ]})
                self.add_bundle({"valu": [(op2, val_A, v_tmp1_A, v_tmp2_A), (op2, val_B, v_tmp1_B, v_tmp2_B)]})

        # Save prev values
        prev_val_A, prev_val_B = val_A, val_B
        prev_idx_A, prev_idx_B = idx_A, idx_B

        # Pairs 1-15: overlapped - vselect for current with hash finish for previous
        for b in range(2, num_batches, 2):
            val_A, val_B = v_val[b], v_val[b + 1]
            idx_A, idx_B = v_idx[b], v_idx[b + 1]

            # Bits for current + hash stage 4 for previous
            vc1_4, vc2_4 = v_hash_consts[4]
            op1_4, _, op2_4, op3_4, _ = HASH_STAGES[4]
            self.add_bundle({"valu": [
                ("&", v_bit0_C, idx_A, v_one), (">>", v_bit1_C, idx_A, v_one),
                (op1_4, v_tmp1_A, prev_val_A, vc1_4), (op3_4, v_tmp2_A, prev_val_A, vc2_4),
            ]})
            self.add_bundle({"valu": [
                ("&", v_bit0_D, idx_B, v_one), (">>", v_bit1_D, idx_B, v_one),
                (op2_4, prev_val_A, v_tmp1_A, v_tmp2_A),
                (op1_4, v_tmp1_B, prev_val_B, vc1_4), (op3_4, v_tmp2_B, prev_val_B, vc2_4),
            ]})
            # Mask + hash stage 5 for prev
            vc1_5, vc2_5 = v_hash_consts[5]
            op1_5, _, op2_5, op3_5, _ = HASH_STAGES[5]
            self.add_bundle({"valu": [
                ("&", v_bit1_C, v_bit1_C, v_one), ("&", v_bit1_D, v_bit1_D, v_one),
                (op2_4, prev_val_B, v_tmp1_B, v_tmp2_B),
            ]})
            # vselect A odd + hash stage 5 for prev
            self.add_bundle({
                "flow": [("vselect", v_r_odd, v_bit1_C, v_f3, v_f5)],
                "valu": [(op1_5, v_tmp1_A, prev_val_A, vc1_5), (op3_5, v_tmp2_A, prev_val_A, vc2_5)],
            })
            # vselect A even + hash finish for prev A
            self.add_bundle({
                "flow": [("vselect", v_r_even, v_bit1_C, v_f6, v_f4)],
                "valu": [(op2_5, prev_val_A, v_tmp1_A, v_tmp2_A)],
            })
            # vselect A final + hash 5 for prev B
            self.add_bundle({
                "flow": [("vselect", v_node_A, v_bit0_C, v_r_odd, v_r_even)],
                "valu": [(op1_5, v_tmp1_B, prev_val_B, vc1_5), (op3_5, v_tmp2_B, prev_val_B, vc2_5)],
            })
            # vselect B odd + hash finish for prev B
            self.add_bundle({
                "flow": [("vselect", v_r_odd, v_bit1_D, v_f3, v_f5)],
                "valu": [(op2_5, prev_val_B, v_tmp1_B, v_tmp2_B)],
            })
            # vselect B even + idx for prev A
            self.add_bundle({
                "flow": [("vselect", v_r_even, v_bit1_D, v_f6, v_f4)],
                "valu": [("&", v_tmp1_A, prev_val_A, v_one), ("<<", prev_idx_A, prev_idx_A, v_one)],
            })
            # vselect B final + idx for prev
            self.add_bundle({
                "flow": [("vselect", v_node_B, v_bit0_D, v_r_odd, v_r_even)],
                "valu": [
                    ("+", v_tmp1_A, v_tmp1_A, v_one),
                    ("&", v_tmp1_B, prev_val_B, v_one), ("<<", prev_idx_B, prev_idx_B, v_one),
                ],
            })
            # Finish prev idx + XOR current
            self.add_bundle({"valu": [
                ("+", prev_idx_A, prev_idx_A, v_tmp1_A),
                ("+", v_tmp1_B, v_tmp1_B, v_one),
            ]})
            self.add_bundle({"valu": [
                ("+", prev_idx_B, prev_idx_B, v_tmp1_B),
                ("^", val_A, val_A, v_node_A), ("^", val_B, val_B, v_node_B),
            ]})

            # Hash stages 0-3 with multiply_add for stages 0, 2
            mult_consts = [v_mult_4097, None, v_mult_33, None]
            for hi in range(4):
                vc1, vc2 = v_hash_consts[hi]
                op1, _, op2, op3, _ = HASH_STAGES[hi]
                if hi in [0, 2]:
                    self.add_bundle({"valu": [
                        ("multiply_add", val_A, val_A, mult_consts[hi], vc1),
                        ("multiply_add", val_B, val_B, mult_consts[hi], vc1),
                    ]})
                else:
                    self.add_bundle({"valu": [
                        (op1, v_tmp1_A, val_A, vc1), (op3, v_tmp2_A, val_A, vc2),
                        (op1, v_tmp1_B, val_B, vc1), (op3, v_tmp2_B, val_B, vc2),
                    ]})
                    self.add_bundle({"valu": [(op2, val_A, v_tmp1_A, v_tmp2_A), (op2, val_B, v_tmp1_B, v_tmp2_B)]})

            prev_val_A, prev_val_B = val_A, val_B
            prev_idx_A, prev_idx_B = idx_A, idx_B

        # Finish last pair (hash 4-5 + idx) - OVERLAPPED WITH PRE-LOADING FOR ROUND 3
        # Allocate node buffers early so we can pre-load during VALU computation
        v_tmp1_D = self.alloc_scratch("v_tmp1_D", VLEN)
        v_tmp2_D = self.alloc_scratch("v_tmp2_D", VLEN)
        tmp_list = [(v_tmp1_A, v_tmp2_A), (v_tmp1_B, v_tmp2_B), (v_tmp1_C, v_tmp2_C), (v_tmp1_D, v_tmp2_D)]

        v_node_C = self.alloc_scratch("v_node_C", VLEN)
        v_node_D = self.alloc_scratch("v_node_D", VLEN)
        node_set_A = [v_node_A, v_node_B, v_node_C, v_node_D]

        v_node_E = self.alloc_scratch("v_node_E", VLEN)
        v_node_F = self.alloc_scratch("v_node_F", VLEN)
        v_node_G = self.alloc_scratch("v_node_G", VLEN)
        v_node_H = self.alloc_scratch("v_node_H", VLEN)
        node_set_B = [v_node_E, v_node_F, v_node_G, v_node_H]

        v_node_I = self.alloc_scratch("v_node_I", VLEN)
        v_node_J = self.alloc_scratch("v_node_J", VLEN)
        v_node_K = self.alloc_scratch("v_node_K", VLEN)
        v_node_L = self.alloc_scratch("v_node_L", VLEN)
        node_set_C = [v_node_I, v_node_J, v_node_K, v_node_L]
        node_sets = [node_set_A, node_set_B, node_set_C]

        # Hash stage constants (extract once for readability)
        vc1_0, vc2_0 = v_hash_consts[0]
        op1_0, _, op2_0, op3_0, _ = HASH_STAGES[0]
        vc1_1, vc2_1 = v_hash_consts[1]
        op1_1, _, op2_1, op3_1, _ = HASH_STAGES[1]
        vc1_2, vc2_2 = v_hash_consts[2]
        op1_2, _, op2_2, op3_2, _ = HASH_STAGES[2]
        vc1_3, vc2_3 = v_hash_consts[3]
        op1_3, _, op2_3, op3_3, _ = HASH_STAGES[3]
        vc1_4, vc2_4 = v_hash_consts[4]
        op1_4, _, op2_4, op3_4, _ = HASH_STAGES[4]
        vc1_5, vc2_5 = v_hash_consts[5]
        op1_5, _, op2_5, op3_5, _ = HASH_STAGES[5]

        # Set up pre-loading: batch 0's nodes will be loaded during this VALU block
        preload_batch_0_idx = v_idx[0]  # Already computed during Round 2 pair loop
        preload_nodes = node_set_C

        # Compute addresses for batch 0 while doing hash stage 4
        self.add_bundle({
            "alu": [("+", addr_A[i], self.scratch["forest_values_p"], preload_batch_0_idx + i) for i in range(VLEN)],
            "valu": [
                (op1_4, v_tmp1_A, prev_val_A, vc1_4), (op3_4, v_tmp2_A, prev_val_A, vc2_4),
                (op1_4, v_tmp1_B, prev_val_B, vc1_4), (op3_4, v_tmp2_B, prev_val_B, vc2_4),
            ],
        })
        # Load batch 0 elements 0-1 while finishing hash 4
        self.add_bundle({
            "load": [("load", preload_nodes[0] + 0, addr_A[0]), ("load", preload_nodes[0] + 1, addr_A[1])],
            "valu": [(op2_4, prev_val_A, v_tmp1_A, v_tmp2_A), (op2_4, prev_val_B, v_tmp1_B, v_tmp2_B)],
        })
        # Load batch 0 elements 2-3 while doing hash stage 5
        self.add_bundle({
            "load": [("load", preload_nodes[0] + 2, addr_A[2]), ("load", preload_nodes[0] + 3, addr_A[3])],
            "valu": [
                (op1_5, v_tmp1_A, prev_val_A, vc1_5), (op3_5, v_tmp2_A, prev_val_A, vc2_5),
                (op1_5, v_tmp1_B, prev_val_B, vc1_5), (op3_5, v_tmp2_B, prev_val_B, vc2_5),
            ],
        })
        # Load batch 0 elements 4-5 while finishing hash 5
        self.add_bundle({
            "load": [("load", preload_nodes[0] + 4, addr_A[4]), ("load", preload_nodes[0] + 5, addr_A[5])],
            "valu": [(op2_5, prev_val_A, v_tmp1_A, v_tmp2_A), (op2_5, prev_val_B, v_tmp1_B, v_tmp2_B)],
        })
        # Load batch 0 elements 6-7 while doing idx computation
        self.add_bundle({
            "load": [("load", preload_nodes[0] + 6, addr_A[6]), ("load", preload_nodes[0] + 7, addr_A[7])],
            "valu": [
                ("&", v_tmp1_A, prev_val_A, v_one), ("<<", prev_idx_A, prev_idx_A, v_one),
                ("&", v_tmp1_B, prev_val_B, v_one), ("<<", prev_idx_B, prev_idx_B, v_one),
            ],
        })
        # Compute addresses for batch 1 while continuing idx
        preload_batch_1_idx = v_idx[1]
        self.add_bundle({
            "alu": [("+", addr_A[i], self.scratch["forest_values_p"], preload_batch_1_idx + i) for i in range(VLEN)],
            "valu": [("+", v_tmp1_A, v_tmp1_A, v_one), ("+", v_tmp1_B, v_tmp1_B, v_one)],
        })
        # Finish idx + load batch 1 elements 0-1
        self.add_bundle({
            "load": [("load", preload_nodes[1] + 0, addr_A[0]), ("load", preload_nodes[1] + 1, addr_A[1])],
            "valu": [("+", prev_idx_A, prev_idx_A, v_tmp1_A), ("+", prev_idx_B, prev_idx_B, v_tmp1_B)],
        })

        # ===== MAIN LOOP (rounds 3-9) - PIPELINED with overlapped loads =====
        # Key insight: Phase 5 has ~19 cycles with NO loads. During those cycles,
        # we can pre-load the next group's nodes (32 loads = 16 cycles at 2 loads/cycle).

        # ===== PRE-LOAD ROUND 3's GROUP 0 NODES INTO node_set_C =====
        # Batches 0-1 were already loaded during Round 2's final processing
        # Now just need to load batches 2-3 and the round counter
        base = 0
        batch_info = [(v_idx[base + i], v_val[base + i]) for i in range(4)]
        preload_nodes = node_set_C  # Already partially loaded (batches 0-1)

        # Load round counter + batch 1 elements 2-3
        self.add_bundle({"load": [("const", round_counter, 3), ("load", preload_nodes[1] + 2, addr_A[2])]})
        self.add_bundle({"load": [("load", preload_nodes[1] + 3, addr_A[3]), ("load", preload_nodes[1] + 4, addr_A[4])]})
        self.add_bundle({"load": [("load", preload_nodes[1] + 5, addr_A[5]), ("load", preload_nodes[1] + 6, addr_A[6])]})
        self.add_bundle({"load": [("load", preload_nodes[1] + 7, addr_A[7])]})

        # Load batches 2-3 (16 loads = 8 bundles)
        for b in range(2, 4):
            self.add_bundle({"alu": [("+", addr_A[i], self.scratch["forest_values_p"], batch_info[b][0] + i) for i in range(VLEN)]})
            for i in range(0, VLEN, 2):
                self.add_bundle({"load": [("load", preload_nodes[b] + i, addr_A[i]), ("load", preload_nodes[b] + i + 1, addr_A[i + 1])]})

        # Now start the main loop - Group 0 will use pre-loaded nodes from node_set_C
        round_loop_start = len(self.instrs)

        # ===== GROUP 0: Use pre-loaded nodes from node_set_C, pre-load Group 1 into node_set_B =====
        nodes = node_set_C  # Use pre-loaded nodes
        next_nodes = node_set_B  # Pre-load Group 1 into set B
        next_base = 4
        next_batch_info = [(v_idx[next_base + i], v_val[next_base + i]) for i in range(4)]

        # XOR all 4 batches + compute addresses for next batch 0
        self.add_bundle({
            "alu": [("+", addr_A[i], self.scratch["forest_values_p"], next_batch_info[0][0] + i) for i in range(VLEN)],
            "valu": [
                ("^", batch_info[0][1], batch_info[0][1], nodes[0]),
                ("^", batch_info[1][1], batch_info[1][1], nodes[1]),
                ("^", batch_info[2][1], batch_info[2][1], nodes[2]),
                ("^", batch_info[3][1], batch_info[3][1], nodes[3]),
            ],
        })

        # Hash stage 0 with multiply_add + load next batch 0 elements 0-1
        self.add_bundle({
            "load": [("load", next_nodes[0] + 0, addr_A[0]), ("load", next_nodes[0] + 1, addr_A[1])],
            "valu": [
                ("multiply_add", batch_info[0][1], batch_info[0][1], v_mult_4097, vc1_0),
                ("multiply_add", batch_info[1][1], batch_info[1][1], v_mult_4097, vc1_0),
                ("multiply_add", batch_info[2][1], batch_info[2][1], v_mult_4097, vc1_0),
                ("multiply_add", batch_info[3][1], batch_info[3][1], v_mult_4097, vc1_0),
            ],
        })

        # Hash stage 1 + load next batch 0 elements 2-7
        self.add_bundle({
            "load": [("load", next_nodes[0] + 2, addr_A[2]), ("load", next_nodes[0] + 3, addr_A[3])],
            "valu": [
                (op1_1, tmp_list[0][0], batch_info[0][1], vc1_1), (op3_1, tmp_list[0][1], batch_info[0][1], vc2_1),
                (op1_1, tmp_list[1][0], batch_info[1][1], vc1_1), (op3_1, tmp_list[1][1], batch_info[1][1], vc2_1),
            ],
        })
        self.add_bundle({
            "load": [("load", next_nodes[0] + 4, addr_A[4]), ("load", next_nodes[0] + 5, addr_A[5])],
            "valu": [
                (op2_1, batch_info[0][1], tmp_list[0][0], tmp_list[0][1]),
                (op2_1, batch_info[1][1], tmp_list[1][0], tmp_list[1][1]),
                (op1_1, tmp_list[2][0], batch_info[2][1], vc1_1), (op3_1, tmp_list[2][1], batch_info[2][1], vc2_1),
            ],
        })
        self.add_bundle({
            "load": [("load", next_nodes[0] + 6, addr_A[6]), ("load", next_nodes[0] + 7, addr_A[7])],
            "valu": [
                (op2_1, batch_info[2][1], tmp_list[2][0], tmp_list[2][1]),
                (op1_1, tmp_list[3][0], batch_info[3][1], vc1_1), (op3_1, tmp_list[3][1], batch_info[3][1], vc2_1),
            ],
        })
        # Finish stage 1 batch 3 + compute addresses for next batch 1
        self.add_bundle({
            "alu": [("+", addr_A[i], self.scratch["forest_values_p"], next_batch_info[1][0] + i) for i in range(VLEN)],
            "valu": [(op2_1, batch_info[3][1], tmp_list[3][0], tmp_list[3][1])],
        })

        # Hash stage 2 with multiply_add + load next batch 1 elements 0-1
        self.add_bundle({
            "load": [("load", next_nodes[1] + 0, addr_A[0]), ("load", next_nodes[1] + 1, addr_A[1])],
            "valu": [
                ("multiply_add", batch_info[0][1], batch_info[0][1], v_mult_33, vc1_2),
                ("multiply_add", batch_info[1][1], batch_info[1][1], v_mult_33, vc1_2),
                ("multiply_add", batch_info[2][1], batch_info[2][1], v_mult_33, vc1_2),
                ("multiply_add", batch_info[3][1], batch_info[3][1], v_mult_33, vc1_2),
            ],
        })

        # Hash stage 3 + load next batch 1 elements 2-7
        self.add_bundle({
            "load": [("load", next_nodes[1] + 2, addr_A[2]), ("load", next_nodes[1] + 3, addr_A[3])],
            "valu": [
                (op1_3, tmp_list[0][0], batch_info[0][1], vc1_3), (op3_3, tmp_list[0][1], batch_info[0][1], vc2_3),
                (op1_3, tmp_list[1][0], batch_info[1][1], vc1_3), (op3_3, tmp_list[1][1], batch_info[1][1], vc2_3),
            ],
        })
        self.add_bundle({
            "load": [("load", next_nodes[1] + 4, addr_A[4]), ("load", next_nodes[1] + 5, addr_A[5])],
            "valu": [
                (op2_3, batch_info[0][1], tmp_list[0][0], tmp_list[0][1]),
                (op2_3, batch_info[1][1], tmp_list[1][0], tmp_list[1][1]),
                (op1_3, tmp_list[2][0], batch_info[2][1], vc1_3), (op3_3, tmp_list[2][1], batch_info[2][1], vc2_3),
            ],
        })
        self.add_bundle({
            "load": [("load", next_nodes[1] + 6, addr_A[6]), ("load", next_nodes[1] + 7, addr_A[7])],
            "valu": [
                (op2_3, batch_info[2][1], tmp_list[2][0], tmp_list[2][1]),
                (op1_3, tmp_list[3][0], batch_info[3][1], vc1_3), (op3_3, tmp_list[3][1], batch_info[3][1], vc2_3),
            ],
        })
        # Finish stage 3 batch 3 + compute addresses for next batch 2
        self.add_bundle({
            "alu": [("+", addr_A[i], self.scratch["forest_values_p"], next_batch_info[2][0] + i) for i in range(VLEN)],
            "valu": [(op2_3, batch_info[3][1], tmp_list[3][0], tmp_list[3][1])],
        })

        # Hash stage 4 with multiply_add + load next batch 2 elements 0-1
        self.add_bundle({
            "load": [("load", next_nodes[2] + 0, addr_A[0]), ("load", next_nodes[2] + 1, addr_A[1])],
            "valu": [
                ("multiply_add", batch_info[0][1], batch_info[0][1], v_mult_9, vc1_4),
                ("multiply_add", batch_info[1][1], batch_info[1][1], v_mult_9, vc1_4),
                ("multiply_add", batch_info[2][1], batch_info[2][1], v_mult_9, vc1_4),
                ("multiply_add", batch_info[3][1], batch_info[3][1], v_mult_9, vc1_4),
            ],
        })

        # Hash stage 5 + load next batch 2 elements 2-7
        self.add_bundle({
            "load": [("load", next_nodes[2] + 2, addr_A[2]), ("load", next_nodes[2] + 3, addr_A[3])],
            "valu": [
                (op1_5, tmp_list[0][0], batch_info[0][1], vc1_5), (op3_5, tmp_list[0][1], batch_info[0][1], vc2_5),
                (op1_5, tmp_list[1][0], batch_info[1][1], vc1_5), (op3_5, tmp_list[1][1], batch_info[1][1], vc2_5),
            ],
        })
        self.add_bundle({
            "load": [("load", next_nodes[2] + 4, addr_A[4]), ("load", next_nodes[2] + 5, addr_A[5])],
            "valu": [
                (op2_5, batch_info[0][1], tmp_list[0][0], tmp_list[0][1]),
                (op2_5, batch_info[1][1], tmp_list[1][0], tmp_list[1][1]),
                (op1_5, tmp_list[2][0], batch_info[2][1], vc1_5), (op3_5, tmp_list[2][1], batch_info[2][1], vc2_5),
            ],
        })
        self.add_bundle({
            "load": [("load", next_nodes[2] + 6, addr_A[6]), ("load", next_nodes[2] + 7, addr_A[7])],
            "valu": [
                (op2_5, batch_info[2][1], tmp_list[2][0], tmp_list[2][1]),
                (op1_5, tmp_list[3][0], batch_info[3][1], vc1_5), (op3_5, tmp_list[3][1], batch_info[3][1], vc2_5),
            ],
        })
        # Finish stage 5 batch 3 + compute addresses for next batch 3
        self.add_bundle({
            "alu": [("+", addr_A[i], self.scratch["forest_values_p"], next_batch_info[3][0] + i) for i in range(VLEN)],
            "valu": [(op2_5, batch_info[3][1], tmp_list[3][0], tmp_list[3][1])],
        })

        # Compute idx for all 4 batches + load next batch 3 elements 0-7
        self.add_bundle({
            "load": [("load", next_nodes[3] + 0, addr_A[0]), ("load", next_nodes[3] + 1, addr_A[1])],
            "valu": [
                ("multiply_add", batch_info[0][0], batch_info[0][0], v_two, v_one),
                ("multiply_add", batch_info[1][0], batch_info[1][0], v_two, v_one),
                ("multiply_add", batch_info[2][0], batch_info[2][0], v_two, v_one),
                ("multiply_add", batch_info[3][0], batch_info[3][0], v_two, v_one),
            ],
        })
        self.add_bundle({
            "load": [("load", next_nodes[3] + 2, addr_A[2]), ("load", next_nodes[3] + 3, addr_A[3])],
            "valu": [
                ("&", tmp_list[0][0], batch_info[0][1], v_one),
                ("&", tmp_list[1][0], batch_info[1][1], v_one),
                ("&", tmp_list[2][0], batch_info[2][1], v_one),
                ("&", tmp_list[3][0], batch_info[3][1], v_one),
            ],
        })
        self.add_bundle({
            "load": [("load", next_nodes[3] + 4, addr_A[4]), ("load", next_nodes[3] + 5, addr_A[5])],
            "valu": [
                ("+", batch_info[0][0], batch_info[0][0], tmp_list[0][0]),
                ("+", batch_info[1][0], batch_info[1][0], tmp_list[1][0]),
                ("+", batch_info[2][0], batch_info[2][0], tmp_list[2][0]),
                ("+", batch_info[3][0], batch_info[3][0], tmp_list[3][0]),
            ],
        })
        self.add_bundle({"load": [("load", next_nodes[3] + 6, addr_A[6]), ("load", next_nodes[3] + 7, addr_A[7])]})

        # ===== GROUPS 1-6: Use pre-loaded nodes + pre-load next group =====
        # Optimized: compute addresses for 2 batches at once using addr_A and addr_B
        for group in range(1, 7):
            base = group * 4
            batch_info = [(v_idx[base + i], v_val[base + i]) for i in range(4)]
            nodes = node_set_B if group % 2 == 1 else node_set_A
            next_nodes = node_set_A if group % 2 == 1 else node_set_B

            next_base = (group + 1) * 4
            next_batch_info = [(v_idx[next_base + i], v_val[next_base + i]) for i in range(4)]

            # Compute addresses for next batches 0,1 + XOR current batches
            self.add_bundle({
                "alu": [("+", addr_A[i], self.scratch["forest_values_p"], next_batch_info[0][0] + i) for i in range(VLEN)],
                "valu": [
                    ("^", batch_info[0][1], batch_info[0][1], nodes[0]),
                    ("^", batch_info[1][1], batch_info[1][1], nodes[1]),
                    ("^", batch_info[2][1], batch_info[2][1], nodes[2]),
                    ("^", batch_info[3][1], batch_info[3][1], nodes[3]),
                ],
            })
            self.add_bundle({
                "alu": [("+", addr_B[i], self.scratch["forest_values_p"], next_batch_info[1][0] + i) for i in range(VLEN)],
                "valu": [
                    (op1_0, tmp_list[0][0], batch_info[0][1], vc1_0), (op3_0, tmp_list[0][1], batch_info[0][1], vc2_0),
                    (op1_0, tmp_list[1][0], batch_info[1][1], vc1_0), (op3_0, tmp_list[1][1], batch_info[1][1], vc2_0),
                ],
            })
            # Load batches 0,1 interleaved (8 cycles for 16 loads)
            self.add_bundle({
                "load": [("load", next_nodes[0] + 0, addr_A[0]), ("load", next_nodes[1] + 0, addr_B[0])],
                "valu": [
                    (op2_0, batch_info[0][1], tmp_list[0][0], tmp_list[0][1]),
                    (op2_0, batch_info[1][1], tmp_list[1][0], tmp_list[1][1]),
                    (op1_0, tmp_list[2][0], batch_info[2][1], vc1_0), (op3_0, tmp_list[2][1], batch_info[2][1], vc2_0),
                ],
            })
            self.add_bundle({
                "load": [("load", next_nodes[0] + 1, addr_A[1]), ("load", next_nodes[1] + 1, addr_B[1])],
                "valu": [
                    (op1_1, tmp_list[0][0], batch_info[0][1], vc1_1), (op3_1, tmp_list[0][1], batch_info[0][1], vc2_1),
                    (op1_1, tmp_list[1][0], batch_info[1][1], vc1_1), (op3_1, tmp_list[1][1], batch_info[1][1], vc2_1),
                ],
            })
            self.add_bundle({
                "load": [("load", next_nodes[0] + 2, addr_A[2]), ("load", next_nodes[1] + 2, addr_B[2])],
                "valu": [
                    (op2_1, batch_info[0][1], tmp_list[0][0], tmp_list[0][1]),
                    (op2_1, batch_info[1][1], tmp_list[1][0], tmp_list[1][1]),
                    (op2_0, batch_info[2][1], tmp_list[2][0], tmp_list[2][1]),
                    (op1_0, tmp_list[3][0], batch_info[3][1], vc1_0), (op3_0, tmp_list[3][1], batch_info[3][1], vc2_0),
                ],
            })
            self.add_bundle({
                "load": [("load", next_nodes[0] + 3, addr_A[3]), ("load", next_nodes[1] + 3, addr_B[3])],
                "valu": [
                    (op1_2, tmp_list[0][0], batch_info[0][1], vc1_2), (op3_2, tmp_list[0][1], batch_info[0][1], vc2_2),
                    (op1_2, tmp_list[1][0], batch_info[1][1], vc1_2), (op3_2, tmp_list[1][1], batch_info[1][1], vc2_2),
                ],
            })
            self.add_bundle({
                "load": [("load", next_nodes[0] + 4, addr_A[4]), ("load", next_nodes[1] + 4, addr_B[4])],
                "valu": [
                    (op2_2, batch_info[0][1], tmp_list[0][0], tmp_list[0][1]),
                    (op2_2, batch_info[1][1], tmp_list[1][0], tmp_list[1][1]),
                    (op1_1, tmp_list[2][0], batch_info[2][1], vc1_1), (op3_1, tmp_list[2][1], batch_info[2][1], vc2_1),
                ],
            })
            self.add_bundle({
                "load": [("load", next_nodes[0] + 5, addr_A[5]), ("load", next_nodes[1] + 5, addr_B[5])],
                "valu": [
                    (op1_3, tmp_list[0][0], batch_info[0][1], vc1_3), (op3_3, tmp_list[0][1], batch_info[0][1], vc2_3),
                    (op1_3, tmp_list[1][0], batch_info[1][1], vc1_3), (op3_3, tmp_list[1][1], batch_info[1][1], vc2_3),
                ],
            })
            self.add_bundle({
                "load": [("load", next_nodes[0] + 6, addr_A[6]), ("load", next_nodes[1] + 6, addr_B[6])],
                "valu": [
                    (op2_3, batch_info[0][1], tmp_list[0][0], tmp_list[0][1]),
                    (op2_3, batch_info[1][1], tmp_list[1][0], tmp_list[1][1]),
                    (op2_1, batch_info[2][1], tmp_list[2][0], tmp_list[2][1]),
                    (op2_0, batch_info[3][1], tmp_list[3][0], tmp_list[3][1]),
                ],
            })
            self.add_bundle({
                "load": [("load", next_nodes[0] + 7, addr_A[7]), ("load", next_nodes[1] + 7, addr_B[7])],
                "valu": [
                    (op1_4, tmp_list[0][0], batch_info[0][1], vc1_4), (op3_4, tmp_list[0][1], batch_info[0][1], vc2_4),
                    (op1_4, tmp_list[1][0], batch_info[1][1], vc1_4), (op3_4, tmp_list[1][1], batch_info[1][1], vc2_4),
                ],
            })
            # Compute addresses for next batches 2,3
            self.add_bundle({
                "alu": [("+", addr_A[i], self.scratch["forest_values_p"], next_batch_info[2][0] + i) for i in range(VLEN)],
                "valu": [
                    (op2_4, batch_info[0][1], tmp_list[0][0], tmp_list[0][1]),
                    (op2_4, batch_info[1][1], tmp_list[1][0], tmp_list[1][1]),
                    (op1_2, tmp_list[2][0], batch_info[2][1], vc1_2), (op3_2, tmp_list[2][1], batch_info[2][1], vc2_2),
                ],
            })
            self.add_bundle({
                "alu": [("+", addr_B[i], self.scratch["forest_values_p"], next_batch_info[3][0] + i) for i in range(VLEN)],
                "valu": [
                    (op1_5, tmp_list[0][0], batch_info[0][1], vc1_5), (op3_5, tmp_list[0][1], batch_info[0][1], vc2_5),
                    (op1_5, tmp_list[1][0], batch_info[1][1], vc1_5), (op3_5, tmp_list[1][1], batch_info[1][1], vc2_5),
                ],
            })
            # Load batches 2,3 interleaved
            self.add_bundle({
                "load": [("load", next_nodes[2] + 0, addr_A[0]), ("load", next_nodes[3] + 0, addr_B[0])],
                "valu": [
                    (op2_5, batch_info[0][1], tmp_list[0][0], tmp_list[0][1]),
                    (op2_5, batch_info[1][1], tmp_list[1][0], tmp_list[1][1]),
                    (op2_2, batch_info[2][1], tmp_list[2][0], tmp_list[2][1]),
                    (op1_1, tmp_list[3][0], batch_info[3][1], vc1_1), (op3_1, tmp_list[3][1], batch_info[3][1], vc2_1),
                ],
            })
            self.add_bundle({
                "load": [("load", next_nodes[2] + 1, addr_A[1]), ("load", next_nodes[3] + 1, addr_B[1])],
                "valu": [
                    ("&", tmp_list[0][0], batch_info[0][1], v_one), ("<<", batch_info[0][0], batch_info[0][0], v_one),
                    ("&", tmp_list[1][0], batch_info[1][1], v_one), ("<<", batch_info[1][0], batch_info[1][0], v_one),
                ],
            })
            self.add_bundle({
                "load": [("load", next_nodes[2] + 2, addr_A[2]), ("load", next_nodes[3] + 2, addr_B[2])],
                "valu": [
                    ("+", tmp_list[0][0], tmp_list[0][0], v_one),
                    ("+", tmp_list[1][0], tmp_list[1][0], v_one),
                    (op1_3, tmp_list[2][0], batch_info[2][1], vc1_3), (op3_3, tmp_list[2][1], batch_info[2][1], vc2_3),
                ],
            })
            self.add_bundle({
                "load": [("load", next_nodes[2] + 3, addr_A[3]), ("load", next_nodes[3] + 3, addr_B[3])],
                "valu": [
                    ("+", batch_info[0][0], batch_info[0][0], tmp_list[0][0]),
                    ("+", batch_info[1][0], batch_info[1][0], tmp_list[1][0]),
                    (op2_3, batch_info[2][1], tmp_list[2][0], tmp_list[2][1]),
                    (op2_1, batch_info[3][1], tmp_list[3][0], tmp_list[3][1]),
                ],
            })
            self.add_bundle({
                "load": [("load", next_nodes[2] + 4, addr_A[4]), ("load", next_nodes[3] + 4, addr_B[4])],
                "valu": [
                    (op1_4, tmp_list[2][0], batch_info[2][1], vc1_4), (op3_4, tmp_list[2][1], batch_info[2][1], vc2_4),
                    (op1_2, tmp_list[3][0], batch_info[3][1], vc1_2), (op3_2, tmp_list[3][1], batch_info[3][1], vc2_2),
                ],
            })
            self.add_bundle({
                "load": [("load", next_nodes[2] + 5, addr_A[5]), ("load", next_nodes[3] + 5, addr_B[5])],
                "valu": [
                    (op2_4, batch_info[2][1], tmp_list[2][0], tmp_list[2][1]),
                    (op2_2, batch_info[3][1], tmp_list[3][0], tmp_list[3][1]),
                ],
            })
            self.add_bundle({
                "load": [("load", next_nodes[2] + 6, addr_A[6]), ("load", next_nodes[3] + 6, addr_B[6])],
                "valu": [
                    (op1_5, tmp_list[2][0], batch_info[2][1], vc1_5), (op3_5, tmp_list[2][1], batch_info[2][1], vc2_5),
                    (op1_3, tmp_list[3][0], batch_info[3][1], vc1_3), (op3_3, tmp_list[3][1], batch_info[3][1], vc2_3),
                ],
            })
            self.add_bundle({
                "load": [("load", next_nodes[2] + 7, addr_A[7]), ("load", next_nodes[3] + 7, addr_B[7])],
                "valu": [
                    (op2_5, batch_info[2][1], tmp_list[2][0], tmp_list[2][1]),
                    (op2_3, batch_info[3][1], tmp_list[3][0], tmp_list[3][1]),
                ],
            })
            # Finish idx for batches 2,3 (OPTIMIZED: multiply_add combines <<1 and +1)
            # Batch 2: idx = idx*2+1, tmp = val&1, idx += tmp  (saves 1 cycle)
            # Batch 3: same pattern after hash 5 finishes
            self.add_bundle({"valu": [
                ("multiply_add", batch_info[2][0], batch_info[2][0], v_two, v_one),  # idx*2+1
                ("&", tmp_list[2][0], batch_info[2][1], v_one),  # tmp = val&1
                (op1_4, tmp_list[3][0], batch_info[3][1], vc1_4), (op3_4, tmp_list[3][1], batch_info[3][1], vc2_4),
            ]})
            self.add_bundle({"valu": [
                ("+", batch_info[2][0], batch_info[2][0], tmp_list[2][0]),  # idx += tmp (batch 2 done)
                (op2_4, batch_info[3][1], tmp_list[3][0], tmp_list[3][1]),
            ]})
            self.add_bundle({"valu": [
                (op1_5, tmp_list[3][0], batch_info[3][1], vc1_5), (op3_5, tmp_list[3][1], batch_info[3][1], vc2_5),
            ]})
            self.add_bundle({"valu": [(op2_5, batch_info[3][1], tmp_list[3][0], tmp_list[3][1])]})
            self.add_bundle({"valu": [
                ("multiply_add", batch_info[3][0], batch_info[3][0], v_two, v_one),  # idx*2+1
                ("&", tmp_list[3][0], batch_info[3][1], v_one),  # tmp = val&1
            ]})
            self.add_bundle({"valu": [("+", batch_info[3][0], batch_info[3][0], tmp_list[3][0])]})

        # ===== GROUP 7: Use pre-loaded nodes + PRE-LOAD FOR NEXT ROUND'S GROUP 0 =====
        # OPTIMIZED: Use multiply_add for hash stages 0, 2, 4
        # CROSS-ROUND PIPELINING: Pre-load next round's Group 0 nodes into node_set_C
        # during the VALU-only hash computation
        base = 7 * 4
        batch_info = [(v_idx[base + i], v_val[base + i]) for i in range(4)]
        nodes = node_set_B  # Group 7 (odd) uses set B

        # Next round's Group 0 uses batches 0-3 (v_idx[0..3] already has this round's results)
        next_batch_info = [(v_idx[i], v_val[i]) for i in range(4)]
        preload_nodes = node_set_C  # Store pre-loaded nodes in set C

        # XOR all 4 batches + compute addresses for next batch 0
        self.add_bundle({
            "alu": [("+", addr_A[i], self.scratch["forest_values_p"], next_batch_info[0][0] + i) for i in range(VLEN)],
            "valu": [
                ("^", batch_info[0][1], batch_info[0][1], nodes[0]),
                ("^", batch_info[1][1], batch_info[1][1], nodes[1]),
                ("^", batch_info[2][1], batch_info[2][1], nodes[2]),
                ("^", batch_info[3][1], batch_info[3][1], nodes[3]),
            ],
        })

        # Hash stage 0 with multiply_add + load next batch 0 elements 0-1
        self.add_bundle({
            "load": [("load", preload_nodes[0] + 0, addr_A[0]), ("load", preload_nodes[0] + 1, addr_A[1])],
            "valu": [
                ("multiply_add", batch_info[0][1], batch_info[0][1], v_mult_4097, vc1_0),
                ("multiply_add", batch_info[1][1], batch_info[1][1], v_mult_4097, vc1_0),
                ("multiply_add", batch_info[2][1], batch_info[2][1], v_mult_4097, vc1_0),
                ("multiply_add", batch_info[3][1], batch_info[3][1], v_mult_4097, vc1_0),
            ],
        })

        # Hash stage 1 + load next batch 0 elements 2-7
        self.add_bundle({
            "load": [("load", preload_nodes[0] + 2, addr_A[2]), ("load", preload_nodes[0] + 3, addr_A[3])],
            "valu": [
                (op1_1, tmp_list[0][0], batch_info[0][1], vc1_1), (op3_1, tmp_list[0][1], batch_info[0][1], vc2_1),
                (op1_1, tmp_list[1][0], batch_info[1][1], vc1_1), (op3_1, tmp_list[1][1], batch_info[1][1], vc2_1),
            ],
        })
        self.add_bundle({
            "load": [("load", preload_nodes[0] + 4, addr_A[4]), ("load", preload_nodes[0] + 5, addr_A[5])],
            "valu": [
                (op2_1, batch_info[0][1], tmp_list[0][0], tmp_list[0][1]),
                (op2_1, batch_info[1][1], tmp_list[1][0], tmp_list[1][1]),
                (op1_1, tmp_list[2][0], batch_info[2][1], vc1_1), (op3_1, tmp_list[2][1], batch_info[2][1], vc2_1),
            ],
        })
        self.add_bundle({
            "load": [("load", preload_nodes[0] + 6, addr_A[6]), ("load", preload_nodes[0] + 7, addr_A[7])],
            "valu": [
                (op2_1, batch_info[2][1], tmp_list[2][0], tmp_list[2][1]),
                (op1_1, tmp_list[3][0], batch_info[3][1], vc1_1), (op3_1, tmp_list[3][1], batch_info[3][1], vc2_1),
            ],
        })
        # Finish stage 1 batch 3 + compute addresses for next batch 1
        self.add_bundle({
            "alu": [("+", addr_A[i], self.scratch["forest_values_p"], next_batch_info[1][0] + i) for i in range(VLEN)],
            "valu": [(op2_1, batch_info[3][1], tmp_list[3][0], tmp_list[3][1])],
        })

        # Hash stage 2 with multiply_add + load next batch 1 elements 0-1
        self.add_bundle({
            "load": [("load", preload_nodes[1] + 0, addr_A[0]), ("load", preload_nodes[1] + 1, addr_A[1])],
            "valu": [
                ("multiply_add", batch_info[0][1], batch_info[0][1], v_mult_33, vc1_2),
                ("multiply_add", batch_info[1][1], batch_info[1][1], v_mult_33, vc1_2),
                ("multiply_add", batch_info[2][1], batch_info[2][1], v_mult_33, vc1_2),
                ("multiply_add", batch_info[3][1], batch_info[3][1], v_mult_33, vc1_2),
            ],
        })

        # Hash stage 3 + load next batch 1 elements 2-7
        self.add_bundle({
            "load": [("load", preload_nodes[1] + 2, addr_A[2]), ("load", preload_nodes[1] + 3, addr_A[3])],
            "valu": [
                (op1_3, tmp_list[0][0], batch_info[0][1], vc1_3), (op3_3, tmp_list[0][1], batch_info[0][1], vc2_3),
                (op1_3, tmp_list[1][0], batch_info[1][1], vc1_3), (op3_3, tmp_list[1][1], batch_info[1][1], vc2_3),
            ],
        })
        self.add_bundle({
            "load": [("load", preload_nodes[1] + 4, addr_A[4]), ("load", preload_nodes[1] + 5, addr_A[5])],
            "valu": [
                (op2_3, batch_info[0][1], tmp_list[0][0], tmp_list[0][1]),
                (op2_3, batch_info[1][1], tmp_list[1][0], tmp_list[1][1]),
                (op1_3, tmp_list[2][0], batch_info[2][1], vc1_3), (op3_3, tmp_list[2][1], batch_info[2][1], vc2_3),
            ],
        })
        self.add_bundle({
            "load": [("load", preload_nodes[1] + 6, addr_A[6]), ("load", preload_nodes[1] + 7, addr_A[7])],
            "valu": [
                (op2_3, batch_info[2][1], tmp_list[2][0], tmp_list[2][1]),
                (op1_3, tmp_list[3][0], batch_info[3][1], vc1_3), (op3_3, tmp_list[3][1], batch_info[3][1], vc2_3),
            ],
        })
        # Finish stage 3 batch 3 + compute addresses for next batch 2
        self.add_bundle({
            "alu": [("+", addr_A[i], self.scratch["forest_values_p"], next_batch_info[2][0] + i) for i in range(VLEN)],
            "valu": [(op2_3, batch_info[3][1], tmp_list[3][0], tmp_list[3][1])],
        })

        # Hash stage 4 with multiply_add + load next batch 2 elements 0-1
        self.add_bundle({
            "load": [("load", preload_nodes[2] + 0, addr_A[0]), ("load", preload_nodes[2] + 1, addr_A[1])],
            "valu": [
                ("multiply_add", batch_info[0][1], batch_info[0][1], v_mult_9, vc1_4),
                ("multiply_add", batch_info[1][1], batch_info[1][1], v_mult_9, vc1_4),
                ("multiply_add", batch_info[2][1], batch_info[2][1], v_mult_9, vc1_4),
                ("multiply_add", batch_info[3][1], batch_info[3][1], v_mult_9, vc1_4),
            ],
        })

        # Hash stage 5 + load next batch 2 elements 2-7
        self.add_bundle({
            "load": [("load", preload_nodes[2] + 2, addr_A[2]), ("load", preload_nodes[2] + 3, addr_A[3])],
            "valu": [
                (op1_5, tmp_list[0][0], batch_info[0][1], vc1_5), (op3_5, tmp_list[0][1], batch_info[0][1], vc2_5),
                (op1_5, tmp_list[1][0], batch_info[1][1], vc1_5), (op3_5, tmp_list[1][1], batch_info[1][1], vc2_5),
            ],
        })
        self.add_bundle({
            "load": [("load", preload_nodes[2] + 4, addr_A[4]), ("load", preload_nodes[2] + 5, addr_A[5])],
            "valu": [
                (op2_5, batch_info[0][1], tmp_list[0][0], tmp_list[0][1]),
                (op2_5, batch_info[1][1], tmp_list[1][0], tmp_list[1][1]),
                (op1_5, tmp_list[2][0], batch_info[2][1], vc1_5), (op3_5, tmp_list[2][1], batch_info[2][1], vc2_5),
            ],
        })
        self.add_bundle({
            "load": [("load", preload_nodes[2] + 6, addr_A[6]), ("load", preload_nodes[2] + 7, addr_A[7])],
            "valu": [
                (op2_5, batch_info[2][1], tmp_list[2][0], tmp_list[2][1]),
                (op1_5, tmp_list[3][0], batch_info[3][1], vc1_5), (op3_5, tmp_list[3][1], batch_info[3][1], vc2_5),
            ],
        })
        # Finish stage 5 batch 3 + compute addresses for next batch 3
        self.add_bundle({
            "alu": [("+", addr_A[i], self.scratch["forest_values_p"], next_batch_info[3][0] + i) for i in range(VLEN)],
            "valu": [(op2_5, batch_info[3][1], tmp_list[3][0], tmp_list[3][1])],
        })

        # Compute idx for all 4 batches + load next batch 3 elements 0-5
        self.add_bundle({
            "load": [("load", preload_nodes[3] + 0, addr_A[0]), ("load", preload_nodes[3] + 1, addr_A[1])],
            "valu": [
                ("multiply_add", batch_info[0][0], batch_info[0][0], v_two, v_one),
                ("multiply_add", batch_info[1][0], batch_info[1][0], v_two, v_one),
                ("multiply_add", batch_info[2][0], batch_info[2][0], v_two, v_one),
                ("multiply_add", batch_info[3][0], batch_info[3][0], v_two, v_one),
            ],
        })
        self.add_bundle({
            "load": [("load", preload_nodes[3] + 2, addr_A[2]), ("load", preload_nodes[3] + 3, addr_A[3])],
            "valu": [
                ("&", tmp_list[0][0], batch_info[0][1], v_one),
                ("&", tmp_list[1][0], batch_info[1][1], v_one),
                ("&", tmp_list[2][0], batch_info[2][1], v_one),
                ("&", tmp_list[3][0], batch_info[3][1], v_one),
            ],
        })
        self.add_bundle({
            "load": [("load", preload_nodes[3] + 4, addr_A[4]), ("load", preload_nodes[3] + 5, addr_A[5])],
            "valu": [
                ("+", batch_info[0][0], batch_info[0][0], tmp_list[0][0]),
                ("+", batch_info[1][0], batch_info[1][0], tmp_list[1][0]),
                ("+", batch_info[2][0], batch_info[2][0], tmp_list[2][0]),
                ("+", batch_info[3][0], batch_info[3][0], tmp_list[3][0]),
            ],
        })

        # Round loop control + finish loading next batch 3 elements 6-7
        self.add_bundle({
            "load": [("load", preload_nodes[3] + 6, addr_A[6]), ("load", preload_nodes[3] + 7, addr_A[7])],
            "flow": [("add_imm", round_counter, round_counter, 1)],
        })
        ten_const = self.scratch_const(10)
        self.add_bundle({"alu": [("<", tmp1, round_counter, ten_const)]})
        round_loop_offset = round_loop_start - len(self.instrs) - 1
        self.add_bundle({"flow": [("cond_jump_rel", tmp1, round_loop_offset)]})

        # ===== ROUND 10: Use pre-loaded nodes from node_set_C (pre-loaded by main loop's Group 7) =====
        # Group 0: Use pre-loaded nodes, pre-load group 1
        base = 0
        batch_info = [(v_idx[base + i], v_val[base + i]) for i in range(4)]
        nodes = node_set_C  # Use pre-loaded nodes from main loop's Group 7
        next_nodes = node_set_B  # Pre-load Group 1 into set B
        next_base = 4
        next_batch_info = [(v_idx[next_base + i], v_val[next_base + i]) for i in range(4)]

        # XOR all 4 batches + compute addresses for next batch 0
        self.add_bundle({
            "alu": [("+", addr_A[i], self.scratch["forest_values_p"], next_batch_info[0][0] + i) for i in range(VLEN)],
            "valu": [
                ("^", batch_info[0][1], batch_info[0][1], nodes[0]),
                ("^", batch_info[1][1], batch_info[1][1], nodes[1]),
                ("^", batch_info[2][1], batch_info[2][1], nodes[2]),
                ("^", batch_info[3][1], batch_info[3][1], nodes[3]),
            ],
        })

        # Hash stage 0 with multiply_add + load next batch 0 elements 0-1
        self.add_bundle({
            "load": [("load", next_nodes[0] + 0, addr_A[0]), ("load", next_nodes[0] + 1, addr_A[1])],
            "valu": [
                ("multiply_add", batch_info[0][1], batch_info[0][1], v_mult_4097, vc1_0),
                ("multiply_add", batch_info[1][1], batch_info[1][1], v_mult_4097, vc1_0),
                ("multiply_add", batch_info[2][1], batch_info[2][1], v_mult_4097, vc1_0),
                ("multiply_add", batch_info[3][1], batch_info[3][1], v_mult_4097, vc1_0),
            ],
        })

        # Hash stage 1 + load next batch 0 elements 2-7
        self.add_bundle({
            "load": [("load", next_nodes[0] + 2, addr_A[2]), ("load", next_nodes[0] + 3, addr_A[3])],
            "valu": [
                (op1_1, tmp_list[0][0], batch_info[0][1], vc1_1), (op3_1, tmp_list[0][1], batch_info[0][1], vc2_1),
                (op1_1, tmp_list[1][0], batch_info[1][1], vc1_1), (op3_1, tmp_list[1][1], batch_info[1][1], vc2_1),
            ],
        })
        self.add_bundle({
            "load": [("load", next_nodes[0] + 4, addr_A[4]), ("load", next_nodes[0] + 5, addr_A[5])],
            "valu": [
                (op2_1, batch_info[0][1], tmp_list[0][0], tmp_list[0][1]),
                (op2_1, batch_info[1][1], tmp_list[1][0], tmp_list[1][1]),
                (op1_1, tmp_list[2][0], batch_info[2][1], vc1_1), (op3_1, tmp_list[2][1], batch_info[2][1], vc2_1),
            ],
        })
        self.add_bundle({
            "load": [("load", next_nodes[0] + 6, addr_A[6]), ("load", next_nodes[0] + 7, addr_A[7])],
            "valu": [
                (op2_1, batch_info[2][1], tmp_list[2][0], tmp_list[2][1]),
                (op1_1, tmp_list[3][0], batch_info[3][1], vc1_1), (op3_1, tmp_list[3][1], batch_info[3][1], vc2_1),
            ],
        })
        # Finish stage 1 batch 3 + compute addresses for next batch 1
        self.add_bundle({
            "alu": [("+", addr_A[i], self.scratch["forest_values_p"], next_batch_info[1][0] + i) for i in range(VLEN)],
            "valu": [(op2_1, batch_info[3][1], tmp_list[3][0], tmp_list[3][1])],
        })

        # Hash stage 2 with multiply_add + load next batch 1 elements 0-1
        self.add_bundle({
            "load": [("load", next_nodes[1] + 0, addr_A[0]), ("load", next_nodes[1] + 1, addr_A[1])],
            "valu": [
                ("multiply_add", batch_info[0][1], batch_info[0][1], v_mult_33, vc1_2),
                ("multiply_add", batch_info[1][1], batch_info[1][1], v_mult_33, vc1_2),
                ("multiply_add", batch_info[2][1], batch_info[2][1], v_mult_33, vc1_2),
                ("multiply_add", batch_info[3][1], batch_info[3][1], v_mult_33, vc1_2),
            ],
        })

        # Hash stage 3 + load next batch 1 elements 2-7
        self.add_bundle({
            "load": [("load", next_nodes[1] + 2, addr_A[2]), ("load", next_nodes[1] + 3, addr_A[3])],
            "valu": [
                (op1_3, tmp_list[0][0], batch_info[0][1], vc1_3), (op3_3, tmp_list[0][1], batch_info[0][1], vc2_3),
                (op1_3, tmp_list[1][0], batch_info[1][1], vc1_3), (op3_3, tmp_list[1][1], batch_info[1][1], vc2_3),
            ],
        })
        self.add_bundle({
            "load": [("load", next_nodes[1] + 4, addr_A[4]), ("load", next_nodes[1] + 5, addr_A[5])],
            "valu": [
                (op2_3, batch_info[0][1], tmp_list[0][0], tmp_list[0][1]),
                (op2_3, batch_info[1][1], tmp_list[1][0], tmp_list[1][1]),
                (op1_3, tmp_list[2][0], batch_info[2][1], vc1_3), (op3_3, tmp_list[2][1], batch_info[2][1], vc2_3),
            ],
        })
        self.add_bundle({
            "load": [("load", next_nodes[1] + 6, addr_A[6]), ("load", next_nodes[1] + 7, addr_A[7])],
            "valu": [
                (op2_3, batch_info[2][1], tmp_list[2][0], tmp_list[2][1]),
                (op1_3, tmp_list[3][0], batch_info[3][1], vc1_3), (op3_3, tmp_list[3][1], batch_info[3][1], vc2_3),
            ],
        })
        # Finish stage 3 batch 3 + compute addresses for next batch 2
        self.add_bundle({
            "alu": [("+", addr_A[i], self.scratch["forest_values_p"], next_batch_info[2][0] + i) for i in range(VLEN)],
            "valu": [(op2_3, batch_info[3][1], tmp_list[3][0], tmp_list[3][1])],
        })

        # Hash stage 4 with multiply_add + load next batch 2 elements 0-1
        self.add_bundle({
            "load": [("load", next_nodes[2] + 0, addr_A[0]), ("load", next_nodes[2] + 1, addr_A[1])],
            "valu": [
                ("multiply_add", batch_info[0][1], batch_info[0][1], v_mult_9, vc1_4),
                ("multiply_add", batch_info[1][1], batch_info[1][1], v_mult_9, vc1_4),
                ("multiply_add", batch_info[2][1], batch_info[2][1], v_mult_9, vc1_4),
                ("multiply_add", batch_info[3][1], batch_info[3][1], v_mult_9, vc1_4),
            ],
        })

        # Hash stage 5 + load next batch 2 elements 2-7
        self.add_bundle({
            "load": [("load", next_nodes[2] + 2, addr_A[2]), ("load", next_nodes[2] + 3, addr_A[3])],
            "valu": [
                (op1_5, tmp_list[0][0], batch_info[0][1], vc1_5), (op3_5, tmp_list[0][1], batch_info[0][1], vc2_5),
                (op1_5, tmp_list[1][0], batch_info[1][1], vc1_5), (op3_5, tmp_list[1][1], batch_info[1][1], vc2_5),
            ],
        })
        self.add_bundle({
            "load": [("load", next_nodes[2] + 4, addr_A[4]), ("load", next_nodes[2] + 5, addr_A[5])],
            "valu": [
                (op2_5, batch_info[0][1], tmp_list[0][0], tmp_list[0][1]),
                (op2_5, batch_info[1][1], tmp_list[1][0], tmp_list[1][1]),
                (op1_5, tmp_list[2][0], batch_info[2][1], vc1_5), (op3_5, tmp_list[2][1], batch_info[2][1], vc2_5),
            ],
        })
        self.add_bundle({
            "load": [("load", next_nodes[2] + 6, addr_A[6]), ("load", next_nodes[2] + 7, addr_A[7])],
            "valu": [
                (op2_5, batch_info[2][1], tmp_list[2][0], tmp_list[2][1]),
                (op1_5, tmp_list[3][0], batch_info[3][1], vc1_5), (op3_5, tmp_list[3][1], batch_info[3][1], vc2_5),
            ],
        })
        # Finish stage 5 batch 3 + compute addresses for next batch 3
        self.add_bundle({
            "alu": [("+", addr_A[i], self.scratch["forest_values_p"], next_batch_info[3][0] + i) for i in range(VLEN)],
            "valu": [(op2_5, batch_info[3][1], tmp_list[3][0], tmp_list[3][1])],
        })

        # Compute idx for all 4 batches + load next batch 3 elements 0-7
        self.add_bundle({
            "load": [("load", next_nodes[3] + 0, addr_A[0]), ("load", next_nodes[3] + 1, addr_A[1])],
            "valu": [
                ("multiply_add", batch_info[0][0], batch_info[0][0], v_two, v_one),
                ("multiply_add", batch_info[1][0], batch_info[1][0], v_two, v_one),
                ("multiply_add", batch_info[2][0], batch_info[2][0], v_two, v_one),
                ("multiply_add", batch_info[3][0], batch_info[3][0], v_two, v_one),
            ],
        })
        self.add_bundle({
            "load": [("load", next_nodes[3] + 2, addr_A[2]), ("load", next_nodes[3] + 3, addr_A[3])],
            "valu": [
                ("&", tmp_list[0][0], batch_info[0][1], v_one),
                ("&", tmp_list[1][0], batch_info[1][1], v_one),
                ("&", tmp_list[2][0], batch_info[2][1], v_one),
                ("&", tmp_list[3][0], batch_info[3][1], v_one),
            ],
        })
        self.add_bundle({
            "load": [("load", next_nodes[3] + 4, addr_A[4]), ("load", next_nodes[3] + 5, addr_A[5])],
            "valu": [
                ("+", batch_info[0][0], batch_info[0][0], tmp_list[0][0]),
                ("+", batch_info[1][0], batch_info[1][0], tmp_list[1][0]),
                ("+", batch_info[2][0], batch_info[2][0], tmp_list[2][0]),
                ("+", batch_info[3][0], batch_info[3][0], tmp_list[3][0]),
            ],
        })
        self.add_bundle({"load": [("load", next_nodes[3] + 6, addr_A[6]), ("load", next_nodes[3] + 7, addr_A[7])]})

        # Bounds check for group 0 (all indices wrap to 0 after this)
        self.add_bundle({"valu": [("<", tmp_list[0][0], batch_info[0][0], v_n_nodes), ("<", tmp_list[1][0], batch_info[1][0], v_n_nodes), ("<", tmp_list[2][0], batch_info[2][0], v_n_nodes), ("<", tmp_list[3][0], batch_info[3][0], v_n_nodes)]})
        self.add_bundle({"flow": [("vselect", batch_info[0][0], tmp_list[0][0], batch_info[0][0], v_zero)]})
        self.add_bundle({"flow": [("vselect", batch_info[1][0], tmp_list[1][0], batch_info[1][0], v_zero)]})
        self.add_bundle({"flow": [("vselect", batch_info[2][0], tmp_list[2][0], batch_info[2][0], v_zero)]})
        self.add_bundle({"flow": [("vselect", batch_info[3][0], tmp_list[3][0], batch_info[3][0], v_zero)]})

        # Groups 1-6: Use pre-loaded nodes + pre-load next + bounds check
        for group in range(1, 7):
            base = group * 4
            batch_info = [(v_idx[base + i], v_val[base + i]) for i in range(4)]
            nodes = node_set_B if group % 2 == 1 else node_set_A
            next_nodes = node_set_A if group % 2 == 1 else node_set_B
            next_base = (group + 1) * 4
            next_batch_info = [(v_idx[next_base + i], v_val[next_base + i]) for i in range(4)]

            self.add_bundle({"alu": [("+", addr_A[i], self.scratch["forest_values_p"], next_batch_info[0][0] + i) for i in range(VLEN)], "valu": [("^", batch_info[0][1], batch_info[0][1], nodes[0]), ("^", batch_info[1][1], batch_info[1][1], nodes[1]), ("^", batch_info[2][1], batch_info[2][1], nodes[2]), ("^", batch_info[3][1], batch_info[3][1], nodes[3])]})
            self.add_bundle({"alu": [("+", addr_B[i], self.scratch["forest_values_p"], next_batch_info[1][0] + i) for i in range(VLEN)], "valu": [(op1_0, tmp_list[0][0], batch_info[0][1], vc1_0), (op3_0, tmp_list[0][1], batch_info[0][1], vc2_0), (op1_0, tmp_list[1][0], batch_info[1][1], vc1_0), (op3_0, tmp_list[1][1], batch_info[1][1], vc2_0)]})
            self.add_bundle({"load": [("load", next_nodes[0] + 0, addr_A[0]), ("load", next_nodes[1] + 0, addr_B[0])], "valu": [(op2_0, batch_info[0][1], tmp_list[0][0], tmp_list[0][1]), (op2_0, batch_info[1][1], tmp_list[1][0], tmp_list[1][1]), (op1_0, tmp_list[2][0], batch_info[2][1], vc1_0), (op3_0, tmp_list[2][1], batch_info[2][1], vc2_0)]})
            self.add_bundle({"load": [("load", next_nodes[0] + 1, addr_A[1]), ("load", next_nodes[1] + 1, addr_B[1])], "valu": [(op1_1, tmp_list[0][0], batch_info[0][1], vc1_1), (op3_1, tmp_list[0][1], batch_info[0][1], vc2_1), (op1_1, tmp_list[1][0], batch_info[1][1], vc1_1), (op3_1, tmp_list[1][1], batch_info[1][1], vc2_1)]})
            self.add_bundle({"load": [("load", next_nodes[0] + 2, addr_A[2]), ("load", next_nodes[1] + 2, addr_B[2])], "valu": [(op2_1, batch_info[0][1], tmp_list[0][0], tmp_list[0][1]), (op2_1, batch_info[1][1], tmp_list[1][0], tmp_list[1][1]), (op2_0, batch_info[2][1], tmp_list[2][0], tmp_list[2][1]), (op1_0, tmp_list[3][0], batch_info[3][1], vc1_0), (op3_0, tmp_list[3][1], batch_info[3][1], vc2_0)]})
            self.add_bundle({"load": [("load", next_nodes[0] + 3, addr_A[3]), ("load", next_nodes[1] + 3, addr_B[3])], "valu": [(op1_2, tmp_list[0][0], batch_info[0][1], vc1_2), (op3_2, tmp_list[0][1], batch_info[0][1], vc2_2), (op1_2, tmp_list[1][0], batch_info[1][1], vc1_2), (op3_2, tmp_list[1][1], batch_info[1][1], vc2_2)]})
            self.add_bundle({"load": [("load", next_nodes[0] + 4, addr_A[4]), ("load", next_nodes[1] + 4, addr_B[4])], "valu": [(op2_2, batch_info[0][1], tmp_list[0][0], tmp_list[0][1]), (op2_2, batch_info[1][1], tmp_list[1][0], tmp_list[1][1]), (op1_1, tmp_list[2][0], batch_info[2][1], vc1_1), (op3_1, tmp_list[2][1], batch_info[2][1], vc2_1)]})
            self.add_bundle({"load": [("load", next_nodes[0] + 5, addr_A[5]), ("load", next_nodes[1] + 5, addr_B[5])], "valu": [(op1_3, tmp_list[0][0], batch_info[0][1], vc1_3), (op3_3, tmp_list[0][1], batch_info[0][1], vc2_3), (op1_3, tmp_list[1][0], batch_info[1][1], vc1_3), (op3_3, tmp_list[1][1], batch_info[1][1], vc2_3)]})
            self.add_bundle({"load": [("load", next_nodes[0] + 6, addr_A[6]), ("load", next_nodes[1] + 6, addr_B[6])], "valu": [(op2_3, batch_info[0][1], tmp_list[0][0], tmp_list[0][1]), (op2_3, batch_info[1][1], tmp_list[1][0], tmp_list[1][1]), (op2_1, batch_info[2][1], tmp_list[2][0], tmp_list[2][1]), (op2_0, batch_info[3][1], tmp_list[3][0], tmp_list[3][1])]})
            self.add_bundle({"load": [("load", next_nodes[0] + 7, addr_A[7]), ("load", next_nodes[1] + 7, addr_B[7])], "valu": [(op1_4, tmp_list[0][0], batch_info[0][1], vc1_4), (op3_4, tmp_list[0][1], batch_info[0][1], vc2_4), (op1_4, tmp_list[1][0], batch_info[1][1], vc1_4), (op3_4, tmp_list[1][1], batch_info[1][1], vc2_4)]})
            self.add_bundle({"alu": [("+", addr_A[i], self.scratch["forest_values_p"], next_batch_info[2][0] + i) for i in range(VLEN)], "valu": [(op2_4, batch_info[0][1], tmp_list[0][0], tmp_list[0][1]), (op2_4, batch_info[1][1], tmp_list[1][0], tmp_list[1][1]), (op1_2, tmp_list[2][0], batch_info[2][1], vc1_2), (op3_2, tmp_list[2][1], batch_info[2][1], vc2_2)]})
            self.add_bundle({"alu": [("+", addr_B[i], self.scratch["forest_values_p"], next_batch_info[3][0] + i) for i in range(VLEN)], "valu": [(op1_5, tmp_list[0][0], batch_info[0][1], vc1_5), (op3_5, tmp_list[0][1], batch_info[0][1], vc2_5), (op1_5, tmp_list[1][0], batch_info[1][1], vc1_5), (op3_5, tmp_list[1][1], batch_info[1][1], vc2_5)]})
            self.add_bundle({"load": [("load", next_nodes[2] + 0, addr_A[0]), ("load", next_nodes[3] + 0, addr_B[0])], "valu": [(op2_5, batch_info[0][1], tmp_list[0][0], tmp_list[0][1]), (op2_5, batch_info[1][1], tmp_list[1][0], tmp_list[1][1]), (op2_2, batch_info[2][1], tmp_list[2][0], tmp_list[2][1]), (op1_1, tmp_list[3][0], batch_info[3][1], vc1_1), (op3_1, tmp_list[3][1], batch_info[3][1], vc2_1)]})
            self.add_bundle({"load": [("load", next_nodes[2] + 1, addr_A[1]), ("load", next_nodes[3] + 1, addr_B[1])], "valu": [("&", tmp_list[0][0], batch_info[0][1], v_one), ("<<", batch_info[0][0], batch_info[0][0], v_one), ("&", tmp_list[1][0], batch_info[1][1], v_one), ("<<", batch_info[1][0], batch_info[1][0], v_one)]})
            self.add_bundle({"load": [("load", next_nodes[2] + 2, addr_A[2]), ("load", next_nodes[3] + 2, addr_B[2])], "valu": [("+", tmp_list[0][0], tmp_list[0][0], v_one), ("+", tmp_list[1][0], tmp_list[1][0], v_one), (op1_3, tmp_list[2][0], batch_info[2][1], vc1_3), (op3_3, tmp_list[2][1], batch_info[2][1], vc2_3)]})
            self.add_bundle({"load": [("load", next_nodes[2] + 3, addr_A[3]), ("load", next_nodes[3] + 3, addr_B[3])], "valu": [("+", batch_info[0][0], batch_info[0][0], tmp_list[0][0]), ("+", batch_info[1][0], batch_info[1][0], tmp_list[1][0]), (op2_3, batch_info[2][1], tmp_list[2][0], tmp_list[2][1]), (op2_1, batch_info[3][1], tmp_list[3][0], tmp_list[3][1])]})
            self.add_bundle({"load": [("load", next_nodes[2] + 4, addr_A[4]), ("load", next_nodes[3] + 4, addr_B[4])], "valu": [(op1_4, tmp_list[2][0], batch_info[2][1], vc1_4), (op3_4, tmp_list[2][1], batch_info[2][1], vc2_4), (op1_2, tmp_list[3][0], batch_info[3][1], vc1_2), (op3_2, tmp_list[3][1], batch_info[3][1], vc2_2)]})
            self.add_bundle({"load": [("load", next_nodes[2] + 5, addr_A[5]), ("load", next_nodes[3] + 5, addr_B[5])], "valu": [(op2_4, batch_info[2][1], tmp_list[2][0], tmp_list[2][1]), (op2_2, batch_info[3][1], tmp_list[3][0], tmp_list[3][1])]})
            self.add_bundle({"load": [("load", next_nodes[2] + 6, addr_A[6]), ("load", next_nodes[3] + 6, addr_B[6])], "valu": [(op1_5, tmp_list[2][0], batch_info[2][1], vc1_5), (op3_5, tmp_list[2][1], batch_info[2][1], vc2_5), (op1_3, tmp_list[3][0], batch_info[3][1], vc1_3), (op3_3, tmp_list[3][1], batch_info[3][1], vc2_3)]})
            self.add_bundle({"load": [("load", next_nodes[2] + 7, addr_A[7]), ("load", next_nodes[3] + 7, addr_B[7])], "valu": [(op2_5, batch_info[2][1], tmp_list[2][0], tmp_list[2][1]), (op2_3, batch_info[3][1], tmp_list[3][0], tmp_list[3][1])]})
            # Finish idx with multiply_add (OPTIMIZED)
            self.add_bundle({"valu": [("multiply_add", batch_info[2][0], batch_info[2][0], v_two, v_one), ("&", tmp_list[2][0], batch_info[2][1], v_one), (op1_4, tmp_list[3][0], batch_info[3][1], vc1_4), (op3_4, tmp_list[3][1], batch_info[3][1], vc2_4)]})
            self.add_bundle({"valu": [("+", batch_info[2][0], batch_info[2][0], tmp_list[2][0]), (op2_4, batch_info[3][1], tmp_list[3][0], tmp_list[3][1])]})
            self.add_bundle({"valu": [(op1_5, tmp_list[3][0], batch_info[3][1], vc1_5), (op3_5, tmp_list[3][1], batch_info[3][1], vc2_5)]})
            self.add_bundle({"valu": [(op2_5, batch_info[3][1], tmp_list[3][0], tmp_list[3][1])]})
            self.add_bundle({"valu": [("multiply_add", batch_info[3][0], batch_info[3][0], v_two, v_one), ("&", tmp_list[3][0], batch_info[3][1], v_one)]})
            self.add_bundle({"valu": [("+", batch_info[3][0], batch_info[3][0], tmp_list[3][0])]})
            # Bounds check
            self.add_bundle({"valu": [("<", tmp_list[0][0], batch_info[0][0], v_n_nodes), ("<", tmp_list[1][0], batch_info[1][0], v_n_nodes), ("<", tmp_list[2][0], batch_info[2][0], v_n_nodes), ("<", tmp_list[3][0], batch_info[3][0], v_n_nodes)]})
            self.add_bundle({"flow": [("vselect", batch_info[0][0], tmp_list[0][0], batch_info[0][0], v_zero)]})
            self.add_bundle({"flow": [("vselect", batch_info[1][0], tmp_list[1][0], batch_info[1][0], v_zero)]})
            self.add_bundle({"flow": [("vselect", batch_info[2][0], tmp_list[2][0], batch_info[2][0], v_zero)]})
            self.add_bundle({"flow": [("vselect", batch_info[3][0], tmp_list[3][0], batch_info[3][0], v_zero)]})

        # Group 7: Use pre-loaded nodes, no next group
        base = 7 * 4
        batch_info = [(v_idx[base + i], v_val[base + i]) for i in range(4)]
        nodes = node_set_B

        self.add_bundle({"valu": [("^", batch_info[0][1], batch_info[0][1], nodes[0]), ("^", batch_info[1][1], batch_info[1][1], nodes[1]), ("^", batch_info[2][1], batch_info[2][1], nodes[2]), ("^", batch_info[3][1], batch_info[3][1], nodes[3])]})
        self.add_bundle({"valu": [(op1_0, tmp_list[0][0], batch_info[0][1], vc1_0), (op3_0, tmp_list[0][1], batch_info[0][1], vc2_0), (op1_0, tmp_list[1][0], batch_info[1][1], vc1_0), (op3_0, tmp_list[1][1], batch_info[1][1], vc2_0)]})
        self.add_bundle({"valu": [(op2_0, batch_info[0][1], tmp_list[0][0], tmp_list[0][1]), (op2_0, batch_info[1][1], tmp_list[1][0], tmp_list[1][1]), (op1_0, tmp_list[2][0], batch_info[2][1], vc1_0), (op3_0, tmp_list[2][1], batch_info[2][1], vc2_0)]})
        self.add_bundle({"valu": [(op1_1, tmp_list[0][0], batch_info[0][1], vc1_1), (op3_1, tmp_list[0][1], batch_info[0][1], vc2_1), (op1_1, tmp_list[1][0], batch_info[1][1], vc1_1), (op3_1, tmp_list[1][1], batch_info[1][1], vc2_1)]})
        self.add_bundle({"valu": [(op2_1, batch_info[0][1], tmp_list[0][0], tmp_list[0][1]), (op2_1, batch_info[1][1], tmp_list[1][0], tmp_list[1][1]), (op2_0, batch_info[2][1], tmp_list[2][0], tmp_list[2][1]), (op1_0, tmp_list[3][0], batch_info[3][1], vc1_0), (op3_0, tmp_list[3][1], batch_info[3][1], vc2_0)]})
        self.add_bundle({"valu": [(op1_2, tmp_list[0][0], batch_info[0][1], vc1_2), (op3_2, tmp_list[0][1], batch_info[0][1], vc2_2), (op1_2, tmp_list[1][0], batch_info[1][1], vc1_2), (op3_2, tmp_list[1][1], batch_info[1][1], vc2_2)]})
        self.add_bundle({"valu": [(op2_2, batch_info[0][1], tmp_list[0][0], tmp_list[0][1]), (op2_2, batch_info[1][1], tmp_list[1][0], tmp_list[1][1]), (op1_1, tmp_list[2][0], batch_info[2][1], vc1_1), (op3_1, tmp_list[2][1], batch_info[2][1], vc2_1)]})
        self.add_bundle({"valu": [(op1_3, tmp_list[0][0], batch_info[0][1], vc1_3), (op3_3, tmp_list[0][1], batch_info[0][1], vc2_3), (op1_3, tmp_list[1][0], batch_info[1][1], vc1_3), (op3_3, tmp_list[1][1], batch_info[1][1], vc2_3)]})
        self.add_bundle({"valu": [(op2_3, batch_info[0][1], tmp_list[0][0], tmp_list[0][1]), (op2_3, batch_info[1][1], tmp_list[1][0], tmp_list[1][1]), (op2_1, batch_info[2][1], tmp_list[2][0], tmp_list[2][1]), (op2_0, batch_info[3][1], tmp_list[3][0], tmp_list[3][1])]})
        self.add_bundle({"valu": [(op1_4, tmp_list[0][0], batch_info[0][1], vc1_4), (op3_4, tmp_list[0][1], batch_info[0][1], vc2_4), (op1_4, tmp_list[1][0], batch_info[1][1], vc1_4), (op3_4, tmp_list[1][1], batch_info[1][1], vc2_4)]})
        self.add_bundle({"valu": [(op2_4, batch_info[0][1], tmp_list[0][0], tmp_list[0][1]), (op2_4, batch_info[1][1], tmp_list[1][0], tmp_list[1][1]), (op1_2, tmp_list[2][0], batch_info[2][1], vc1_2), (op3_2, tmp_list[2][1], batch_info[2][1], vc2_2)]})
        self.add_bundle({"valu": [(op1_5, tmp_list[0][0], batch_info[0][1], vc1_5), (op3_5, tmp_list[0][1], batch_info[0][1], vc2_5), (op1_5, tmp_list[1][0], batch_info[1][1], vc1_5), (op3_5, tmp_list[1][1], batch_info[1][1], vc2_5)]})
        self.add_bundle({"valu": [(op2_5, batch_info[0][1], tmp_list[0][0], tmp_list[0][1]), (op2_5, batch_info[1][1], tmp_list[1][0], tmp_list[1][1]), (op2_2, batch_info[2][1], tmp_list[2][0], tmp_list[2][1]), (op1_1, tmp_list[3][0], batch_info[3][1], vc1_1), (op3_1, tmp_list[3][1], batch_info[3][1], vc2_1)]})
        self.add_bundle({"valu": [("&", tmp_list[0][0], batch_info[0][1], v_one), ("<<", batch_info[0][0], batch_info[0][0], v_one), ("&", tmp_list[1][0], batch_info[1][1], v_one), ("<<", batch_info[1][0], batch_info[1][0], v_one)]})
        self.add_bundle({"valu": [("+", tmp_list[0][0], tmp_list[0][0], v_one), ("+", tmp_list[1][0], tmp_list[1][0], v_one), (op1_3, tmp_list[2][0], batch_info[2][1], vc1_3), (op3_3, tmp_list[2][1], batch_info[2][1], vc2_3)]})
        self.add_bundle({"valu": [("+", batch_info[0][0], batch_info[0][0], tmp_list[0][0]), ("+", batch_info[1][0], batch_info[1][0], tmp_list[1][0]), (op2_3, batch_info[2][1], tmp_list[2][0], tmp_list[2][1]), (op2_1, batch_info[3][1], tmp_list[3][0], tmp_list[3][1])]})
        self.add_bundle({"valu": [(op1_4, tmp_list[2][0], batch_info[2][1], vc1_4), (op3_4, tmp_list[2][1], batch_info[2][1], vc2_4), (op1_2, tmp_list[3][0], batch_info[3][1], vc1_2), (op3_2, tmp_list[3][1], batch_info[3][1], vc2_2)]})
        self.add_bundle({"valu": [(op2_4, batch_info[2][1], tmp_list[2][0], tmp_list[2][1]), (op2_2, batch_info[3][1], tmp_list[3][0], tmp_list[3][1])]})
        self.add_bundle({"valu": [(op1_5, tmp_list[2][0], batch_info[2][1], vc1_5), (op3_5, tmp_list[2][1], batch_info[2][1], vc2_5), (op1_3, tmp_list[3][0], batch_info[3][1], vc1_3), (op3_3, tmp_list[3][1], batch_info[3][1], vc2_3)]})
        self.add_bundle({"valu": [(op2_5, batch_info[2][1], tmp_list[2][0], tmp_list[2][1]), (op2_3, batch_info[3][1], tmp_list[3][0], tmp_list[3][1])]})
        # Finish idx with multiply_add (OPTIMIZED - Group 7)
        self.add_bundle({"valu": [("multiply_add", batch_info[2][0], batch_info[2][0], v_two, v_one), ("&", tmp_list[2][0], batch_info[2][1], v_one), (op1_4, tmp_list[3][0], batch_info[3][1], vc1_4), (op3_4, tmp_list[3][1], batch_info[3][1], vc2_4)]})
        self.add_bundle({"valu": [("+", batch_info[2][0], batch_info[2][0], tmp_list[2][0]), (op2_4, batch_info[3][1], tmp_list[3][0], tmp_list[3][1])]})
        self.add_bundle({"valu": [(op1_5, tmp_list[3][0], batch_info[3][1], vc1_5), (op3_5, tmp_list[3][1], batch_info[3][1], vc2_5)]})
        self.add_bundle({"valu": [(op2_5, batch_info[3][1], tmp_list[3][0], tmp_list[3][1])]})
        self.add_bundle({"valu": [("multiply_add", batch_info[3][0], batch_info[3][0], v_two, v_one), ("&", tmp_list[3][0], batch_info[3][1], v_one)]})
        self.add_bundle({"valu": [("+", batch_info[3][0], batch_info[3][0], tmp_list[3][0])]})
        # Bounds check
        self.add_bundle({"valu": [("<", tmp_list[0][0], batch_info[0][0], v_n_nodes), ("<", tmp_list[1][0], batch_info[1][0], v_n_nodes), ("<", tmp_list[2][0], batch_info[2][0], v_n_nodes), ("<", tmp_list[3][0], batch_info[3][0], v_n_nodes)]})
        self.add_bundle({"flow": [("vselect", batch_info[0][0], tmp_list[0][0], batch_info[0][0], v_zero)]})
        self.add_bundle({"flow": [("vselect", batch_info[1][0], tmp_list[1][0], batch_info[1][0], v_zero)]})
        self.add_bundle({"flow": [("vselect", batch_info[2][0], tmp_list[2][0], batch_info[2][0], v_zero)]})
        self.add_bundle({"flow": [("vselect", batch_info[3][0], tmp_list[3][0], batch_info[3][0], v_zero)]})

        # ===== ROUNDS 11-15: Unrolled (mirror rounds 0-4 after wrapping) =====
        # After round 10, ALL indices wrap to 0!

        # Round 11 (like round 0): all indices are 0 - process 3 batches at a time
        # OPTIMIZATION: Reuse v_node_shared from round 0 (forest[0] doesn't change)

        for group in range(11):
            if group < 10:
                b = group * 3
                val_A, val_B, val_C = v_val[b], v_val[b + 1], v_val[b + 2]
                idx_A, idx_B, idx_C = v_idx[b], v_idx[b + 1], v_idx[b + 2]

                self.add_bundle({"valu": [
                    ("^", val_A, val_A, v_node_shared), ("^", val_B, val_B, v_node_shared),
                    ("^", val_C, val_C, v_node_shared),
                ]})

                # Hash stages 0-5 with multiply_add for stages 0, 2, 4
                mult_consts = [v_mult_4097, None, v_mult_33, None, v_mult_9, None]
                for hi in range(6):
                    vc1, vc2 = v_hash_consts[hi]
                    op1, _, op2, op3, _ = HASH_STAGES[hi]
                    if hi in [0, 2, 4]:
                        self.add_bundle({"valu": [
                            ("multiply_add", val_A, val_A, mult_consts[hi], vc1),
                            ("multiply_add", val_B, val_B, mult_consts[hi], vc1),
                            ("multiply_add", val_C, val_C, mult_consts[hi], vc1),
                        ]})
                    else:
                        self.add_bundle({"valu": [
                            (op1, v_tmp1_A, val_A, vc1), (op3, v_tmp2_A, val_A, vc2),
                            (op1, v_tmp1_B, val_B, vc1), (op3, v_tmp2_B, val_B, vc2),
                            (op1, v_tmp1_E, val_C, vc1), (op3, v_tmp2_E, val_C, vc2),
                        ]})
                        self.add_bundle({"valu": [
                            (op2, val_A, v_tmp1_A, v_tmp2_A),
                            (op2, val_B, v_tmp1_B, v_tmp2_B),
                            (op2, val_C, v_tmp1_E, v_tmp2_E),
                        ]})

                self.add_bundle({"valu": [
                    ("&", idx_A, val_A, v_one), ("&", idx_B, val_B, v_one), ("&", idx_C, val_C, v_one),
                ]})
                self.add_bundle({"valu": [
                    ("+", idx_A, idx_A, v_one), ("+", idx_B, idx_B, v_one), ("+", idx_C, idx_C, v_one),
                ]})
            else:
                b = 30
                val_A, val_B = v_val[b], v_val[b + 1]
                idx_A, idx_B = v_idx[b], v_idx[b + 1]
                self.add_bundle({"valu": [("^", val_A, val_A, v_node_shared), ("^", val_B, val_B, v_node_shared)]})
                mult_consts = [v_mult_4097, None, v_mult_33, None, v_mult_9, None]
                for hi in range(6):
                    vc1, vc2 = v_hash_consts[hi]
                    op1, _, op2, op3, _ = HASH_STAGES[hi]
                    if hi in [0, 2, 4]:
                        self.add_bundle({"valu": [
                            ("multiply_add", val_A, val_A, mult_consts[hi], vc1),
                            ("multiply_add", val_B, val_B, mult_consts[hi], vc1),
                        ]})
                    else:
                        self.add_bundle({"valu": [
                            (op1, v_tmp1_A, val_A, vc1), (op3, v_tmp2_A, val_A, vc2),
                            (op1, v_tmp1_B, val_B, vc1), (op3, v_tmp2_B, val_B, vc2),
                        ]})
                        self.add_bundle({"valu": [(op2, val_A, v_tmp1_A, v_tmp2_A), (op2, val_B, v_tmp1_B, v_tmp2_B)]})
                self.add_bundle({"valu": [("&", idx_A, val_A, v_one), ("&", idx_B, val_B, v_one)]})
                self.add_bundle({"valu": [("+", idx_A, idx_A, v_one), ("+", idx_B, idx_B, v_one)]})

        # Round 12 (like round 1): indices in {1,2}
        # OPTIMIZATION: Reuse v_node1, v_node2 from round 1 (forest[1,2] don't change)

        for b in range(0, num_batches, 2):
            val_A, val_B = v_val[b], v_val[b + 1]
            idx_A, idx_B = v_idx[b], v_idx[b + 1]
            # Select: idx==1 -> v_node1, idx==2 -> v_node2
            self.add_bundle({"valu": [("==", v_tmp1_A, idx_A, v_one), ("==", v_tmp1_B, idx_B, v_one)]})
            self.add_bundle({"flow": [("vselect", v_node_A, v_tmp1_A, v_node1, v_node2)]})
            self.add_bundle({"flow": [("vselect", v_node_B, v_tmp1_B, v_node1, v_node2)]})
            self.add_bundle({"valu": [("^", val_A, val_A, v_node_A), ("^", val_B, val_B, v_node_B)]})
            # Hash with multiply_add for stages 0, 2, 4
            mult_consts = [v_mult_4097, None, v_mult_33, None, v_mult_9, None]
            for hi in range(6):
                vc1, vc2 = v_hash_consts[hi]
                op1, _, op2, op3, _ = HASH_STAGES[hi]
                if hi in [0, 2, 4]:
                    self.add_bundle({"valu": [
                        ("multiply_add", val_A, val_A, mult_consts[hi], vc1),
                        ("multiply_add", val_B, val_B, mult_consts[hi], vc1),
                    ]})
                else:
                    self.add_bundle({"valu": [
                        (op1, v_tmp1_A, val_A, vc1), (op3, v_tmp2_A, val_A, vc2),
                        (op1, v_tmp1_B, val_B, vc1), (op3, v_tmp2_B, val_B, vc2),
                    ]})
                    self.add_bundle({"valu": [(op2, val_A, v_tmp1_A, v_tmp2_A), (op2, val_B, v_tmp1_B, v_tmp2_B)]})
            # idx = 2*idx + 1 + (val&1) using multiply_add
            self.add_bundle({"valu": [("&", v_tmp1_A, val_A, v_one), ("&", v_tmp1_B, val_B, v_one)]})
            self.add_bundle({"valu": [
                ("multiply_add", idx_A, idx_A, v_two, v_one),
                ("multiply_add", idx_B, idx_B, v_two, v_one),
            ]})
            self.add_bundle({"valu": [("+", idx_A, idx_A, v_tmp1_A), ("+", idx_B, idx_B, v_tmp1_B)]})
            # No bounds check needed - indices will be {3,4,5,6} < n_nodes

        # Round 13 (like round 2): indices in {3,4,5,6}
        # OPTIMIZATION: Reuse v_f3, v_f4, v_f5, v_f6 from round 2 (forest[3..6] don't change)

        # Pair 0: full processing without overlap
        val_A, val_B = v_val[0], v_val[1]
        idx_A, idx_B = v_idx[0], v_idx[1]

        self.add_bundle({"valu": [
            ("&", v_tmp1_A, idx_A, v_one), (">>", v_tmp2_A, idx_A, v_one),
            ("&", v_tmp1_B, idx_B, v_one), (">>", v_tmp2_B, idx_B, v_one),
        ]})
        self.add_bundle({"valu": [("&", v_tmp2_A, v_tmp2_A, v_one), ("&", v_tmp2_B, v_tmp2_B, v_one)]})
        self.add_bundle({"flow": [("vselect", v_r_odd, v_tmp2_A, v_f3, v_f5)]})
        self.add_bundle({"flow": [("vselect", v_r_even, v_tmp2_A, v_f6, v_f4)]})
        self.add_bundle({"flow": [("vselect", v_node_A, v_tmp1_A, v_r_odd, v_r_even)]})
        self.add_bundle({"flow": [("vselect", v_r_odd, v_tmp2_B, v_f3, v_f5)]})
        self.add_bundle({"flow": [("vselect", v_r_even, v_tmp2_B, v_f6, v_f4)]})
        self.add_bundle({"flow": [("vselect", v_node_B, v_tmp1_B, v_r_odd, v_r_even)]})
        self.add_bundle({"valu": [("^", val_A, val_A, v_node_A), ("^", val_B, val_B, v_node_B)]})

        # Hash stages 0-3 with multiply_add for stages 0, 2
        mult_consts = [v_mult_4097, None, v_mult_33, None]
        for hi in range(4):
            vc1, vc2 = v_hash_consts[hi]
            op1, _, op2, op3, _ = HASH_STAGES[hi]
            if hi in [0, 2]:
                self.add_bundle({"valu": [
                    ("multiply_add", val_A, val_A, mult_consts[hi], vc1),
                    ("multiply_add", val_B, val_B, mult_consts[hi], vc1),
                ]})
            else:
                self.add_bundle({"valu": [
                    (op1, v_tmp1_A, val_A, vc1), (op3, v_tmp2_A, val_A, vc2),
                    (op1, v_tmp1_B, val_B, vc1), (op3, v_tmp2_B, val_B, vc2),
                ]})
                self.add_bundle({"valu": [(op2, val_A, v_tmp1_A, v_tmp2_A), (op2, val_B, v_tmp1_B, v_tmp2_B)]})

        prev_val_A, prev_val_B = val_A, val_B
        prev_idx_A, prev_idx_B = idx_A, idx_B

        # Pairs 1-15: pipelined with vselect overlapping hash finish
        for b in range(2, num_batches, 2):
            val_A, val_B = v_val[b], v_val[b + 1]
            idx_A, idx_B = v_idx[b], v_idx[b + 1]

            vc1_4, vc2_4 = v_hash_consts[4]
            op1_4, _, op2_4, op3_4, _ = HASH_STAGES[4]
            self.add_bundle({"valu": [
                ("&", v_bit0_C, idx_A, v_one), (">>", v_bit1_C, idx_A, v_one),
                (op1_4, v_tmp1_A, prev_val_A, vc1_4), (op3_4, v_tmp2_A, prev_val_A, vc2_4),
            ]})
            self.add_bundle({"valu": [
                ("&", v_bit0_D, idx_B, v_one), (">>", v_bit1_D, idx_B, v_one),
                (op2_4, prev_val_A, v_tmp1_A, v_tmp2_A),
                (op1_4, v_tmp1_B, prev_val_B, vc1_4), (op3_4, v_tmp2_B, prev_val_B, vc2_4),
            ]})
            vc1_5, vc2_5 = v_hash_consts[5]
            op1_5, _, op2_5, op3_5, _ = HASH_STAGES[5]
            self.add_bundle({"valu": [
                ("&", v_bit1_C, v_bit1_C, v_one), ("&", v_bit1_D, v_bit1_D, v_one),
                (op2_4, prev_val_B, v_tmp1_B, v_tmp2_B),
            ]})
            self.add_bundle({
                "flow": [("vselect", v_r_odd, v_bit1_C, v_f3, v_f5)],
                "valu": [(op1_5, v_tmp1_A, prev_val_A, vc1_5), (op3_5, v_tmp2_A, prev_val_A, vc2_5)],
            })
            self.add_bundle({
                "flow": [("vselect", v_r_even, v_bit1_C, v_f6, v_f4)],
                "valu": [(op2_5, prev_val_A, v_tmp1_A, v_tmp2_A)],
            })
            self.add_bundle({
                "flow": [("vselect", v_node_A, v_bit0_C, v_r_odd, v_r_even)],
                "valu": [(op1_5, v_tmp1_B, prev_val_B, vc1_5), (op3_5, v_tmp2_B, prev_val_B, vc2_5)],
            })
            self.add_bundle({
                "flow": [("vselect", v_r_odd, v_bit1_D, v_f3, v_f5)],
                "valu": [(op2_5, prev_val_B, v_tmp1_B, v_tmp2_B)],
            })
            self.add_bundle({
                "flow": [("vselect", v_r_even, v_bit1_D, v_f6, v_f4)],
                "valu": [("&", v_tmp1_A, prev_val_A, v_one), ("<<", prev_idx_A, prev_idx_A, v_one)],
            })
            self.add_bundle({
                "flow": [("vselect", v_node_B, v_bit0_D, v_r_odd, v_r_even)],
                "valu": [
                    ("+", v_tmp1_A, v_tmp1_A, v_one),
                    ("&", v_tmp1_B, prev_val_B, v_one), ("<<", prev_idx_B, prev_idx_B, v_one),
                ],
            })
            self.add_bundle({"valu": [
                ("+", prev_idx_A, prev_idx_A, v_tmp1_A),
                ("+", v_tmp1_B, v_tmp1_B, v_one),
            ]})
            self.add_bundle({"valu": [
                ("+", prev_idx_B, prev_idx_B, v_tmp1_B),
                ("^", val_A, val_A, v_node_A), ("^", val_B, val_B, v_node_B),
            ]})

            # Hash stages 0-3 with multiply_add for stages 0, 2
            mult_consts = [v_mult_4097, None, v_mult_33, None]
            for hi in range(4):
                vc1, vc2 = v_hash_consts[hi]
                op1, _, op2, op3, _ = HASH_STAGES[hi]
                if hi in [0, 2]:
                    self.add_bundle({"valu": [
                        ("multiply_add", val_A, val_A, mult_consts[hi], vc1),
                        ("multiply_add", val_B, val_B, mult_consts[hi], vc1),
                    ]})
                else:
                    self.add_bundle({"valu": [
                        (op1, v_tmp1_A, val_A, vc1), (op3, v_tmp2_A, val_A, vc2),
                        (op1, v_tmp1_B, val_B, vc1), (op3, v_tmp2_B, val_B, vc2),
                    ]})
                    self.add_bundle({"valu": [(op2, val_A, v_tmp1_A, v_tmp2_A), (op2, val_B, v_tmp1_B, v_tmp2_B)]})

            prev_val_A, prev_val_B = val_A, val_B
            prev_idx_A, prev_idx_B = idx_A, idx_B

        # Finish last pair - OVERLAPPED with pre-loading Round 14's batch 0
        vc1_4, vc2_4 = v_hash_consts[4]
        op1_4, _, op2_4, op3_4, _ = HASH_STAGES[4]
        vc1_5, vc2_5 = v_hash_consts[5]
        op1_5, _, op2_5, op3_5, _ = HASH_STAGES[5]

        # Set up pre-loading for Round 14
        base_14 = 0
        batch_info_14 = [(v_idx[base_14 + i], v_val[base_14 + i]) for i in range(4)]
        nodes_14 = node_set_A
        preload_batch_0_idx = batch_info_14[0][0]  # v_idx[0] already computed

        # Compute addresses for batch 0 while doing hash stage 4
        self.add_bundle({
            "alu": [("+", addr_A[i], self.scratch["forest_values_p"], preload_batch_0_idx + i) for i in range(VLEN)],
            "valu": [
                (op1_4, v_tmp1_A, prev_val_A, vc1_4), (op3_4, v_tmp2_A, prev_val_A, vc2_4),
                (op1_4, v_tmp1_B, prev_val_B, vc1_4), (op3_4, v_tmp2_B, prev_val_B, vc2_4),
            ],
        })
        # Load batch 0 elements 0-1 while finishing hash 4
        self.add_bundle({
            "load": [("load", nodes_14[0] + 0, addr_A[0]), ("load", nodes_14[0] + 1, addr_A[1])],
            "valu": [(op2_4, prev_val_A, v_tmp1_A, v_tmp2_A), (op2_4, prev_val_B, v_tmp1_B, v_tmp2_B)],
        })
        # Load batch 0 elements 2-3 while doing hash stage 5
        self.add_bundle({
            "load": [("load", nodes_14[0] + 2, addr_A[2]), ("load", nodes_14[0] + 3, addr_A[3])],
            "valu": [
                (op1_5, v_tmp1_A, prev_val_A, vc1_5), (op3_5, v_tmp2_A, prev_val_A, vc2_5),
                (op1_5, v_tmp1_B, prev_val_B, vc1_5), (op3_5, v_tmp2_B, prev_val_B, vc2_5),
            ],
        })
        # Load batch 0 elements 4-5 while finishing hash 5
        self.add_bundle({
            "load": [("load", nodes_14[0] + 4, addr_A[4]), ("load", nodes_14[0] + 5, addr_A[5])],
            "valu": [(op2_5, prev_val_A, v_tmp1_A, v_tmp2_A), (op2_5, prev_val_B, v_tmp1_B, v_tmp2_B)],
        })
        # Load batch 0 elements 6-7 while doing idx computation
        self.add_bundle({
            "load": [("load", nodes_14[0] + 6, addr_A[6]), ("load", nodes_14[0] + 7, addr_A[7])],
            "valu": [
                ("&", v_tmp1_A, prev_val_A, v_one), ("<<", prev_idx_A, prev_idx_A, v_one),
                ("&", v_tmp1_B, prev_val_B, v_one), ("<<", prev_idx_B, prev_idx_B, v_one),
            ],
        })
        # Compute addresses for batch 1 while continuing idx
        preload_batch_1_idx = batch_info_14[1][0]  # v_idx[1]
        self.add_bundle({
            "alu": [("+", addr_A[i], self.scratch["forest_values_p"], preload_batch_1_idx + i) for i in range(VLEN)],
            "valu": [("+", v_tmp1_A, v_tmp1_A, v_one), ("+", v_tmp1_B, v_tmp1_B, v_one)],
        })
        # Finish idx + load batch 1 elements 0-1
        self.add_bundle({
            "load": [("load", nodes_14[1] + 0, addr_A[0]), ("load", nodes_14[1] + 1, addr_A[1])],
            "valu": [("+", prev_idx_A, prev_idx_A, v_tmp1_A), ("+", prev_idx_B, prev_idx_B, v_tmp1_B)],
        })

        # Rounds 14-15: Use cross-group pre-loading like main loop
        for _round in range(14, 16):
            if _round == 14:
                # GROUP 0: Already pre-loaded batch 0 above, start from batch 1
                base = 0
                batch_info = [(v_idx[base + i], v_val[base + i]) for i in range(4)]
                nodes = nodes_14  # Already set
            else:
                # GROUP 0: Use pre-loaded nodes from Round 14's Group 7 (node_set_C)
                base = 0
                batch_info = [(v_idx[base + i], v_val[base + i]) for i in range(4)]
                nodes = node_set_C  # Pre-loaded during Round 14's Group 7

            # Phase 2: Gather batch 1 + XOR batch 0 + start hash 0
            self.add_bundle({"alu": [("+", addr_A[i], self.scratch["forest_values_p"], batch_info[1][0] + i) for i in range(VLEN)]})
            self.add_bundle({
                "load": [("load", nodes[1] + 0, addr_A[0]), ("load", nodes[1] + 1, addr_A[1])],
                "valu": [("^", batch_info[0][1], batch_info[0][1], nodes[0])],
            })
            self.add_bundle({
                "load": [("load", nodes[1] + 2, addr_A[2]), ("load", nodes[1] + 3, addr_A[3])],
                "valu": [(op1_0, tmp_list[0][0], batch_info[0][1], vc1_0), (op3_0, tmp_list[0][1], batch_info[0][1], vc2_0)],
            })
            self.add_bundle({
                "load": [("load", nodes[1] + 4, addr_A[4]), ("load", nodes[1] + 5, addr_A[5])],
                "valu": [(op2_0, batch_info[0][1], tmp_list[0][0], tmp_list[0][1])],
            })
            self.add_bundle({
                "load": [("load", nodes[1] + 6, addr_A[6]), ("load", nodes[1] + 7, addr_A[7])],
                "valu": [(op1_1, tmp_list[0][0], batch_info[0][1], vc1_1), (op3_1, tmp_list[0][1], batch_info[0][1], vc2_1)],
            })

            # Phase 3: Gather batch 2 + continue hash batches 0,1
            self.add_bundle({"alu": [("+", addr_A[i], self.scratch["forest_values_p"], batch_info[2][0] + i) for i in range(VLEN)]})
            self.add_bundle({
                "load": [("load", nodes[2] + 0, addr_A[0]), ("load", nodes[2] + 1, addr_A[1])],
                "valu": [(op2_1, batch_info[0][1], tmp_list[0][0], tmp_list[0][1]), ("^", batch_info[1][1], batch_info[1][1], nodes[1])],
            })
            self.add_bundle({
                "load": [("load", nodes[2] + 2, addr_A[2]), ("load", nodes[2] + 3, addr_A[3])],
                "valu": [
                    (op1_2, tmp_list[0][0], batch_info[0][1], vc1_2), (op3_2, tmp_list[0][1], batch_info[0][1], vc2_2),
                    (op1_0, tmp_list[1][0], batch_info[1][1], vc1_0), (op3_0, tmp_list[1][1], batch_info[1][1], vc2_0),
                ],
            })
            self.add_bundle({
                "load": [("load", nodes[2] + 4, addr_A[4]), ("load", nodes[2] + 5, addr_A[5])],
                "valu": [(op2_2, batch_info[0][1], tmp_list[0][0], tmp_list[0][1]), (op2_0, batch_info[1][1], tmp_list[1][0], tmp_list[1][1])],
            })
            self.add_bundle({
                "load": [("load", nodes[2] + 6, addr_A[6]), ("load", nodes[2] + 7, addr_A[7])],
                "valu": [
                    (op1_3, tmp_list[0][0], batch_info[0][1], vc1_3), (op3_3, tmp_list[0][1], batch_info[0][1], vc2_3),
                    (op1_1, tmp_list[1][0], batch_info[1][1], vc1_1), (op3_1, tmp_list[1][1], batch_info[1][1], vc2_1),
                ],
            })

            # Phase 4: Gather batch 3 + continue hash batches 0,1,2
            self.add_bundle({"alu": [("+", addr_A[i], self.scratch["forest_values_p"], batch_info[3][0] + i) for i in range(VLEN)]})
            self.add_bundle({
                "load": [("load", nodes[3] + 0, addr_A[0]), ("load", nodes[3] + 1, addr_A[1])],
                "valu": [
                    (op2_3, batch_info[0][1], tmp_list[0][0], tmp_list[0][1]),
                    (op2_1, batch_info[1][1], tmp_list[1][0], tmp_list[1][1]),
                    ("^", batch_info[2][1], batch_info[2][1], nodes[2]),
                ],
            })
            self.add_bundle({
                "load": [("load", nodes[3] + 2, addr_A[2]), ("load", nodes[3] + 3, addr_A[3])],
                "valu": [
                    (op1_4, tmp_list[0][0], batch_info[0][1], vc1_4), (op3_4, tmp_list[0][1], batch_info[0][1], vc2_4),
                    (op1_2, tmp_list[1][0], batch_info[1][1], vc1_2), (op3_2, tmp_list[1][1], batch_info[1][1], vc2_2),
                ],
            })
            self.add_bundle({
                "load": [("load", nodes[3] + 4, addr_A[4]), ("load", nodes[3] + 5, addr_A[5])],
                "valu": [
                    (op2_4, batch_info[0][1], tmp_list[0][0], tmp_list[0][1]),
                    (op2_2, batch_info[1][1], tmp_list[1][0], tmp_list[1][1]),
                    (op1_0, tmp_list[2][0], batch_info[2][1], vc1_0), (op3_0, tmp_list[2][1], batch_info[2][1], vc2_0),
                ],
            })
            self.add_bundle({
                "load": [("load", nodes[3] + 6, addr_A[6]), ("load", nodes[3] + 7, addr_A[7])],
                "valu": [
                    (op1_5, tmp_list[0][0], batch_info[0][1], vc1_5), (op3_5, tmp_list[0][1], batch_info[0][1], vc2_5),
                    (op1_3, tmp_list[1][0], batch_info[1][1], vc1_3), (op3_3, tmp_list[1][1], batch_info[1][1], vc2_3),
                ],
            })

            # Phase 5: Finish hash/idx + PRE-LOAD group 1's nodes into node_set_B
            next_base = 4
            next_batch_info = [(v_idx[next_base + i], v_val[next_base + i]) for i in range(4)]
            next_nodes = node_set_B

            self.add_bundle({
                "alu": [("+", addr_A[i], self.scratch["forest_values_p"], next_batch_info[0][0] + i) for i in range(VLEN)],
                "valu": [
                    (op2_5, batch_info[0][1], tmp_list[0][0], tmp_list[0][1]),
                    (op2_3, batch_info[1][1], tmp_list[1][0], tmp_list[1][1]),
                    (op2_0, batch_info[2][1], tmp_list[2][0], tmp_list[2][1]),
                    ("^", batch_info[3][1], batch_info[3][1], nodes[3]),
                ],
            })
            self.add_bundle({
                "load": [("load", next_nodes[0] + 0, addr_A[0]), ("load", next_nodes[0] + 1, addr_A[1])],
                "valu": [
                    ("&", tmp_list[0][0], batch_info[0][1], v_one), ("<<", batch_info[0][0], batch_info[0][0], v_one),
                    (op1_4, tmp_list[1][0], batch_info[1][1], vc1_4), (op3_4, tmp_list[1][1], batch_info[1][1], vc2_4),
                ],
            })
            self.add_bundle({
                "load": [("load", next_nodes[0] + 2, addr_A[2]), ("load", next_nodes[0] + 3, addr_A[3])],
                "valu": [
                    ("+", tmp_list[0][0], tmp_list[0][0], v_one),
                    (op2_4, batch_info[1][1], tmp_list[1][0], tmp_list[1][1]),
                    (op1_1, tmp_list[2][0], batch_info[2][1], vc1_1), (op3_1, tmp_list[2][1], batch_info[2][1], vc2_1),
                ],
            })
            self.add_bundle({
                "load": [("load", next_nodes[0] + 4, addr_A[4]), ("load", next_nodes[0] + 5, addr_A[5])],
                "valu": [
                    ("+", batch_info[0][0], batch_info[0][0], tmp_list[0][0]),
                    (op1_5, tmp_list[1][0], batch_info[1][1], vc1_5), (op3_5, tmp_list[1][1], batch_info[1][1], vc2_5),
                    (op1_0, tmp_list[3][0], batch_info[3][1], vc1_0), (op3_0, tmp_list[3][1], batch_info[3][1], vc2_0),
                ],
            })
            self.add_bundle({
                "load": [("load", next_nodes[0] + 6, addr_A[6]), ("load", next_nodes[0] + 7, addr_A[7])],
                "valu": [
                    (op2_5, batch_info[1][1], tmp_list[1][0], tmp_list[1][1]),
                    (op2_1, batch_info[2][1], tmp_list[2][0], tmp_list[2][1]),
                    (op2_0, batch_info[3][1], tmp_list[3][0], tmp_list[3][1]),
                ],
            })
            self.add_bundle({
                "alu": [("+", addr_A[i], self.scratch["forest_values_p"], next_batch_info[1][0] + i) for i in range(VLEN)],
                "valu": [
                    ("&", tmp_list[1][0], batch_info[1][1], v_one), ("<<", batch_info[1][0], batch_info[1][0], v_one),
                    (op1_2, tmp_list[2][0], batch_info[2][1], vc1_2), (op3_2, tmp_list[2][1], batch_info[2][1], vc2_2),
                ],
            })
            self.add_bundle({
                "load": [("load", next_nodes[1] + 0, addr_A[0]), ("load", next_nodes[1] + 1, addr_A[1])],
                "valu": [
                    ("+", tmp_list[1][0], tmp_list[1][0], v_one),
                    (op2_2, batch_info[2][1], tmp_list[2][0], tmp_list[2][1]),
                    (op1_1, tmp_list[3][0], batch_info[3][1], vc1_1), (op3_1, tmp_list[3][1], batch_info[3][1], vc2_1),
                ],
            })
            self.add_bundle({
                "load": [("load", next_nodes[1] + 2, addr_A[2]), ("load", next_nodes[1] + 3, addr_A[3])],
                "valu": [
                    ("+", batch_info[1][0], batch_info[1][0], tmp_list[1][0]),
                    (op1_3, tmp_list[2][0], batch_info[2][1], vc1_3), (op3_3, tmp_list[2][1], batch_info[2][1], vc2_3),
                    (op2_1, batch_info[3][1], tmp_list[3][0], tmp_list[3][1]),
                ],
            })
            self.add_bundle({
                "load": [("load", next_nodes[1] + 4, addr_A[4]), ("load", next_nodes[1] + 5, addr_A[5])],
                "valu": [
                    (op2_3, batch_info[2][1], tmp_list[2][0], tmp_list[2][1]),
                    (op1_2, tmp_list[3][0], batch_info[3][1], vc1_2), (op3_2, tmp_list[3][1], batch_info[3][1], vc2_2),
                ],
            })
            self.add_bundle({
                "load": [("load", next_nodes[1] + 6, addr_A[6]), ("load", next_nodes[1] + 7, addr_A[7])],
                "valu": [
                    (op1_4, tmp_list[2][0], batch_info[2][1], vc1_4), (op3_4, tmp_list[2][1], batch_info[2][1], vc2_4),
                    (op2_2, batch_info[3][1], tmp_list[3][0], tmp_list[3][1]),
                ],
            })
            self.add_bundle({
                "alu": [("+", addr_A[i], self.scratch["forest_values_p"], next_batch_info[2][0] + i) for i in range(VLEN)],
                "valu": [
                    (op2_4, batch_info[2][1], tmp_list[2][0], tmp_list[2][1]),
                    (op1_3, tmp_list[3][0], batch_info[3][1], vc1_3), (op3_3, tmp_list[3][1], batch_info[3][1], vc2_3),
                ],
            })
            self.add_bundle({
                "load": [("load", next_nodes[2] + 0, addr_A[0]), ("load", next_nodes[2] + 1, addr_A[1])],
                "valu": [
                    (op1_5, tmp_list[2][0], batch_info[2][1], vc1_5), (op3_5, tmp_list[2][1], batch_info[2][1], vc2_5),
                    (op2_3, batch_info[3][1], tmp_list[3][0], tmp_list[3][1]),
                ],
            })
            self.add_bundle({
                "load": [("load", next_nodes[2] + 2, addr_A[2]), ("load", next_nodes[2] + 3, addr_A[3])],
                "valu": [
                    (op2_5, batch_info[2][1], tmp_list[2][0], tmp_list[2][1]),
                    (op1_4, tmp_list[3][0], batch_info[3][1], vc1_4), (op3_4, tmp_list[3][1], batch_info[3][1], vc2_4),
                ],
            })
            self.add_bundle({
                "load": [("load", next_nodes[2] + 4, addr_A[4]), ("load", next_nodes[2] + 5, addr_A[5])],
                "valu": [
                    ("&", tmp_list[2][0], batch_info[2][1], v_one), ("<<", batch_info[2][0], batch_info[2][0], v_one),
                    (op2_4, batch_info[3][1], tmp_list[3][0], tmp_list[3][1]),
                ],
            })
            self.add_bundle({
                "load": [("load", next_nodes[2] + 6, addr_A[6]), ("load", next_nodes[2] + 7, addr_A[7])],
                "valu": [
                    ("+", tmp_list[2][0], tmp_list[2][0], v_one),
                    (op1_5, tmp_list[3][0], batch_info[3][1], vc1_5), (op3_5, tmp_list[3][1], batch_info[3][1], vc2_5),
                ],
            })
            self.add_bundle({
                "alu": [("+", addr_A[i], self.scratch["forest_values_p"], next_batch_info[3][0] + i) for i in range(VLEN)],
                "valu": [
                    ("+", batch_info[2][0], batch_info[2][0], tmp_list[2][0]),
                    (op2_5, batch_info[3][1], tmp_list[3][0], tmp_list[3][1]),
                ],
            })
            self.add_bundle({
                "load": [("load", next_nodes[3] + 0, addr_A[0]), ("load", next_nodes[3] + 1, addr_A[1])],
                "valu": [("&", tmp_list[3][0], batch_info[3][1], v_one), ("<<", batch_info[3][0], batch_info[3][0], v_one)],
            })
            self.add_bundle({
                "load": [("load", next_nodes[3] + 2, addr_A[2]), ("load", next_nodes[3] + 3, addr_A[3])],
                "valu": [("+", tmp_list[3][0], tmp_list[3][0], v_one)],
            })
            self.add_bundle({
                "load": [("load", next_nodes[3] + 4, addr_A[4]), ("load", next_nodes[3] + 5, addr_A[5])],
                "valu": [("+", batch_info[3][0], batch_info[3][0], tmp_list[3][0])],
            })
            self.add_bundle({"load": [("load", next_nodes[3] + 6, addr_A[6]), ("load", next_nodes[3] + 7, addr_A[7])]})

            # ===== GROUPS 1-6: Use pre-loaded nodes + pre-load next group =====
            for group in range(1, 7):
                base = group * 4
                batch_info = [(v_idx[base + i], v_val[base + i]) for i in range(4)]
                nodes = node_set_B if group % 2 == 1 else node_set_A
                next_nodes = node_set_A if group % 2 == 1 else node_set_B

                next_base = (group + 1) * 4
                next_batch_info = [(v_idx[next_base + i], v_val[next_base + i]) for i in range(4)]

                # Compute addresses for next batches 0,1 + XOR current batches
                self.add_bundle({
                    "alu": [("+", addr_A[i], self.scratch["forest_values_p"], next_batch_info[0][0] + i) for i in range(VLEN)],
                    "valu": [
                        ("^", batch_info[0][1], batch_info[0][1], nodes[0]),
                        ("^", batch_info[1][1], batch_info[1][1], nodes[1]),
                        ("^", batch_info[2][1], batch_info[2][1], nodes[2]),
                        ("^", batch_info[3][1], batch_info[3][1], nodes[3]),
                    ],
                })
                self.add_bundle({
                    "alu": [("+", addr_B[i], self.scratch["forest_values_p"], next_batch_info[1][0] + i) for i in range(VLEN)],
                    "valu": [
                        (op1_0, tmp_list[0][0], batch_info[0][1], vc1_0), (op3_0, tmp_list[0][1], batch_info[0][1], vc2_0),
                        (op1_0, tmp_list[1][0], batch_info[1][1], vc1_0), (op3_0, tmp_list[1][1], batch_info[1][1], vc2_0),
                    ],
                })
                # Load batches 0,1 interleaved
                self.add_bundle({
                    "load": [("load", next_nodes[0] + 0, addr_A[0]), ("load", next_nodes[1] + 0, addr_B[0])],
                    "valu": [
                        (op2_0, batch_info[0][1], tmp_list[0][0], tmp_list[0][1]),
                        (op2_0, batch_info[1][1], tmp_list[1][0], tmp_list[1][1]),
                        (op1_0, tmp_list[2][0], batch_info[2][1], vc1_0), (op3_0, tmp_list[2][1], batch_info[2][1], vc2_0),
                    ],
                })
                self.add_bundle({
                    "load": [("load", next_nodes[0] + 1, addr_A[1]), ("load", next_nodes[1] + 1, addr_B[1])],
                    "valu": [
                        (op1_1, tmp_list[0][0], batch_info[0][1], vc1_1), (op3_1, tmp_list[0][1], batch_info[0][1], vc2_1),
                        (op1_1, tmp_list[1][0], batch_info[1][1], vc1_1), (op3_1, tmp_list[1][1], batch_info[1][1], vc2_1),
                    ],
                })
                self.add_bundle({
                    "load": [("load", next_nodes[0] + 2, addr_A[2]), ("load", next_nodes[1] + 2, addr_B[2])],
                    "valu": [
                        (op2_1, batch_info[0][1], tmp_list[0][0], tmp_list[0][1]),
                        (op2_1, batch_info[1][1], tmp_list[1][0], tmp_list[1][1]),
                        (op2_0, batch_info[2][1], tmp_list[2][0], tmp_list[2][1]),
                        (op1_0, tmp_list[3][0], batch_info[3][1], vc1_0), (op3_0, tmp_list[3][1], batch_info[3][1], vc2_0),
                    ],
                })
                self.add_bundle({
                    "load": [("load", next_nodes[0] + 3, addr_A[3]), ("load", next_nodes[1] + 3, addr_B[3])],
                    "valu": [
                        (op1_2, tmp_list[0][0], batch_info[0][1], vc1_2), (op3_2, tmp_list[0][1], batch_info[0][1], vc2_2),
                        (op1_2, tmp_list[1][0], batch_info[1][1], vc1_2), (op3_2, tmp_list[1][1], batch_info[1][1], vc2_2),
                    ],
                })
                self.add_bundle({
                    "load": [("load", next_nodes[0] + 4, addr_A[4]), ("load", next_nodes[1] + 4, addr_B[4])],
                    "valu": [
                        (op2_2, batch_info[0][1], tmp_list[0][0], tmp_list[0][1]),
                        (op2_2, batch_info[1][1], tmp_list[1][0], tmp_list[1][1]),
                        (op1_1, tmp_list[2][0], batch_info[2][1], vc1_1), (op3_1, tmp_list[2][1], batch_info[2][1], vc2_1),
                    ],
                })
                self.add_bundle({
                    "load": [("load", next_nodes[0] + 5, addr_A[5]), ("load", next_nodes[1] + 5, addr_B[5])],
                    "valu": [
                        (op1_3, tmp_list[0][0], batch_info[0][1], vc1_3), (op3_3, tmp_list[0][1], batch_info[0][1], vc2_3),
                        (op1_3, tmp_list[1][0], batch_info[1][1], vc1_3), (op3_3, tmp_list[1][1], batch_info[1][1], vc2_3),
                    ],
                })
                self.add_bundle({
                    "load": [("load", next_nodes[0] + 6, addr_A[6]), ("load", next_nodes[1] + 6, addr_B[6])],
                    "valu": [
                        (op2_3, batch_info[0][1], tmp_list[0][0], tmp_list[0][1]),
                        (op2_3, batch_info[1][1], tmp_list[1][0], tmp_list[1][1]),
                        (op2_1, batch_info[2][1], tmp_list[2][0], tmp_list[2][1]),
                        (op2_0, batch_info[3][1], tmp_list[3][0], tmp_list[3][1]),
                    ],
                })
                self.add_bundle({
                    "load": [("load", next_nodes[0] + 7, addr_A[7]), ("load", next_nodes[1] + 7, addr_B[7])],
                    "valu": [
                        (op1_4, tmp_list[0][0], batch_info[0][1], vc1_4), (op3_4, tmp_list[0][1], batch_info[0][1], vc2_4),
                        (op1_4, tmp_list[1][0], batch_info[1][1], vc1_4), (op3_4, tmp_list[1][1], batch_info[1][1], vc2_4),
                    ],
                })
                # Compute addresses for next batches 2,3
                self.add_bundle({
                    "alu": [("+", addr_A[i], self.scratch["forest_values_p"], next_batch_info[2][0] + i) for i in range(VLEN)],
                    "valu": [
                        (op2_4, batch_info[0][1], tmp_list[0][0], tmp_list[0][1]),
                        (op2_4, batch_info[1][1], tmp_list[1][0], tmp_list[1][1]),
                        (op1_2, tmp_list[2][0], batch_info[2][1], vc1_2), (op3_2, tmp_list[2][1], batch_info[2][1], vc2_2),
                    ],
                })
                self.add_bundle({
                    "alu": [("+", addr_B[i], self.scratch["forest_values_p"], next_batch_info[3][0] + i) for i in range(VLEN)],
                    "valu": [
                        (op1_5, tmp_list[0][0], batch_info[0][1], vc1_5), (op3_5, tmp_list[0][1], batch_info[0][1], vc2_5),
                        (op1_5, tmp_list[1][0], batch_info[1][1], vc1_5), (op3_5, tmp_list[1][1], batch_info[1][1], vc2_5),
                    ],
                })
                # Load batches 2,3 interleaved
                self.add_bundle({
                    "load": [("load", next_nodes[2] + 0, addr_A[0]), ("load", next_nodes[3] + 0, addr_B[0])],
                    "valu": [
                        (op2_5, batch_info[0][1], tmp_list[0][0], tmp_list[0][1]),
                        (op2_5, batch_info[1][1], tmp_list[1][0], tmp_list[1][1]),
                        (op2_2, batch_info[2][1], tmp_list[2][0], tmp_list[2][1]),
                        (op1_1, tmp_list[3][0], batch_info[3][1], vc1_1), (op3_1, tmp_list[3][1], batch_info[3][1], vc2_1),
                    ],
                })
                self.add_bundle({
                    "load": [("load", next_nodes[2] + 1, addr_A[1]), ("load", next_nodes[3] + 1, addr_B[1])],
                    "valu": [
                        ("&", tmp_list[0][0], batch_info[0][1], v_one), ("<<", batch_info[0][0], batch_info[0][0], v_one),
                        ("&", tmp_list[1][0], batch_info[1][1], v_one), ("<<", batch_info[1][0], batch_info[1][0], v_one),
                    ],
                })
                self.add_bundle({
                    "load": [("load", next_nodes[2] + 2, addr_A[2]), ("load", next_nodes[3] + 2, addr_B[2])],
                    "valu": [
                        ("+", tmp_list[0][0], tmp_list[0][0], v_one),
                        ("+", tmp_list[1][0], tmp_list[1][0], v_one),
                        (op1_3, tmp_list[2][0], batch_info[2][1], vc1_3), (op3_3, tmp_list[2][1], batch_info[2][1], vc2_3),
                    ],
                })
                self.add_bundle({
                    "load": [("load", next_nodes[2] + 3, addr_A[3]), ("load", next_nodes[3] + 3, addr_B[3])],
                    "valu": [
                        ("+", batch_info[0][0], batch_info[0][0], tmp_list[0][0]),
                        ("+", batch_info[1][0], batch_info[1][0], tmp_list[1][0]),
                        (op2_3, batch_info[2][1], tmp_list[2][0], tmp_list[2][1]),
                        (op2_1, batch_info[3][1], tmp_list[3][0], tmp_list[3][1]),
                    ],
                })
                self.add_bundle({
                    "load": [("load", next_nodes[2] + 4, addr_A[4]), ("load", next_nodes[3] + 4, addr_B[4])],
                    "valu": [
                        (op1_4, tmp_list[2][0], batch_info[2][1], vc1_4), (op3_4, tmp_list[2][1], batch_info[2][1], vc2_4),
                        (op1_2, tmp_list[3][0], batch_info[3][1], vc1_2), (op3_2, tmp_list[3][1], batch_info[3][1], vc2_2),
                    ],
                })
                self.add_bundle({
                    "load": [("load", next_nodes[2] + 5, addr_A[5]), ("load", next_nodes[3] + 5, addr_B[5])],
                    "valu": [
                        (op2_4, batch_info[2][1], tmp_list[2][0], tmp_list[2][1]),
                        (op2_2, batch_info[3][1], tmp_list[3][0], tmp_list[3][1]),
                    ],
                })
                self.add_bundle({
                    "load": [("load", next_nodes[2] + 6, addr_A[6]), ("load", next_nodes[3] + 6, addr_B[6])],
                    "valu": [
                        (op1_5, tmp_list[2][0], batch_info[2][1], vc1_5), (op3_5, tmp_list[2][1], batch_info[2][1], vc2_5),
                        (op1_3, tmp_list[3][0], batch_info[3][1], vc1_3), (op3_3, tmp_list[3][1], batch_info[3][1], vc2_3),
                    ],
                })
                self.add_bundle({
                    "load": [("load", next_nodes[2] + 7, addr_A[7]), ("load", next_nodes[3] + 7, addr_B[7])],
                    "valu": [
                        (op2_5, batch_info[2][1], tmp_list[2][0], tmp_list[2][1]),
                        (op2_3, batch_info[3][1], tmp_list[3][0], tmp_list[3][1]),
                    ],
                })
                # Finish idx for batches 2,3 (OPTIMIZED: multiply_add combines <<1 and +1)
                self.add_bundle({"valu": [
                    ("multiply_add", batch_info[2][0], batch_info[2][0], v_two, v_one),  # idx*2+1
                    ("&", tmp_list[2][0], batch_info[2][1], v_one),  # tmp = val&1
                    (op1_4, tmp_list[3][0], batch_info[3][1], vc1_4), (op3_4, tmp_list[3][1], batch_info[3][1], vc2_4),
                ]})
                self.add_bundle({"valu": [
                    ("+", batch_info[2][0], batch_info[2][0], tmp_list[2][0]),  # idx += tmp (batch 2 done)
                    (op2_4, batch_info[3][1], tmp_list[3][0], tmp_list[3][1]),
                ]})
                self.add_bundle({"valu": [
                    (op1_5, tmp_list[3][0], batch_info[3][1], vc1_5), (op3_5, tmp_list[3][1], batch_info[3][1], vc2_5),
                ]})
                self.add_bundle({"valu": [(op2_5, batch_info[3][1], tmp_list[3][0], tmp_list[3][1])]})
                self.add_bundle({"valu": [
                    ("multiply_add", batch_info[3][0], batch_info[3][0], v_two, v_one),  # idx*2+1
                    ("&", tmp_list[3][0], batch_info[3][1], v_one),  # tmp = val&1
                ]})
                self.add_bundle({"valu": [("+", batch_info[3][0], batch_info[3][0], tmp_list[3][0])]})

            # ===== GROUP 7: Use pre-loaded nodes =====
            # OPTIMIZED: Use multiply_add for hash stages 0, 2, 4
            # When _round == 14, pre-load Round 15's Group 0 nodes into node_set_C
            base = 7 * 4
            batch_info = [(v_idx[base + i], v_val[base + i]) for i in range(4)]
            nodes = node_set_B

            # For Round 14, set up pre-loading for Round 15's Group 0
            # v_idx[0..3] already contain the indices Round 15 will use (computed in Group 0)
            preload_nodes = node_set_C
            preload_batch_idx = v_idx[0]  # Round 15's Group 0 batch 0

            if _round == 14:
                # Compute addresses for Round 15's Group 0 batch 0 + XOR all 4 batches
                self.add_bundle({
                    "alu": [("+", addr_A[i], self.scratch["forest_values_p"], preload_batch_idx + i) for i in range(VLEN)],
                    "valu": [
                        ("^", batch_info[0][1], batch_info[0][1], nodes[0]),
                        ("^", batch_info[1][1], batch_info[1][1], nodes[1]),
                        ("^", batch_info[2][1], batch_info[2][1], nodes[2]),
                        ("^", batch_info[3][1], batch_info[3][1], nodes[3]),
                    ],
                })

                # Hash stage 0 with multiply_add + load elements 0-1
                self.add_bundle({
                    "load": [("load", preload_nodes[0] + 0, addr_A[0]), ("load", preload_nodes[0] + 1, addr_A[1])],
                    "valu": [
                        ("multiply_add", batch_info[0][1], batch_info[0][1], v_mult_4097, vc1_0),
                        ("multiply_add", batch_info[1][1], batch_info[1][1], v_mult_4097, vc1_0),
                        ("multiply_add", batch_info[2][1], batch_info[2][1], v_mult_4097, vc1_0),
                        ("multiply_add", batch_info[3][1], batch_info[3][1], v_mult_4097, vc1_0),
                    ],
                })

                # Hash stage 1 + load elements 2-7
                self.add_bundle({
                    "load": [("load", preload_nodes[0] + 2, addr_A[2]), ("load", preload_nodes[0] + 3, addr_A[3])],
                    "valu": [
                        (op1_1, tmp_list[0][0], batch_info[0][1], vc1_1), (op3_1, tmp_list[0][1], batch_info[0][1], vc2_1),
                        (op1_1, tmp_list[1][0], batch_info[1][1], vc1_1), (op3_1, tmp_list[1][1], batch_info[1][1], vc2_1),
                    ],
                })
                self.add_bundle({
                    "load": [("load", preload_nodes[0] + 4, addr_A[4]), ("load", preload_nodes[0] + 5, addr_A[5])],
                    "valu": [
                        (op2_1, batch_info[0][1], tmp_list[0][0], tmp_list[0][1]),
                        (op2_1, batch_info[1][1], tmp_list[1][0], tmp_list[1][1]),
                        (op1_1, tmp_list[2][0], batch_info[2][1], vc1_1), (op3_1, tmp_list[2][1], batch_info[2][1], vc2_1),
                    ],
                })
                self.add_bundle({
                    "load": [("load", preload_nodes[0] + 6, addr_A[6]), ("load", preload_nodes[0] + 7, addr_A[7])],
                    "valu": [
                        (op2_1, batch_info[2][1], tmp_list[2][0], tmp_list[2][1]),
                        (op1_1, tmp_list[3][0], batch_info[3][1], vc1_1), (op3_1, tmp_list[3][1], batch_info[3][1], vc2_1),
                    ],
                })
                self.add_bundle({"valu": [
                    (op2_1, batch_info[3][1], tmp_list[3][0], tmp_list[3][1]),
                ]})

                # Hash stage 2 with multiply_add
                self.add_bundle({"valu": [
                    ("multiply_add", batch_info[0][1], batch_info[0][1], v_mult_33, vc1_2),
                    ("multiply_add", batch_info[1][1], batch_info[1][1], v_mult_33, vc1_2),
                    ("multiply_add", batch_info[2][1], batch_info[2][1], v_mult_33, vc1_2),
                    ("multiply_add", batch_info[3][1], batch_info[3][1], v_mult_33, vc1_2),
                ]})

                # Hash stage 3
                self.add_bundle({"valu": [
                    (op1_3, tmp_list[0][0], batch_info[0][1], vc1_3), (op3_3, tmp_list[0][1], batch_info[0][1], vc2_3),
                    (op1_3, tmp_list[1][0], batch_info[1][1], vc1_3), (op3_3, tmp_list[1][1], batch_info[1][1], vc2_3),
                ]})
                self.add_bundle({"valu": [
                    (op2_3, batch_info[0][1], tmp_list[0][0], tmp_list[0][1]),
                    (op2_3, batch_info[1][1], tmp_list[1][0], tmp_list[1][1]),
                    (op1_3, tmp_list[2][0], batch_info[2][1], vc1_3), (op3_3, tmp_list[2][1], batch_info[2][1], vc2_3),
                ]})
                self.add_bundle({"valu": [
                    (op2_3, batch_info[2][1], tmp_list[2][0], tmp_list[2][1]),
                    (op1_3, tmp_list[3][0], batch_info[3][1], vc1_3), (op3_3, tmp_list[3][1], batch_info[3][1], vc2_3),
                ]})
                self.add_bundle({"valu": [
                    (op2_3, batch_info[3][1], tmp_list[3][0], tmp_list[3][1]),
                ]})

                # Hash stage 4 with multiply_add
                self.add_bundle({"valu": [
                    ("multiply_add", batch_info[0][1], batch_info[0][1], v_mult_9, vc1_4),
                    ("multiply_add", batch_info[1][1], batch_info[1][1], v_mult_9, vc1_4),
                    ("multiply_add", batch_info[2][1], batch_info[2][1], v_mult_9, vc1_4),
                    ("multiply_add", batch_info[3][1], batch_info[3][1], v_mult_9, vc1_4),
                ]})

                # Hash stage 5
                self.add_bundle({"valu": [
                    (op1_5, tmp_list[0][0], batch_info[0][1], vc1_5), (op3_5, tmp_list[0][1], batch_info[0][1], vc2_5),
                    (op1_5, tmp_list[1][0], batch_info[1][1], vc1_5), (op3_5, tmp_list[1][1], batch_info[1][1], vc2_5),
                ]})
                self.add_bundle({"valu": [
                    (op2_5, batch_info[0][1], tmp_list[0][0], tmp_list[0][1]),
                    (op2_5, batch_info[1][1], tmp_list[1][0], tmp_list[1][1]),
                    (op1_5, tmp_list[2][0], batch_info[2][1], vc1_5), (op3_5, tmp_list[2][1], batch_info[2][1], vc2_5),
                ]})
                self.add_bundle({"valu": [
                    (op2_5, batch_info[2][1], tmp_list[2][0], tmp_list[2][1]),
                    (op1_5, tmp_list[3][0], batch_info[3][1], vc1_5), (op3_5, tmp_list[3][1], batch_info[3][1], vc2_5),
                ]})
                self.add_bundle({"valu": [
                    (op2_5, batch_info[3][1], tmp_list[3][0], tmp_list[3][1]),
                ]})

                # Compute idx for all 4 batches
                self.add_bundle({"valu": [
                    ("multiply_add", batch_info[0][0], batch_info[0][0], v_two, v_one),
                    ("multiply_add", batch_info[1][0], batch_info[1][0], v_two, v_one),
                    ("multiply_add", batch_info[2][0], batch_info[2][0], v_two, v_one),
                    ("multiply_add", batch_info[3][0], batch_info[3][0], v_two, v_one),
                ]})
                self.add_bundle({"valu": [
                    ("&", tmp_list[0][0], batch_info[0][1], v_one),
                    ("&", tmp_list[1][0], batch_info[1][1], v_one),
                    ("&", tmp_list[2][0], batch_info[2][1], v_one),
                    ("&", tmp_list[3][0], batch_info[3][1], v_one),
                ]})
                self.add_bundle({"valu": [
                    ("+", batch_info[0][0], batch_info[0][0], tmp_list[0][0]),
                    ("+", batch_info[1][0], batch_info[1][0], tmp_list[1][0]),
                    ("+", batch_info[2][0], batch_info[2][0], tmp_list[2][0]),
                    ("+", batch_info[3][0], batch_info[3][0], tmp_list[3][0]),
                ]})
            else:
                # Round 15: No next round to pre-load, just do VALU-only computation
                # XOR all 4 batches
                self.add_bundle({"valu": [
                    ("^", batch_info[0][1], batch_info[0][1], nodes[0]),
                    ("^", batch_info[1][1], batch_info[1][1], nodes[1]),
                    ("^", batch_info[2][1], batch_info[2][1], nodes[2]),
                    ("^", batch_info[3][1], batch_info[3][1], nodes[3]),
                ]})

                # Hash stage 0 with multiply_add
                self.add_bundle({"valu": [
                    ("multiply_add", batch_info[0][1], batch_info[0][1], v_mult_4097, vc1_0),
                    ("multiply_add", batch_info[1][1], batch_info[1][1], v_mult_4097, vc1_0),
                    ("multiply_add", batch_info[2][1], batch_info[2][1], v_mult_4097, vc1_0),
                    ("multiply_add", batch_info[3][1], batch_info[3][1], v_mult_4097, vc1_0),
                ]})

                # Hash stage 1
                self.add_bundle({"valu": [
                    (op1_1, tmp_list[0][0], batch_info[0][1], vc1_1), (op3_1, tmp_list[0][1], batch_info[0][1], vc2_1),
                    (op1_1, tmp_list[1][0], batch_info[1][1], vc1_1), (op3_1, tmp_list[1][1], batch_info[1][1], vc2_1),
                ]})
                self.add_bundle({"valu": [
                    (op2_1, batch_info[0][1], tmp_list[0][0], tmp_list[0][1]),
                    (op2_1, batch_info[1][1], tmp_list[1][0], tmp_list[1][1]),
                    (op1_1, tmp_list[2][0], batch_info[2][1], vc1_1), (op3_1, tmp_list[2][1], batch_info[2][1], vc2_1),
                ]})
                self.add_bundle({"valu": [
                    (op2_1, batch_info[2][1], tmp_list[2][0], tmp_list[2][1]),
                    (op1_1, tmp_list[3][0], batch_info[3][1], vc1_1), (op3_1, tmp_list[3][1], batch_info[3][1], vc2_1),
                ]})
                self.add_bundle({"valu": [
                    (op2_1, batch_info[3][1], tmp_list[3][0], tmp_list[3][1]),
                ]})

                # Hash stage 2 with multiply_add
                self.add_bundle({"valu": [
                    ("multiply_add", batch_info[0][1], batch_info[0][1], v_mult_33, vc1_2),
                    ("multiply_add", batch_info[1][1], batch_info[1][1], v_mult_33, vc1_2),
                    ("multiply_add", batch_info[2][1], batch_info[2][1], v_mult_33, vc1_2),
                    ("multiply_add", batch_info[3][1], batch_info[3][1], v_mult_33, vc1_2),
                ]})

                # Hash stage 3
                self.add_bundle({"valu": [
                    (op1_3, tmp_list[0][0], batch_info[0][1], vc1_3), (op3_3, tmp_list[0][1], batch_info[0][1], vc2_3),
                    (op1_3, tmp_list[1][0], batch_info[1][1], vc1_3), (op3_3, tmp_list[1][1], batch_info[1][1], vc2_3),
                ]})
                self.add_bundle({"valu": [
                    (op2_3, batch_info[0][1], tmp_list[0][0], tmp_list[0][1]),
                    (op2_3, batch_info[1][1], tmp_list[1][0], tmp_list[1][1]),
                    (op1_3, tmp_list[2][0], batch_info[2][1], vc1_3), (op3_3, tmp_list[2][1], batch_info[2][1], vc2_3),
                ]})
                self.add_bundle({"valu": [
                    (op2_3, batch_info[2][1], tmp_list[2][0], tmp_list[2][1]),
                    (op1_3, tmp_list[3][0], batch_info[3][1], vc1_3), (op3_3, tmp_list[3][1], batch_info[3][1], vc2_3),
                ]})
                self.add_bundle({"valu": [
                    (op2_3, batch_info[3][1], tmp_list[3][0], tmp_list[3][1]),
                ]})

                # Hash stage 4 with multiply_add
                self.add_bundle({"valu": [
                    ("multiply_add", batch_info[0][1], batch_info[0][1], v_mult_9, vc1_4),
                    ("multiply_add", batch_info[1][1], batch_info[1][1], v_mult_9, vc1_4),
                    ("multiply_add", batch_info[2][1], batch_info[2][1], v_mult_9, vc1_4),
                    ("multiply_add", batch_info[3][1], batch_info[3][1], v_mult_9, vc1_4),
                ]})

                # Hash stage 5
                self.add_bundle({"valu": [
                    (op1_5, tmp_list[0][0], batch_info[0][1], vc1_5), (op3_5, tmp_list[0][1], batch_info[0][1], vc2_5),
                    (op1_5, tmp_list[1][0], batch_info[1][1], vc1_5), (op3_5, tmp_list[1][1], batch_info[1][1], vc2_5),
                ]})
                self.add_bundle({"valu": [
                    (op2_5, batch_info[0][1], tmp_list[0][0], tmp_list[0][1]),
                    (op2_5, batch_info[1][1], tmp_list[1][0], tmp_list[1][1]),
                    (op1_5, tmp_list[2][0], batch_info[2][1], vc1_5), (op3_5, tmp_list[2][1], batch_info[2][1], vc2_5),
                ]})
                self.add_bundle({"valu": [
                    (op2_5, batch_info[2][1], tmp_list[2][0], tmp_list[2][1]),
                    (op1_5, tmp_list[3][0], batch_info[3][1], vc1_5), (op3_5, tmp_list[3][1], batch_info[3][1], vc2_5),
                ]})
                self.add_bundle({"valu": [
                    (op2_5, batch_info[3][1], tmp_list[3][0], tmp_list[3][1]),
                ]})

                # Compute idx for all 4 batches
                self.add_bundle({"valu": [
                    ("multiply_add", batch_info[0][0], batch_info[0][0], v_two, v_one),
                    ("multiply_add", batch_info[1][0], batch_info[1][0], v_two, v_one),
                    ("multiply_add", batch_info[2][0], batch_info[2][0], v_two, v_one),
                    ("multiply_add", batch_info[3][0], batch_info[3][0], v_two, v_one),
                ]})
                self.add_bundle({"valu": [
                    ("&", tmp_list[0][0], batch_info[0][1], v_one),
                    ("&", tmp_list[1][0], batch_info[1][1], v_one),
                    ("&", tmp_list[2][0], batch_info[2][1], v_one),
                    ("&", tmp_list[3][0], batch_info[3][1], v_one),
                ]})
                self.add_bundle({"valu": [
                    ("+", batch_info[0][0], batch_info[0][0], tmp_list[0][0]),
                    ("+", batch_info[1][0], batch_info[1][0], tmp_list[1][0]),
                    ("+", batch_info[2][0], batch_info[2][0], tmp_list[2][0]),
                    ("+", batch_info[3][0], batch_info[3][0], tmp_list[3][0]),
                ]})

        # Store all indices and values back - use 2 pointers and ALU for parallel increment
        ptr2 = self.alloc_scratch("ptr2")
        vlen_const = self.scratch_const(VLEN)
        self.add_bundle({"alu": [
            ("+", ptr, self.scratch["inp_indices_p"], zero_const),
            ("+", ptr2, self.scratch["inp_values_p"], zero_const),
        ]})

        # Interleave idx and val stores, 2 stores per cycle, increment both ptrs in parallel
        for i in range(num_batches):
            self.add_bundle({"store": [("vstore", ptr, v_idx[i]), ("vstore", ptr2, v_val[i])]})
            if i + 1 < num_batches:
                # Use ALU to increment both pointers in single cycle
                self.add_bundle({"alu": [("+", ptr, ptr, vlen_const), ("+", ptr2, ptr2, vlen_const)]})

        self.instrs.append({"flow": [("pause",)]})

BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    # print(kb.instrs)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])
        # Updating these in memory isn't required, but you can enable this check for debugging
        # assert machine.mem[inp_indices_p:inp_indices_p+len(inp.indices)] == ref_mem[inp_indices_p:inp_indices_p+len(inp.indices)]

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    # Passing this test is not required for submission, see submission_tests.py for the actual correctness test
    # You can uncomment this if you think it might help you debug
    # def test_kernel_correctness(self):
    #     for batch in range(1, 3):
    #         for forest_height in range(3):
    #             do_kernel_test(
    #                 forest_height + 2, forest_height + 4, batch * 16 * VLEN * N_CORES
    #             )

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


# To run all the tests:
#    python perf_takehome.py
# To run a specific test:
#    python perf_takehome.py Tests.test_kernel_cycles
# To view a hot-reloading trace of all the instructions:  **Recommended debug loop**
# NOTE: The trace hot-reloading only works in Chrome. In the worst case if things aren't working, drag trace.json onto https://ui.perfetto.dev/
#    python perf_takehome.py Tests.test_kernel_trace
# Then run `python watch_trace.py` in another tab, it'll open a browser tab, then click "Open Perfetto"
# You can then keep that open and re-run the test to see a new trace.

# To run the proper checks to see which thresholds you pass:
#    python tests/submission_tests.py

if __name__ == "__main__":
    unittest.main()

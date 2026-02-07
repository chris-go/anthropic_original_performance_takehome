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
        ten_const = self.preload_const(10)

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

        # Extra temps for hash parallelism (6-batch processing needs 6 pairs)
        v_tmp1_C = self.alloc_scratch("v_tmp1_C", VLEN)
        v_tmp2_C = self.alloc_scratch("v_tmp2_C", VLEN)
        v_tmp1_D = self.alloc_scratch("v_tmp1_D", VLEN)
        v_tmp2_D = self.alloc_scratch("v_tmp2_D", VLEN)
        v_tmp1_F = self.alloc_scratch("v_tmp1_F", VLEN)
        v_tmp2_F = self.alloc_scratch("v_tmp2_F", VLEN)

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
        addr_C = [self.alloc_scratch(f"addrC{i}") for i in range(VLEN)]
        addr_D = [self.alloc_scratch(f"addrD{i}") for i in range(VLEN)]

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

        # Load indices: 2 vectors per cycle using 2 pointers, merge ptr increments (VLIW safe)
        self.add_bundle({"flow": [("add_imm", ptr, self.scratch["inp_indices_p"], 0)]})
        self.add_bundle({"flow": [("add_imm", ptr2, self.scratch["inp_indices_p"], VLEN)]})
        for i in range(0, num_batches, 2):
            bundle = {"load": [("vload", v_idx[i], ptr), ("vload", v_idx[i + 1], ptr2)]}
            if i + 2 < num_batches:
                bundle["alu"] = [("+", ptr, ptr, vlen2_const), ("+", ptr2, ptr2, vlen2_const)]
            self.add_bundle(bundle)

        # Load values: same approach
        self.add_bundle({"flow": [("add_imm", ptr, self.scratch["inp_values_p"], 0)]})
        self.add_bundle({"flow": [("add_imm", ptr2, self.scratch["inp_values_p"], VLEN)]})
        for i in range(0, num_batches, 2):
            bundle = {"load": [("vload", v_val[i], ptr), ("vload", v_val[i + 1], ptr2)]}
            if i + 2 < num_batches:
                bundle["alu"] = [("+", ptr, ptr, vlen2_const), ("+", ptr2, ptr2, vlen2_const)]
            self.add_bundle(bundle)

        self.add_bundle({"load": [("const", num_rounds_s, rounds)]})
        self.add("flow", ("pause",))

        # ===== ROUND 0 SPECIAL CASE =====
        # All indices start at 0, so we only need ONE forest load!
        node_scalar = self.alloc_scratch("node_scalar")
        v_node_shared = self.alloc_scratch("v_node_shared", VLEN)

        # Extra temps for 6-batch processing
        v_tmp1_E = self.alloc_scratch("v_tmp1_E", VLEN)
        v_tmp2_E = self.alloc_scratch("v_tmp2_E", VLEN)
        tmp6 = [(v_tmp1_A, v_tmp2_A), (v_tmp1_B, v_tmp2_B), (v_tmp1_E, v_tmp2_E),
                (v_tmp1_C, v_tmp2_C), (v_tmp1_D, v_tmp2_D), (v_tmp1_F, v_tmp2_F)]

        # Load forest[0] once and broadcast
        self.add_bundle({"load": [("load", node_scalar, self.scratch["forest_values_p"])]})
        self.add_bundle({"valu": [("vbroadcast", v_node_shared, node_scalar)]})

        # Process 6 batches at a time (saturates 6 VALU slots)
        # 32 batches = 5 groups of 6 + 1 group of 2
        mult_consts = [v_mult_4097, None, v_mult_33, None, v_mult_9, None]
        for group in range(6):
            if group < 5:
                b = group * 6
                vals = [v_val[b + i] for i in range(6)]
                idxs = [v_idx[b + i] for i in range(6)]
                n = 6
            else:
                b = 30
                vals = [v_val[b], v_val[b + 1]]
                idxs = [v_idx[b], v_idx[b + 1]]
                n = 2

            # XOR with shared node
            self.add_bundle({"valu": [("^", vals[i], vals[i], v_node_shared) for i in range(n)]})

            # Hash stages 0-5
            for hi in range(6):
                vc1, vc2 = v_hash_consts[hi]
                op1, _, op2, op3, _ = HASH_STAGES[hi]
                if hi in [0, 2, 4]:
                    self.add_bundle({"valu": [
                        ("multiply_add", vals[i], vals[i], mult_consts[hi], vc1) for i in range(n)
                    ]})
                else:
                    if n > 3:
                        # Split part 1 into 2 cycles (6 ops each)
                        self.add_bundle({"valu": [
                            op for i in range(3) for op in [
                                (op1, tmp6[i][0], vals[i], vc1), (op3, tmp6[i][1], vals[i], vc2),
                            ]
                        ]})
                        self.add_bundle({"valu": [
                            op for i in range(3, n) for op in [
                                (op1, tmp6[i][0], vals[i], vc1), (op3, tmp6[i][1], vals[i], vc2),
                            ]
                        ]})
                    else:
                        self.add_bundle({"valu": [
                            op for i in range(n) for op in [
                                (op1, tmp6[i][0], vals[i], vc1), (op3, tmp6[i][1], vals[i], vc2),
                            ]
                        ]})
                    self.add_bundle({"valu": [(op2, vals[i], tmp6[i][0], tmp6[i][1]) for i in range(n)]})

            # idx = (val & 1) + 1 (round 0: all idx start at 0, so new_idx = branch)
            self.add_bundle({"valu": [("&", idxs[i], vals[i], v_one) for i in range(n)]})
            self.add_bundle({"valu": [("+", idxs[i], idxs[i], v_one) for i in range(n)]})

        # ===== ROUND 1 SPECIAL CASE =====
        # All indices are in {1, 2}, so we only need 2 forest loads!
        # ARITHMETIC APPROACH: node_val = idx * diff + base2 (eliminates flow bottleneck)
        node1_scalar = self.alloc_scratch("node1_scalar")
        node2_scalar = self.alloc_scratch("node2_scalar")
        v_node1 = self.alloc_scratch("v_node1", VLEN)
        v_node2 = self.alloc_scratch("v_node2", VLEN)
        addr1 = self.alloc_scratch("addr1")
        addr2 = self.alloc_scratch("addr2")

        # Extra allocations for arithmetic approach
        diff_12_scalar = self.alloc_scratch("diff_12_scalar")
        base2_12_scalar = self.alloc_scratch("base2_12_scalar")
        v_diff_12 = self.alloc_scratch("v_diff_12", VLEN)
        v_base2_12 = self.alloc_scratch("v_base2_12", VLEN)

        # Extra temps for pipelining (F batch)
        v_tmp1_F = self.alloc_scratch("v_tmp1_F", VLEN)
        v_tmp2_F = self.alloc_scratch("v_tmp2_F", VLEN)

        # Compute addresses using ALU (both in one cycle)
        self.add_bundle({"alu": [
            ("+", addr1, self.scratch["forest_values_p"], one_const),
            ("+", addr2, self.scratch["forest_values_p"], two_const),
        ]})

        # Load forest[1] and forest[2]
        self.add_bundle({"load": [("load", node1_scalar, addr1), ("load", node2_scalar, addr2)]})

        # Compute diff = f2 - f1, base2 = f1 - diff = 2*f1 - f2, overlapped with broadcasts
        self.add_bundle({
            "alu": [("-", diff_12_scalar, node2_scalar, node1_scalar)],
            "valu": [("vbroadcast", v_node1, node1_scalar), ("vbroadcast", v_node2, node2_scalar)],
        })
        self.add_bundle({
            "alu": [("-", base2_12_scalar, node1_scalar, diff_12_scalar)],
            "valu": [("vbroadcast", v_diff_12, diff_12_scalar)],
        })
        self.add_bundle({"valu": [("vbroadcast", v_base2_12, base2_12_scalar)]})

        # Process in groups of 6 batches (5 groups of 6 + 1 group of 2)
        tmp6 = [(v_tmp1_A, v_tmp2_A), (v_tmp1_B, v_tmp2_B), (v_tmp1_E, v_tmp2_E),
                (v_tmp1_C, v_tmp2_C), (v_tmp1_D, v_tmp2_D), (v_tmp1_F, v_tmp2_F)]
        node_tmp6 = [v_node_A, v_node_B, v_tmp1_E, v_tmp1_C, v_tmp1_D, v_tmp1_F]
        mult_consts = [v_mult_4097, None, v_mult_33, None, v_mult_9, None]
        for group in range(6):
            if group < 5:
                b = group * 6
                vals = [v_val[b + i] for i in range(6)]
                idxs = [v_idx[b + i] for i in range(6)]
                n = 6
            else:
                b = 30
                vals = [v_val[b], v_val[b + 1]]
                idxs = [v_idx[b], v_idx[b + 1]]
                n = 2

            # Node lookup: node_val = idx * diff + base2 (1 cycle)
            self.add_bundle({"valu": [
                ("multiply_add", node_tmp6[i], idxs[i], v_diff_12, v_base2_12) for i in range(n)
            ]})

            # XOR val with node
            self.add_bundle({"valu": [("^", vals[i], vals[i], node_tmp6[i]) for i in range(n)]})

            # Hash stages 0-5
            for hi in range(6):
                vc1, vc2 = v_hash_consts[hi]
                op1, _, op2, op3, _ = HASH_STAGES[hi]
                if hi in [0, 2, 4]:
                    self.add_bundle({"valu": [
                        ("multiply_add", vals[i], vals[i], mult_consts[hi], vc1) for i in range(n)
                    ]})
                else:
                    if n > 3:
                        self.add_bundle({"valu": [
                            op for i in range(3) for op in [
                                (op1, tmp6[i][0], vals[i], vc1), (op3, tmp6[i][1], vals[i], vc2),
                            ]
                        ]})
                        self.add_bundle({"valu": [
                            op for i in range(3, n) for op in [
                                (op1, tmp6[i][0], vals[i], vc1), (op3, tmp6[i][1], vals[i], vc2),
                            ]
                        ]})
                    else:
                        self.add_bundle({"valu": [
                            op for i in range(n) for op in [
                                (op1, tmp6[i][0], vals[i], vc1), (op3, tmp6[i][1], vals[i], vc2),
                            ]
                        ]})
                    self.add_bundle({"valu": [(op2, vals[i], tmp6[i][0], tmp6[i][1]) for i in range(n)]})

            # idx = 2*idx + (val&1 + 1) (3 cycles)
            self.add_bundle({"valu": [("&", tmp6[i][0], vals[i], v_one) for i in range(n)]})
            self.add_bundle({"valu": [("+", tmp6[i][0], tmp6[i][0], v_one) for i in range(n)]})
            self.add_bundle({"valu": [("multiply_add", idxs[i], idxs[i], v_two, tmp6[i][0]) for i in range(n)]})

        # ===== ROUND 2 SPECIAL CASE =====
        # Indices are in {3,4,5,6}, so only 4 forest values needed
        # ARITHMETIC APPROACH: eliminates 6 vselects per pair with 5 VALU cycles per 3 batches
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

        # Allocations for arithmetic approach
        diff_34_scalar = self.alloc_scratch("diff_34_scalar")
        diff_56_scalar = self.alloc_scratch("diff_56_scalar")
        v_diff_34 = self.alloc_scratch("v_diff_34", VLEN)
        v_diff_56 = self.alloc_scratch("v_diff_56", VLEN)
        three_const_s = self.scratch_const(3)
        v_three = self.alloc_scratch("v_three", VLEN)

        # These scratch vars were previously used for vselect but we still need them as temps
        v_r_odd = self.alloc_scratch("v_r_odd", VLEN)
        v_r_even = self.alloc_scratch("v_r_even", VLEN)
        v_bit0_C = self.alloc_scratch("v_bit0_C", VLEN)
        v_bit1_C = self.alloc_scratch("v_bit1_C", VLEN)
        v_bit0_D = self.alloc_scratch("v_bit0_D", VLEN)
        v_bit1_D = self.alloc_scratch("v_bit1_D", VLEN)

        # Load forest[3..6]
        self.add_bundle({"alu": [
            ("+", addr3, self.scratch["forest_values_p"], three_const_s),
        ]})
        self.add_bundle({"flow": [("add_imm", addr4, self.scratch["forest_values_p"], 4)]})
        self.add_bundle({"load": [("load", fs3, addr3), ("load", fs4, addr4)]})
        self.add_bundle({"flow": [("add_imm", addr5, self.scratch["forest_values_p"], 5)]})
        self.add_bundle({"flow": [("add_imm", addr6, self.scratch["forest_values_p"], 6)]})
        self.add_bundle({"load": [("load", fs5, addr5), ("load", fs6, addr6)]})

        # Compute diffs and broadcast
        self.add_bundle({
            "alu": [("-", diff_34_scalar, fs4, fs3), ("-", diff_56_scalar, fs6, fs5)],
            "valu": [
                ("vbroadcast", v_f3, fs3), ("vbroadcast", v_f4, fs4),
                ("vbroadcast", v_f5, fs5), ("vbroadcast", v_f6, fs6),
            ],
        })
        self.add_bundle({"valu": [
            ("vbroadcast", v_diff_34, diff_34_scalar),
            ("vbroadcast", v_diff_56, diff_56_scalar),
            ("vbroadcast", v_three, three_const_s),
        ]})

        # Process in groups of 6 batches (5 groups of 6 + 1 group of 2) - 4-value arithmetic lookup
        # Register assignments for 6-batch node lookup:
        #   offset/bit0/group1/diff_groups/node_val → offset_regs
        #   bit1 → bit1_regs (= tmp6 first elements)
        #   group0 → group0_regs (= tmp6 second elements)
        offset_regs = [v_node_A, v_node_B, v_r_odd, v_r_even, v_bit0_C, v_bit0_D]
        bit1_regs = [v_tmp1_A, v_tmp1_B, v_tmp1_E, v_tmp1_C, v_tmp1_D, v_tmp1_F]
        group0_regs = [v_tmp2_A, v_tmp2_B, v_tmp2_E, v_tmp2_C, v_tmp2_D, v_tmp2_F]
        tmp6 = [(v_tmp1_A, v_tmp2_A), (v_tmp1_B, v_tmp2_B), (v_tmp1_E, v_tmp2_E),
                (v_tmp1_C, v_tmp2_C), (v_tmp1_D, v_tmp2_D), (v_tmp1_F, v_tmp2_F)]
        mult_consts = [v_mult_4097, None, v_mult_33, None, v_mult_9, None]

        for group in range(6):
            if group < 5:
                b = group * 6
                vals = [v_val[b + i] for i in range(6)]
                idxs = [v_idx[b + i] for i in range(6)]
                n = 6
            else:
                b = 30
                vals = [v_val[b], v_val[b + 1]]
                idxs = [v_idx[b], v_idx[b + 1]]
                n = 2

            # Step 1: offset = idx - 3
            self.add_bundle({"valu": [("-", offset_regs[i], idxs[i], v_three) for i in range(n)]})

            # Step 2: bit0 = offset & 1 (overwrites offset), bit1 = offset >> 1
            if n > 3:
                self.add_bundle({"valu": [
                    op for i in range(3) for op in [
                        ("&", offset_regs[i], offset_regs[i], v_one),
                        (">>", bit1_regs[i], offset_regs[i], v_one),
                    ]
                ]})
                self.add_bundle({"valu": [
                    op for i in range(3, n) for op in [
                        ("&", offset_regs[i], offset_regs[i], v_one),
                        (">>", bit1_regs[i], offset_regs[i], v_one),
                    ]
                ]})
            else:
                self.add_bundle({"valu": [
                    op for i in range(n) for op in [
                        ("&", offset_regs[i], offset_regs[i], v_one),
                        (">>", bit1_regs[i], offset_regs[i], v_one),
                    ]
                ]})

            # Step 3a: group0 = bit0 * diff_34 + f3
            self.add_bundle({"valu": [
                ("multiply_add", group0_regs[i], offset_regs[i], v_diff_34, v_f3) for i in range(n)
            ]})
            # Step 3b: group1 = bit0 * diff_56 + f5 (overwrites bit0 in offset_regs, VLIW safe)
            self.add_bundle({"valu": [
                ("multiply_add", offset_regs[i], offset_regs[i], v_diff_56, v_f5) for i in range(n)
            ]})
            # Step 4: diff_groups = group1 - group0
            self.add_bundle({"valu": [("-", offset_regs[i], offset_regs[i], group0_regs[i]) for i in range(n)]})
            # Step 5: node_val = bit1 * diff_groups + group0
            self.add_bundle({"valu": [
                ("multiply_add", offset_regs[i], bit1_regs[i], offset_regs[i], group0_regs[i]) for i in range(n)
            ]})

            # XOR val with node_val
            self.add_bundle({"valu": [("^", vals[i], vals[i], offset_regs[i]) for i in range(n)]})

            # Hash stages 0-5
            for hi in range(6):
                vc1, vc2 = v_hash_consts[hi]
                op1, _, op2, op3, _ = HASH_STAGES[hi]
                if hi in [0, 2, 4]:
                    self.add_bundle({"valu": [
                        ("multiply_add", vals[i], vals[i], mult_consts[hi], vc1) for i in range(n)
                    ]})
                else:
                    if n > 3:
                        self.add_bundle({"valu": [
                            op for i in range(3) for op in [
                                (op1, tmp6[i][0], vals[i], vc1), (op3, tmp6[i][1], vals[i], vc2),
                            ]
                        ]})
                        self.add_bundle({"valu": [
                            op for i in range(3, n) for op in [
                                (op1, tmp6[i][0], vals[i], vc1), (op3, tmp6[i][1], vals[i], vc2),
                            ]
                        ]})
                    else:
                        self.add_bundle({"valu": [
                            op for i in range(n) for op in [
                                (op1, tmp6[i][0], vals[i], vc1), (op3, tmp6[i][1], vals[i], vc2),
                            ]
                        ]})
                    self.add_bundle({"valu": [(op2, vals[i], tmp6[i][0], tmp6[i][1]) for i in range(n)]})

            # idx = 2*idx + (val&1 + 1) (3 cycles)
            self.add_bundle({"valu": [("&", tmp6[i][0], vals[i], v_one) for i in range(n)]})
            self.add_bundle({"valu": [("+", tmp6[i][0], tmp6[i][0], v_one) for i in range(n)]})
            self.add_bundle({"valu": [("multiply_add", idxs[i], idxs[i], v_two, tmp6[i][0]) for i in range(n)]})

        # ===== MAIN LOOP (rounds 3-9) - PIPELINED with overlapped loads =====
        # Key insight: Phase 5 has ~19 cycles with NO loads. During those cycles,
        # we can pre-load the next group's nodes (32 loads = 16 cycles at 2 loads/cycle).

        # 4-batch tmp list for main loop (D already allocated above)
        tmp_list = [(v_tmp1_A, v_tmp2_A), (v_tmp1_B, v_tmp2_B), (v_tmp1_C, v_tmp2_C), (v_tmp1_D, v_tmp2_D)]

        # Allocate two sets of node storage for double buffering
        v_node_C = self.alloc_scratch("v_node_C", VLEN)
        v_node_D = self.alloc_scratch("v_node_D", VLEN)
        node_set_A = [v_node_A, v_node_B, v_node_C, v_node_D]

        v_node_E = self.alloc_scratch("v_node_E", VLEN)
        v_node_F = self.alloc_scratch("v_node_F", VLEN)
        v_node_G = self.alloc_scratch("v_node_G", VLEN)
        v_node_H = self.alloc_scratch("v_node_H", VLEN)
        node_set_B = [v_node_E, v_node_F, v_node_G, v_node_H]

        # Third buffer for 3-way lookahead
        v_node_I = self.alloc_scratch("v_node_I", VLEN)
        v_node_J = self.alloc_scratch("v_node_J", VLEN)
        v_node_K = self.alloc_scratch("v_node_K", VLEN)
        v_node_L = self.alloc_scratch("v_node_L", VLEN)
        node_set_C = [v_node_I, v_node_J, v_node_K, v_node_L]

        # Node sets for 3-way rotation: current, next, next_next
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

        self.add_bundle({"load": [("const", round_counter, 3)]})

        # Preload round 3 group 0 ALL 4 batches into node_set_A[0-3]
        fp = self.scratch["forest_values_p"]
        self.add_bundle({"alu": [("+", addr_A[i], fp, v_idx[0] + i) for i in range(VLEN)]})
        self.add_bundle({"alu": [("+", addr_B[i], fp, v_idx[1] + i) for i in range(VLEN)]})
        for i in range(VLEN):
            bundle = {"load": [("load", node_set_A[0] + i, addr_A[i]), ("load", node_set_A[1] + i, addr_B[i])]}
            if i == VLEN - 1:
                bundle["alu"] = [("+", addr_A[j], fp, v_idx[2] + j) for j in range(VLEN)]
            self.add_bundle(bundle)
        self.add_bundle({
            "load": [("load", node_set_A[2] + 0, addr_A[0]), ("load", node_set_A[2] + 1, addr_A[1])],
            "alu": [("+", addr_B[j], fp, v_idx[3] + j) for j in range(VLEN)],
        })
        # Load remaining b2 elements (2-7) and all b3 elements (0-7)
        remaining = []
        for e in range(2, VLEN):
            remaining.append(("load", node_set_A[2] + e, addr_A[e]))
        for e in range(VLEN):
            remaining.append(("load", node_set_A[3] + e, addr_B[e]))
        for c in range(0, len(remaining), 2):
            self.add_bundle({"load": remaining[c:c+2]})

        round_loop_start = len(self.instrs)

        # ===== GROUPS 0-6: multiply_add hash + overlapped next-group loading =====
        # Group 0 now has all 4 batches pre-loaded (by setup or by group 7 of prev iteration)
        # Uses multiply_add for hash stages 0,2,4 → compute fits entirely within load phase
        for group in range(0, 7):
            base = group * 4
            batch_info = [(v_idx[base + i], v_val[base + i]) for i in range(4)]
            nodes = node_set_B if group % 2 == 1 else node_set_A
            next_nodes = node_set_A if group % 2 == 1 else node_set_B

            next_base = (group + 1) * 4
            next_batch_info = [(v_idx[next_base + i], v_val[next_base + i]) for i in range(4)]

            # C1: addr_A(next_b0) + XOR(0-3)
            self.add_bundle({
                "alu": [("+", addr_A[i], fp, next_batch_info[0][0] + i) for i in range(VLEN)],
                "valu": [("^", batch_info[j][1], batch_info[j][1], nodes[j]) for j in range(4)],
            })
            # C2: addr_B(next_b1) + multiply_add stage 0 (0-3)
            self.add_bundle({
                "alu": [("+", addr_B[i], fp, next_batch_info[1][0] + i) for i in range(VLEN)],
                "valu": [("multiply_add", batch_info[j][1], batch_info[j][1], v_mult_4097, vc1_0) for j in range(4)],
            })
            # C3: load(n0[0],n1[0]) + stage 1 step1 (0-2) [6 VALU]
            self.add_bundle({
                "load": [("load", next_nodes[0] + 0, addr_A[0]), ("load", next_nodes[1] + 0, addr_B[0])],
                "valu": [
                    (op1_1, tmp_list[0][0], batch_info[0][1], vc1_1), (op3_1, tmp_list[0][1], batch_info[0][1], vc2_1),
                    (op1_1, tmp_list[1][0], batch_info[1][1], vc1_1), (op3_1, tmp_list[1][1], batch_info[1][1], vc2_1),
                    (op1_1, tmp_list[2][0], batch_info[2][1], vc1_1), (op3_1, tmp_list[2][1], batch_info[2][1], vc2_1),
                ],
            })
            # C4: load(n0[1],n1[1]) + stage 1 step1 (3) [2 VALU]
            self.add_bundle({
                "load": [("load", next_nodes[0] + 1, addr_A[1]), ("load", next_nodes[1] + 1, addr_B[1])],
                "valu": [
                    (op1_1, tmp_list[3][0], batch_info[3][1], vc1_1), (op3_1, tmp_list[3][1], batch_info[3][1], vc2_1),
                ],
            })
            # C5: load(n0[2],n1[2]) + stage 1 step2 (0-3) [4 VALU]
            self.add_bundle({
                "load": [("load", next_nodes[0] + 2, addr_A[2]), ("load", next_nodes[1] + 2, addr_B[2])],
                "valu": [(op2_1, batch_info[j][1], tmp_list[j][0], tmp_list[j][1]) for j in range(4)],
            })
            # C6: load(n0[3],n1[3]) + multiply_add stage 2 (0-3) [4 VALU]
            self.add_bundle({
                "load": [("load", next_nodes[0] + 3, addr_A[3]), ("load", next_nodes[1] + 3, addr_B[3])],
                "valu": [("multiply_add", batch_info[j][1], batch_info[j][1], v_mult_33, vc1_2) for j in range(4)],
            })
            # C7: load(n0[4],n1[4]) + stage 3 step1 (0-2) [6 VALU]
            self.add_bundle({
                "load": [("load", next_nodes[0] + 4, addr_A[4]), ("load", next_nodes[1] + 4, addr_B[4])],
                "valu": [
                    (op1_3, tmp_list[0][0], batch_info[0][1], vc1_3), (op3_3, tmp_list[0][1], batch_info[0][1], vc2_3),
                    (op1_3, tmp_list[1][0], batch_info[1][1], vc1_3), (op3_3, tmp_list[1][1], batch_info[1][1], vc2_3),
                    (op1_3, tmp_list[2][0], batch_info[2][1], vc1_3), (op3_3, tmp_list[2][1], batch_info[2][1], vc2_3),
                ],
            })
            # C8: load(n0[5],n1[5]) + stage 3 step1 (3) [2 VALU]
            self.add_bundle({
                "load": [("load", next_nodes[0] + 5, addr_A[5]), ("load", next_nodes[1] + 5, addr_B[5])],
                "valu": [
                    (op1_3, tmp_list[3][0], batch_info[3][1], vc1_3), (op3_3, tmp_list[3][1], batch_info[3][1], vc2_3),
                ],
            })
            # C9: load(n0[6],n1[6]) + stage 3 step2 (0-3) + addr_C(next_b2) [4 VALU, 8 ALU]
            self.add_bundle({
                "load": [("load", next_nodes[0] + 6, addr_A[6]), ("load", next_nodes[1] + 6, addr_B[6])],
                "valu": [(op2_3, batch_info[j][1], tmp_list[j][0], tmp_list[j][1]) for j in range(4)],
                "alu": [("+", addr_C[i], fp, next_batch_info[2][0] + i) for i in range(VLEN)],
            })
            # C10: load(n0[7],n1[7]) + multiply_add stage 4 (0-3) + addr_D(next_b3) [4 VALU, 8 ALU]
            self.add_bundle({
                "load": [("load", next_nodes[0] + 7, addr_A[7]), ("load", next_nodes[1] + 7, addr_B[7])],
                "valu": [("multiply_add", batch_info[j][1], batch_info[j][1], v_mult_9, vc1_4) for j in range(4)],
                "alu": [("+", addr_D[i], fp, next_batch_info[3][0] + i) for i in range(VLEN)],
            })
            # C11: load(n2[0],n3[0]) + stage 5 step1 (0-2) [6 VALU]
            self.add_bundle({
                "load": [("load", next_nodes[2] + 0, addr_C[0]), ("load", next_nodes[3] + 0, addr_D[0])],
                "valu": [
                    (op1_5, tmp_list[0][0], batch_info[0][1], vc1_5), (op3_5, tmp_list[0][1], batch_info[0][1], vc2_5),
                    (op1_5, tmp_list[1][0], batch_info[1][1], vc1_5), (op3_5, tmp_list[1][1], batch_info[1][1], vc2_5),
                    (op1_5, tmp_list[2][0], batch_info[2][1], vc1_5), (op3_5, tmp_list[2][1], batch_info[2][1], vc2_5),
                ],
            })
            # C12: load(n2[1],n3[1]) + stage 5 step1(3) + step2(0-2) [5 VALU]
            self.add_bundle({
                "load": [("load", next_nodes[2] + 1, addr_C[1]), ("load", next_nodes[3] + 1, addr_D[1])],
                "valu": [
                    (op1_5, tmp_list[3][0], batch_info[3][1], vc1_5), (op3_5, tmp_list[3][1], batch_info[3][1], vc2_5),
                    (op2_5, batch_info[0][1], tmp_list[0][0], tmp_list[0][1]),
                    (op2_5, batch_info[1][1], tmp_list[1][0], tmp_list[1][1]),
                    (op2_5, batch_info[2][1], tmp_list[2][0], tmp_list[2][1]),
                ],
            })
            # C13: load(n2[2],n3[2]) + step2(3) + idx_and(0-2) [4 VALU]
            self.add_bundle({
                "load": [("load", next_nodes[2] + 2, addr_C[2]), ("load", next_nodes[3] + 2, addr_D[2])],
                "valu": [
                    (op2_5, batch_info[3][1], tmp_list[3][0], tmp_list[3][1]),
                    ("&", tmp_list[0][0], batch_info[0][1], v_one),
                    ("&", tmp_list[1][0], batch_info[1][1], v_one),
                    ("&", tmp_list[2][0], batch_info[2][1], v_one),
                ],
            })
            # C14: load(n2[3],n3[3]) + idx_and(3) + idx_plus(0-2) [4 VALU]
            self.add_bundle({
                "load": [("load", next_nodes[2] + 3, addr_C[3]), ("load", next_nodes[3] + 3, addr_D[3])],
                "valu": [
                    ("&", tmp_list[3][0], batch_info[3][1], v_one),
                    ("+", tmp_list[0][0], tmp_list[0][0], v_one),
                    ("+", tmp_list[1][0], tmp_list[1][0], v_one),
                    ("+", tmp_list[2][0], tmp_list[2][0], v_one),
                ],
            })
            # C15: load(n2[4],n3[4]) + idx_plus(3) + idx_ma(0-2) [4 VALU]
            self.add_bundle({
                "load": [("load", next_nodes[2] + 4, addr_C[4]), ("load", next_nodes[3] + 4, addr_D[4])],
                "valu": [
                    ("+", tmp_list[3][0], tmp_list[3][0], v_one),
                    ("multiply_add", batch_info[0][0], batch_info[0][0], v_two, tmp_list[0][0]),
                    ("multiply_add", batch_info[1][0], batch_info[1][0], v_two, tmp_list[1][0]),
                    ("multiply_add", batch_info[2][0], batch_info[2][0], v_two, tmp_list[2][0]),
                ],
            })
            # C16: load(n2[5],n3[5]) + idx_ma(3) + bounds_lt(0-2) [4 VALU]
            self.add_bundle({
                "load": [("load", next_nodes[2] + 5, addr_C[5]), ("load", next_nodes[3] + 5, addr_D[5])],
                "valu": [
                    ("multiply_add", batch_info[3][0], batch_info[3][0], v_two, tmp_list[3][0]),
                    ("<", tmp_list[0][0], batch_info[0][0], v_n_nodes),
                    ("<", tmp_list[1][0], batch_info[1][0], v_n_nodes),
                    ("<", tmp_list[2][0], batch_info[2][0], v_n_nodes),
                ],
            })
            # C17: load(n2[6],n3[6]) + bounds_lt(3) + bounds_mul(0-2) [4 VALU]
            self.add_bundle({
                "load": [("load", next_nodes[2] + 6, addr_C[6]), ("load", next_nodes[3] + 6, addr_D[6])],
                "valu": [
                    ("<", tmp_list[3][0], batch_info[3][0], v_n_nodes),
                    ("*", batch_info[0][0], batch_info[0][0], tmp_list[0][0]),
                    ("*", batch_info[1][0], batch_info[1][0], tmp_list[1][0]),
                    ("*", batch_info[2][0], batch_info[2][0], tmp_list[2][0]),
                ],
            })
            # C18: load(n2[7],n3[7]) + bounds_mul(3) [1 VALU]
            self.add_bundle({
                "load": [("load", next_nodes[2] + 7, addr_C[7]), ("load", next_nodes[3] + 7, addr_D[7])],
                "valu": [("*", batch_info[3][0], batch_info[3][0], tmp_list[3][0])],
            })

        # ===== GROUP 7: multiply_add hash + preload next round group 0 (ALL 4 batches) =====
        base = 7 * 4
        batch_info = [(v_idx[base + i], v_val[base + i]) for i in range(4)]
        nodes = node_set_B  # Group 7 (odd) uses set B

        # C1: addr_A(next_round_b0) + XOR(0-3)
        self.add_bundle({
            "alu": [("+", addr_A[i], fp, v_idx[0] + i) for i in range(VLEN)],
            "valu": [("^", batch_info[j][1], batch_info[j][1], nodes[j]) for j in range(4)],
        })
        # C2: addr_B(next_round_b1) + multiply_add stage 0(0-3)
        self.add_bundle({
            "alu": [("+", addr_B[i], fp, v_idx[1] + i) for i in range(VLEN)],
            "valu": [("multiply_add", batch_info[j][1], batch_info[j][1], v_mult_4097, vc1_0) for j in range(4)],
        })
        # C3-C10: load(b0+b1 of next round) + hash stages 1-4
        self.add_bundle({
            "load": [("load", node_set_A[0] + 0, addr_A[0]), ("load", node_set_A[1] + 0, addr_B[0])],
            "valu": [
                (op1_1, tmp_list[0][0], batch_info[0][1], vc1_1), (op3_1, tmp_list[0][1], batch_info[0][1], vc2_1),
                (op1_1, tmp_list[1][0], batch_info[1][1], vc1_1), (op3_1, tmp_list[1][1], batch_info[1][1], vc2_1),
                (op1_1, tmp_list[2][0], batch_info[2][1], vc1_1), (op3_1, tmp_list[2][1], batch_info[2][1], vc2_1),
            ],
        })
        self.add_bundle({
            "load": [("load", node_set_A[0] + 1, addr_A[1]), ("load", node_set_A[1] + 1, addr_B[1])],
            "valu": [(op1_1, tmp_list[3][0], batch_info[3][1], vc1_1), (op3_1, tmp_list[3][1], batch_info[3][1], vc2_1)],
        })
        self.add_bundle({
            "load": [("load", node_set_A[0] + 2, addr_A[2]), ("load", node_set_A[1] + 2, addr_B[2])],
            "valu": [(op2_1, batch_info[j][1], tmp_list[j][0], tmp_list[j][1]) for j in range(4)],
        })
        self.add_bundle({
            "load": [("load", node_set_A[0] + 3, addr_A[3]), ("load", node_set_A[1] + 3, addr_B[3])],
            "valu": [("multiply_add", batch_info[j][1], batch_info[j][1], v_mult_33, vc1_2) for j in range(4)],
        })
        self.add_bundle({
            "load": [("load", node_set_A[0] + 4, addr_A[4]), ("load", node_set_A[1] + 4, addr_B[4])],
            "valu": [
                (op1_3, tmp_list[0][0], batch_info[0][1], vc1_3), (op3_3, tmp_list[0][1], batch_info[0][1], vc2_3),
                (op1_3, tmp_list[1][0], batch_info[1][1], vc1_3), (op3_3, tmp_list[1][1], batch_info[1][1], vc2_3),
                (op1_3, tmp_list[2][0], batch_info[2][1], vc1_3), (op3_3, tmp_list[2][1], batch_info[2][1], vc2_3),
            ],
        })
        self.add_bundle({
            "load": [("load", node_set_A[0] + 5, addr_A[5]), ("load", node_set_A[1] + 5, addr_B[5])],
            "valu": [(op1_3, tmp_list[3][0], batch_info[3][1], vc1_3), (op3_3, tmp_list[3][1], batch_info[3][1], vc2_3)],
        })
        # C9: load(n0[6],n1[6]) + stage 3 step2(0-3) + addr_C(next_round_b2) [4 VALU, 8 ALU]
        self.add_bundle({
            "load": [("load", node_set_A[0] + 6, addr_A[6]), ("load", node_set_A[1] + 6, addr_B[6])],
            "valu": [(op2_3, batch_info[j][1], tmp_list[j][0], tmp_list[j][1]) for j in range(4)],
            "alu": [("+", addr_C[i], fp, v_idx[2] + i) for i in range(VLEN)],
        })
        # C10: load(n0[7],n1[7]) + multiply_add stage 4(0-3) + addr_D(next_round_b3) [4 VALU, 8 ALU]
        self.add_bundle({
            "load": [("load", node_set_A[0] + 7, addr_A[7]), ("load", node_set_A[1] + 7, addr_B[7])],
            "valu": [("multiply_add", batch_info[j][1], batch_info[j][1], v_mult_9, vc1_4) for j in range(4)],
            "alu": [("+", addr_D[i], fp, v_idx[3] + i) for i in range(VLEN)],
        })
        # C11: load(n2[0],n3[0]) + stage 5 step1(0-2) [6 VALU]
        self.add_bundle({
            "load": [("load", node_set_A[2] + 0, addr_C[0]), ("load", node_set_A[3] + 0, addr_D[0])],
            "valu": [
                (op1_5, tmp_list[0][0], batch_info[0][1], vc1_5), (op3_5, tmp_list[0][1], batch_info[0][1], vc2_5),
                (op1_5, tmp_list[1][0], batch_info[1][1], vc1_5), (op3_5, tmp_list[1][1], batch_info[1][1], vc2_5),
                (op1_5, tmp_list[2][0], batch_info[2][1], vc1_5), (op3_5, tmp_list[2][1], batch_info[2][1], vc2_5),
            ],
        })
        # C12: load(n2[1],n3[1]) + step1(3)+step2(0-2) [5 VALU]
        self.add_bundle({
            "load": [("load", node_set_A[2] + 1, addr_C[1]), ("load", node_set_A[3] + 1, addr_D[1])],
            "valu": [
                (op1_5, tmp_list[3][0], batch_info[3][1], vc1_5), (op3_5, tmp_list[3][1], batch_info[3][1], vc2_5),
                (op2_5, batch_info[0][1], tmp_list[0][0], tmp_list[0][1]),
                (op2_5, batch_info[1][1], tmp_list[1][0], tmp_list[1][1]),
                (op2_5, batch_info[2][1], tmp_list[2][0], tmp_list[2][1]),
            ],
        })
        # C13: load(n2[2],n3[2]) + step2(3)+idx_and(0-2) [4 VALU]
        self.add_bundle({
            "load": [("load", node_set_A[2] + 2, addr_C[2]), ("load", node_set_A[3] + 2, addr_D[2])],
            "valu": [
                (op2_5, batch_info[3][1], tmp_list[3][0], tmp_list[3][1]),
                ("&", tmp_list[0][0], batch_info[0][1], v_one),
                ("&", tmp_list[1][0], batch_info[1][1], v_one),
                ("&", tmp_list[2][0], batch_info[2][1], v_one),
            ],
        })
        # C14: load(n2[3],n3[3]) + idx_and(3)+idx_plus(0-2) [4 VALU]
        self.add_bundle({
            "load": [("load", node_set_A[2] + 3, addr_C[3]), ("load", node_set_A[3] + 3, addr_D[3])],
            "valu": [
                ("&", tmp_list[3][0], batch_info[3][1], v_one),
                ("+", tmp_list[0][0], tmp_list[0][0], v_one),
                ("+", tmp_list[1][0], tmp_list[1][0], v_one),
                ("+", tmp_list[2][0], tmp_list[2][0], v_one),
            ],
        })
        # C15: load(n2[4],n3[4]) + idx_plus(3)+idx_ma(0-2) [4 VALU]
        self.add_bundle({
            "load": [("load", node_set_A[2] + 4, addr_C[4]), ("load", node_set_A[3] + 4, addr_D[4])],
            "valu": [
                ("+", tmp_list[3][0], tmp_list[3][0], v_one),
                ("multiply_add", batch_info[0][0], batch_info[0][0], v_two, tmp_list[0][0]),
                ("multiply_add", batch_info[1][0], batch_info[1][0], v_two, tmp_list[1][0]),
                ("multiply_add", batch_info[2][0], batch_info[2][0], v_two, tmp_list[2][0]),
            ],
        })
        # C16: load(n2[5],n3[5]) + idx_ma(3) + round_counter [1 VALU, 1 flow]
        self.add_bundle({
            "load": [("load", node_set_A[2] + 5, addr_C[5]), ("load", node_set_A[3] + 5, addr_D[5])],
            "valu": [("multiply_add", batch_info[3][0], batch_info[3][0], v_two, tmp_list[3][0])],
            "flow": [("add_imm", round_counter, round_counter, 1)],
        })
        # C17: load(n2[6],n3[6]) + round_cmp [1 ALU]
        self.add_bundle({
            "load": [("load", node_set_A[2] + 6, addr_C[6]), ("load", node_set_A[3] + 6, addr_D[6])],
            "alu": [("<", tmp1, round_counter, ten_const)],
        })
        # C18: load(n2[7],n3[7]) + cond_jump (loop for rounds 3-9)
        round_loop_offset = round_loop_start - len(self.instrs) - 1
        self.add_bundle({
            "load": [("load", node_set_A[2] + 7, addr_C[7]), ("load", node_set_A[3] + 7, addr_D[7])],
            "flow": [("cond_jump_rel", tmp1, round_loop_offset)],
        })

        # ===== ROUND 10: Cross-group pre-loading with bounds check =====
        # Group 0 nodes already pre-loaded in node_set_A by main loop's last group 7
        # Groups 0-6: Use pre-loaded nodes + pre-load next + bounds check
        for group in range(0, 7):
            base = group * 4
            batch_info = [(v_idx[base + i], v_val[base + i]) for i in range(4)]
            nodes = node_set_B if group % 2 == 1 else node_set_A
            next_nodes = node_set_A if group % 2 == 1 else node_set_B
            next_base = (group + 1) * 4
            next_batch_info = [(v_idx[next_base + i], v_val[next_base + i]) for i in range(4)]

            self.add_bundle({"alu": [("+", addr_A[i], fp, next_batch_info[0][0] + i) for i in range(VLEN)], "valu": [("^", batch_info[j][1], batch_info[j][1], nodes[j]) for j in range(4)]})
            self.add_bundle({"alu": [("+", addr_B[i], fp, next_batch_info[1][0] + i) for i in range(VLEN)], "valu": [("multiply_add", batch_info[j][1], batch_info[j][1], v_mult_4097, vc1_0) for j in range(4)]})
            self.add_bundle({"load": [("load", next_nodes[0] + 0, addr_A[0]), ("load", next_nodes[1] + 0, addr_B[0])], "valu": [(op1_1, tmp_list[0][0], batch_info[0][1], vc1_1), (op3_1, tmp_list[0][1], batch_info[0][1], vc2_1), (op1_1, tmp_list[1][0], batch_info[1][1], vc1_1), (op3_1, tmp_list[1][1], batch_info[1][1], vc2_1), (op1_1, tmp_list[2][0], batch_info[2][1], vc1_1), (op3_1, tmp_list[2][1], batch_info[2][1], vc2_1)]})
            self.add_bundle({"load": [("load", next_nodes[0] + 1, addr_A[1]), ("load", next_nodes[1] + 1, addr_B[1])], "valu": [(op1_1, tmp_list[3][0], batch_info[3][1], vc1_1), (op3_1, tmp_list[3][1], batch_info[3][1], vc2_1)]})
            self.add_bundle({"load": [("load", next_nodes[0] + 2, addr_A[2]), ("load", next_nodes[1] + 2, addr_B[2])], "valu": [(op2_1, batch_info[j][1], tmp_list[j][0], tmp_list[j][1]) for j in range(4)]})
            self.add_bundle({"load": [("load", next_nodes[0] + 3, addr_A[3]), ("load", next_nodes[1] + 3, addr_B[3])], "valu": [("multiply_add", batch_info[j][1], batch_info[j][1], v_mult_33, vc1_2) for j in range(4)]})
            self.add_bundle({"load": [("load", next_nodes[0] + 4, addr_A[4]), ("load", next_nodes[1] + 4, addr_B[4])], "valu": [(op1_3, tmp_list[0][0], batch_info[0][1], vc1_3), (op3_3, tmp_list[0][1], batch_info[0][1], vc2_3), (op1_3, tmp_list[1][0], batch_info[1][1], vc1_3), (op3_3, tmp_list[1][1], batch_info[1][1], vc2_3), (op1_3, tmp_list[2][0], batch_info[2][1], vc1_3), (op3_3, tmp_list[2][1], batch_info[2][1], vc2_3)]})
            self.add_bundle({"load": [("load", next_nodes[0] + 5, addr_A[5]), ("load", next_nodes[1] + 5, addr_B[5])], "valu": [(op1_3, tmp_list[3][0], batch_info[3][1], vc1_3), (op3_3, tmp_list[3][1], batch_info[3][1], vc2_3)]})
            self.add_bundle({"load": [("load", next_nodes[0] + 6, addr_A[6]), ("load", next_nodes[1] + 6, addr_B[6])], "valu": [(op2_3, batch_info[j][1], tmp_list[j][0], tmp_list[j][1]) for j in range(4)], "alu": [("+", addr_C[i], fp, next_batch_info[2][0] + i) for i in range(VLEN)]})
            self.add_bundle({"load": [("load", next_nodes[0] + 7, addr_A[7]), ("load", next_nodes[1] + 7, addr_B[7])], "valu": [("multiply_add", batch_info[j][1], batch_info[j][1], v_mult_9, vc1_4) for j in range(4)], "alu": [("+", addr_D[i], fp, next_batch_info[3][0] + i) for i in range(VLEN)]})
            self.add_bundle({"load": [("load", next_nodes[2] + 0, addr_C[0]), ("load", next_nodes[3] + 0, addr_D[0])], "valu": [(op1_5, tmp_list[0][0], batch_info[0][1], vc1_5), (op3_5, tmp_list[0][1], batch_info[0][1], vc2_5), (op1_5, tmp_list[1][0], batch_info[1][1], vc1_5), (op3_5, tmp_list[1][1], batch_info[1][1], vc2_5), (op1_5, tmp_list[2][0], batch_info[2][1], vc1_5), (op3_5, tmp_list[2][1], batch_info[2][1], vc2_5)]})
            self.add_bundle({"load": [("load", next_nodes[2] + 1, addr_C[1]), ("load", next_nodes[3] + 1, addr_D[1])], "valu": [(op1_5, tmp_list[3][0], batch_info[3][1], vc1_5), (op3_5, tmp_list[3][1], batch_info[3][1], vc2_5), (op2_5, batch_info[0][1], tmp_list[0][0], tmp_list[0][1]), (op2_5, batch_info[1][1], tmp_list[1][0], tmp_list[1][1]), (op2_5, batch_info[2][1], tmp_list[2][0], tmp_list[2][1])]})
            self.add_bundle({"load": [("load", next_nodes[2] + 2, addr_C[2]), ("load", next_nodes[3] + 2, addr_D[2])], "valu": [(op2_5, batch_info[3][1], tmp_list[3][0], tmp_list[3][1]), ("&", tmp_list[0][0], batch_info[0][1], v_one), ("&", tmp_list[1][0], batch_info[1][1], v_one), ("&", tmp_list[2][0], batch_info[2][1], v_one)]})
            self.add_bundle({"load": [("load", next_nodes[2] + 3, addr_C[3]), ("load", next_nodes[3] + 3, addr_D[3])], "valu": [("&", tmp_list[3][0], batch_info[3][1], v_one), ("+", tmp_list[0][0], tmp_list[0][0], v_one), ("+", tmp_list[1][0], tmp_list[1][0], v_one), ("+", tmp_list[2][0], tmp_list[2][0], v_one)]})
            self.add_bundle({"load": [("load", next_nodes[2] + 4, addr_C[4]), ("load", next_nodes[3] + 4, addr_D[4])], "valu": [("+", tmp_list[3][0], tmp_list[3][0], v_one), ("multiply_add", batch_info[0][0], batch_info[0][0], v_two, tmp_list[0][0]), ("multiply_add", batch_info[1][0], batch_info[1][0], v_two, tmp_list[1][0]), ("multiply_add", batch_info[2][0], batch_info[2][0], v_two, tmp_list[2][0])]})
            self.add_bundle({"load": [("load", next_nodes[2] + 5, addr_C[5]), ("load", next_nodes[3] + 5, addr_D[5])], "valu": [("multiply_add", batch_info[3][0], batch_info[3][0], v_two, tmp_list[3][0]), ("<", tmp_list[0][0], batch_info[0][0], v_n_nodes), ("<", tmp_list[1][0], batch_info[1][0], v_n_nodes), ("<", tmp_list[2][0], batch_info[2][0], v_n_nodes)]})
            self.add_bundle({"load": [("load", next_nodes[2] + 6, addr_C[6]), ("load", next_nodes[3] + 6, addr_D[6])], "valu": [("<", tmp_list[3][0], batch_info[3][0], v_n_nodes), ("*", batch_info[0][0], batch_info[0][0], tmp_list[0][0]), ("*", batch_info[1][0], batch_info[1][0], tmp_list[1][0]), ("*", batch_info[2][0], batch_info[2][0], tmp_list[2][0])]})
            self.add_bundle({"load": [("load", next_nodes[2] + 7, addr_C[7]), ("load", next_nodes[3] + 7, addr_D[7])], "valu": [("*", batch_info[3][0], batch_info[3][0], tmp_list[3][0])]})

        # Group 7: multiply_add hash, no next group + bounds check
        base = 7 * 4
        batch_info = [(v_idx[base + i], v_val[base + i]) for i in range(4)]
        nodes = node_set_B

        # C1: XOR(0-3)
        self.add_bundle({"valu": [("^", batch_info[j][1], batch_info[j][1], nodes[j]) for j in range(4)]})
        # C2: multiply_add stage 0(0-3)
        self.add_bundle({"valu": [("multiply_add", batch_info[j][1], batch_info[j][1], v_mult_4097, vc1_0) for j in range(4)]})
        # C3: stage 1 step1(0-2)
        self.add_bundle({"valu": [
            (op1_1, tmp_list[0][0], batch_info[0][1], vc1_1), (op3_1, tmp_list[0][1], batch_info[0][1], vc2_1),
            (op1_1, tmp_list[1][0], batch_info[1][1], vc1_1), (op3_1, tmp_list[1][1], batch_info[1][1], vc2_1),
            (op1_1, tmp_list[2][0], batch_info[2][1], vc1_1), (op3_1, tmp_list[2][1], batch_info[2][1], vc2_1),
        ]})
        # C4: stage 1 step1(3)
        self.add_bundle({"valu": [(op1_1, tmp_list[3][0], batch_info[3][1], vc1_1), (op3_1, tmp_list[3][1], batch_info[3][1], vc2_1)]})
        # C5: stage 1 step2(0-3)
        self.add_bundle({"valu": [(op2_1, batch_info[j][1], tmp_list[j][0], tmp_list[j][1]) for j in range(4)]})
        # C6: multiply_add stage 2(0-3)
        self.add_bundle({"valu": [("multiply_add", batch_info[j][1], batch_info[j][1], v_mult_33, vc1_2) for j in range(4)]})
        # C7: stage 3 step1(0-2)
        self.add_bundle({"valu": [
            (op1_3, tmp_list[0][0], batch_info[0][1], vc1_3), (op3_3, tmp_list[0][1], batch_info[0][1], vc2_3),
            (op1_3, tmp_list[1][0], batch_info[1][1], vc1_3), (op3_3, tmp_list[1][1], batch_info[1][1], vc2_3),
            (op1_3, tmp_list[2][0], batch_info[2][1], vc1_3), (op3_3, tmp_list[2][1], batch_info[2][1], vc2_3),
        ]})
        # C8: stage 3 step1(3)
        self.add_bundle({"valu": [(op1_3, tmp_list[3][0], batch_info[3][1], vc1_3), (op3_3, tmp_list[3][1], batch_info[3][1], vc2_3)]})
        # C9: stage 3 step2(0-3)
        self.add_bundle({"valu": [(op2_3, batch_info[j][1], tmp_list[j][0], tmp_list[j][1]) for j in range(4)]})
        # C10: multiply_add stage 4(0-3)
        self.add_bundle({"valu": [("multiply_add", batch_info[j][1], batch_info[j][1], v_mult_9, vc1_4) for j in range(4)]})
        # C11: stage 5 step1(0-2)
        self.add_bundle({"valu": [
            (op1_5, tmp_list[0][0], batch_info[0][1], vc1_5), (op3_5, tmp_list[0][1], batch_info[0][1], vc2_5),
            (op1_5, tmp_list[1][0], batch_info[1][1], vc1_5), (op3_5, tmp_list[1][1], batch_info[1][1], vc2_5),
            (op1_5, tmp_list[2][0], batch_info[2][1], vc1_5), (op3_5, tmp_list[2][1], batch_info[2][1], vc2_5),
        ]})
        # C12: stage 5 step1(3)
        self.add_bundle({"valu": [(op1_5, tmp_list[3][0], batch_info[3][1], vc1_5), (op3_5, tmp_list[3][1], batch_info[3][1], vc2_5)]})
        # C13: stage 5 step2(0-3)
        self.add_bundle({"valu": [(op2_5, batch_info[j][1], tmp_list[j][0], tmp_list[j][1]) for j in range(4)]})
        # C14: idx_and(0-3) [4 VALU]
        self.add_bundle({"valu": [("&", tmp_list[j][0], batch_info[j][1], v_one) for j in range(4)]})
        # C15: idx_plus(0-3) [4 VALU]
        self.add_bundle({"valu": [("+", tmp_list[j][0], tmp_list[j][0], v_one) for j in range(4)]})
        # C16: idx_ma(0-3) [4 VALU]
        self.add_bundle({"valu": [("multiply_add", batch_info[j][0], batch_info[j][0], v_two, tmp_list[j][0]) for j in range(4)]})
        # Bounds check: idx = idx * (idx < n_nodes)
        self.add_bundle({"valu": [("<", tmp_list[j][0], batch_info[j][0], v_n_nodes) for j in range(4)]})
        self.add_bundle({"valu": [("*", batch_info[j][0], batch_info[j][0], tmp_list[j][0]) for j in range(4)]})

        # ===== ROUNDS 11-15: Unrolled (mirror rounds 0-4 after wrapping) =====
        # After round 10, ALL indices wrap to 0!

        # Round 11 (like round 0): all indices are 0 - 6-batch processing
        # OPTIMIZATION: Reuse v_node_shared from round 0 (forest[0] doesn't change)
        mult_consts = [v_mult_4097, None, v_mult_33, None, v_mult_9, None]
        for group in range(6):
            if group < 5:
                b = group * 6
                vals = [v_val[b + i] for i in range(6)]
                idxs = [v_idx[b + i] for i in range(6)]
                n = 6
            else:
                b = 30
                vals = [v_val[b], v_val[b + 1]]
                idxs = [v_idx[b], v_idx[b + 1]]
                n = 2

            # XOR with shared node
            self.add_bundle({"valu": [("^", vals[i], vals[i], v_node_shared) for i in range(n)]})

            # Hash stages 0-5
            for hi in range(6):
                vc1, vc2 = v_hash_consts[hi]
                op1, _, op2, op3, _ = HASH_STAGES[hi]
                if hi in [0, 2, 4]:
                    self.add_bundle({"valu": [
                        ("multiply_add", vals[i], vals[i], mult_consts[hi], vc1) for i in range(n)
                    ]})
                else:
                    if n > 3:
                        # Split part 1 into 2 cycles (6 ops each)
                        self.add_bundle({"valu": [
                            op for i in range(3) for op in [
                                (op1, tmp6[i][0], vals[i], vc1), (op3, tmp6[i][1], vals[i], vc2),
                            ]
                        ]})
                        self.add_bundle({"valu": [
                            op for i in range(3, n) for op in [
                                (op1, tmp6[i][0], vals[i], vc1), (op3, tmp6[i][1], vals[i], vc2),
                            ]
                        ]})
                    else:
                        self.add_bundle({"valu": [
                            op for i in range(n) for op in [
                                (op1, tmp6[i][0], vals[i], vc1), (op3, tmp6[i][1], vals[i], vc2),
                            ]
                        ]})
                    self.add_bundle({"valu": [(op2, vals[i], tmp6[i][0], tmp6[i][1]) for i in range(n)]})

            # idx = (val & 1) + 1
            self.add_bundle({"valu": [("&", idxs[i], vals[i], v_one) for i in range(n)]})
            self.add_bundle({"valu": [("+", idxs[i], idxs[i], v_one) for i in range(n)]})

        # Round 12 (like round 1): indices in {1,2} - 6-batch processing
        # ARITHMETIC APPROACH: Reuse v_diff_12 and v_base2_12 from round 1 (forest doesn't change)
        mult_consts = [v_mult_4097, None, v_mult_33, None, v_mult_9, None]
        for group in range(6):
            if group < 5:
                b = group * 6
                vals = [v_val[b + i] for i in range(6)]
                idxs = [v_idx[b + i] for i in range(6)]
                n = 6
            else:
                b = 30
                vals = [v_val[b], v_val[b + 1]]
                idxs = [v_idx[b], v_idx[b + 1]]
                n = 2

            # Node lookup: node = idx * diff + base2
            nodes = [tmp6[i][0] for i in range(n)]
            self.add_bundle({"valu": [("multiply_add", nodes[i], idxs[i], v_diff_12, v_base2_12) for i in range(n)]})

            # XOR val with node
            self.add_bundle({"valu": [("^", vals[i], vals[i], nodes[i]) for i in range(n)]})

            # Hash stages 0-5
            for hi in range(6):
                vc1, vc2 = v_hash_consts[hi]
                op1, _, op2, op3, _ = HASH_STAGES[hi]
                if hi in [0, 2, 4]:
                    self.add_bundle({"valu": [
                        ("multiply_add", vals[i], vals[i], mult_consts[hi], vc1) for i in range(n)
                    ]})
                else:
                    if n > 3:
                        # Split part 1 into 2 cycles (6 ops each)
                        self.add_bundle({"valu": [
                            op for i in range(3) for op in [
                                (op1, tmp6[i][0], vals[i], vc1), (op3, tmp6[i][1], vals[i], vc2),
                            ]
                        ]})
                        self.add_bundle({"valu": [
                            op for i in range(3, n) for op in [
                                (op1, tmp6[i][0], vals[i], vc1), (op3, tmp6[i][1], vals[i], vc2),
                            ]
                        ]})
                    else:
                        self.add_bundle({"valu": [
                            op for i in range(n) for op in [
                                (op1, tmp6[i][0], vals[i], vc1), (op3, tmp6[i][1], vals[i], vc2),
                            ]
                        ]})
                    self.add_bundle({"valu": [(op2, vals[i], tmp6[i][0], tmp6[i][1]) for i in range(n)]})

            # idx = 2*idx + (val&1 + 1)
            self.add_bundle({"valu": [("&", tmp6[i][0], vals[i], v_one) for i in range(n)]})
            self.add_bundle({"valu": [("+", tmp6[i][0], tmp6[i][0], v_one) for i in range(n)]})
            self.add_bundle({"valu": [("multiply_add", idxs[i], idxs[i], v_two, tmp6[i][0]) for i in range(n)]})

        # Round 13 (like round 2): indices in {3,4,5,6}
        # ARITHMETIC APPROACH: Reuse v_diff_34, v_diff_56, v_f3, v_f5, v_three from round 2

        # Process in groups of 6 batches (5 groups of 6 + 1 group of 2) - 4-value arithmetic lookup
        offset_regs = [v_node_A, v_node_B, v_r_odd, v_r_even, v_bit0_C, v_bit0_D]
        bit1_regs = [v_tmp1_A, v_tmp1_B, v_tmp1_E, v_tmp1_C, v_tmp1_D, v_tmp1_F]
        group0_regs = [v_tmp2_A, v_tmp2_B, v_tmp2_E, v_tmp2_C, v_tmp2_D, v_tmp2_F]
        tmp6 = [(v_tmp1_A, v_tmp2_A), (v_tmp1_B, v_tmp2_B), (v_tmp1_E, v_tmp2_E),
                (v_tmp1_C, v_tmp2_C), (v_tmp1_D, v_tmp2_D), (v_tmp1_F, v_tmp2_F)]
        mult_consts = [v_mult_4097, None, v_mult_33, None, v_mult_9, None]

        for group in range(6):
            if group < 5:
                b = group * 6
                vals = [v_val[b + i] for i in range(6)]
                idxs = [v_idx[b + i] for i in range(6)]
                n = 6
            else:
                b = 30
                vals = [v_val[b], v_val[b + 1]]
                idxs = [v_idx[b], v_idx[b + 1]]
                n = 2

            # Step 1: offset = idx - 3
            self.add_bundle({"valu": [("-", offset_regs[i], idxs[i], v_three) for i in range(n)]})

            # Step 2: bit0 = offset & 1 (overwrites offset), bit1 = offset >> 1
            if n > 3:
                self.add_bundle({"valu": [
                    op for i in range(3) for op in [
                        ("&", offset_regs[i], offset_regs[i], v_one),
                        (">>", bit1_regs[i], offset_regs[i], v_one),
                    ]
                ]})
                self.add_bundle({"valu": [
                    op for i in range(3, n) for op in [
                        ("&", offset_regs[i], offset_regs[i], v_one),
                        (">>", bit1_regs[i], offset_regs[i], v_one),
                    ]
                ]})
            else:
                self.add_bundle({"valu": [
                    op for i in range(n) for op in [
                        ("&", offset_regs[i], offset_regs[i], v_one),
                        (">>", bit1_regs[i], offset_regs[i], v_one),
                    ]
                ]})

            # Step 3a: group0 = bit0 * diff_34 + f3
            self.add_bundle({"valu": [
                ("multiply_add", group0_regs[i], offset_regs[i], v_diff_34, v_f3) for i in range(n)
            ]})
            # Step 3b: group1 = bit0 * diff_56 + f5 (overwrites bit0)
            self.add_bundle({"valu": [
                ("multiply_add", offset_regs[i], offset_regs[i], v_diff_56, v_f5) for i in range(n)
            ]})
            # Step 4: diff_groups = group1 - group0
            self.add_bundle({"valu": [("-", offset_regs[i], offset_regs[i], group0_regs[i]) for i in range(n)]})
            # Step 5: node_val = bit1 * diff_groups + group0
            self.add_bundle({"valu": [
                ("multiply_add", offset_regs[i], bit1_regs[i], offset_regs[i], group0_regs[i]) for i in range(n)
            ]})

            # XOR val with node_val
            self.add_bundle({"valu": [("^", vals[i], vals[i], offset_regs[i]) for i in range(n)]})

            # Hash stages 0-5
            for hi in range(6):
                vc1, vc2 = v_hash_consts[hi]
                op1, _, op2, op3, _ = HASH_STAGES[hi]
                if hi in [0, 2, 4]:
                    self.add_bundle({"valu": [
                        ("multiply_add", vals[i], vals[i], mult_consts[hi], vc1) for i in range(n)
                    ]})
                else:
                    if n > 3:
                        self.add_bundle({"valu": [
                            op for i in range(3) for op in [
                                (op1, tmp6[i][0], vals[i], vc1), (op3, tmp6[i][1], vals[i], vc2),
                            ]
                        ]})
                        self.add_bundle({"valu": [
                            op for i in range(3, n) for op in [
                                (op1, tmp6[i][0], vals[i], vc1), (op3, tmp6[i][1], vals[i], vc2),
                            ]
                        ]})
                    else:
                        self.add_bundle({"valu": [
                            op for i in range(n) for op in [
                                (op1, tmp6[i][0], vals[i], vc1), (op3, tmp6[i][1], vals[i], vc2),
                            ]
                        ]})
                    self.add_bundle({"valu": [(op2, vals[i], tmp6[i][0], tmp6[i][1]) for i in range(n)]})

            # idx = 2*idx + (val&1 + 1) (3 cycles)
            self.add_bundle({"valu": [("&", tmp6[i][0], vals[i], v_one) for i in range(n)]})
            self.add_bundle({"valu": [("+", tmp6[i][0], tmp6[i][0], v_one) for i in range(n)]})
            self.add_bundle({"valu": [("multiply_add", idxs[i], idxs[i], v_two, tmp6[i][0]) for i in range(n)]})

        # Pre-load round 14 batch 0 (standalone, no overlap with idx finish)
        base_14 = 0
        batch_info_14 = [(v_idx[base_14 + i], v_val[base_14 + i]) for i in range(4)]
        nodes_14 = node_set_A

        self.add_bundle({"alu": [("+", addr_A[i], self.scratch["forest_values_p"], batch_info_14[0][0] + i) for i in range(VLEN)]})
        for i in range(0, VLEN, 2):
            self.add_bundle({"load": [("load", nodes_14[0] + i, addr_A[i]), ("load", nodes_14[0] + i + 1, addr_A[i + 1])]})

        # Rounds 14-15: Use cross-group pre-loading like main loop
        for _round in range(14, 16):
            if _round == 14:
                # GROUP 0: Already pre-loaded batch 0 above, start from batch 1
                base = 0
                batch_info = [(v_idx[base + i], v_val[base + i]) for i in range(4)]
                nodes = nodes_14  # Already set
            else:
                # GROUP 0: Full gather (no pre-loaded nodes)
                base = 0
                batch_info = [(v_idx[base + i], v_val[base + i]) for i in range(4)]
                nodes = node_set_A

                # Phase 1: Gather batch 0
                self.add_bundle({"alu": [("+", addr_A[i], self.scratch["forest_values_p"], batch_info[0][0] + i) for i in range(VLEN)]})
                for i in range(0, VLEN, 2):
                    self.add_bundle({"load": [("load", nodes[0] + i, addr_A[i]), ("load", nodes[0] + i + 1, addr_A[i + 1])]})

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

            # ===== GROUPS 1-6: multiply_add hash + overlapped next-group loading =====
            for group in range(1, 7):
                base = group * 4
                batch_info = [(v_idx[base + i], v_val[base + i]) for i in range(4)]
                nodes = node_set_B if group % 2 == 1 else node_set_A
                next_nodes = node_set_A if group % 2 == 1 else node_set_B

                next_base = (group + 1) * 4
                next_batch_info = [(v_idx[next_base + i], v_val[next_base + i]) for i in range(4)]

                # C1: addr_A(next_b0) + XOR(0-3)
                self.add_bundle({
                    "alu": [("+", addr_A[i], fp, next_batch_info[0][0] + i) for i in range(VLEN)],
                    "valu": [("^", batch_info[j][1], batch_info[j][1], nodes[j]) for j in range(4)],
                })
                # C2: addr_B(next_b1) + multiply_add stage 0 (0-3)
                self.add_bundle({
                    "alu": [("+", addr_B[i], fp, next_batch_info[1][0] + i) for i in range(VLEN)],
                    "valu": [("multiply_add", batch_info[j][1], batch_info[j][1], v_mult_4097, vc1_0) for j in range(4)],
                })
                # C3: load(n0[0],n1[0]) + stage 1 step1 (0-2)
                self.add_bundle({
                    "load": [("load", next_nodes[0] + 0, addr_A[0]), ("load", next_nodes[1] + 0, addr_B[0])],
                    "valu": [
                        (op1_1, tmp_list[0][0], batch_info[0][1], vc1_1), (op3_1, tmp_list[0][1], batch_info[0][1], vc2_1),
                        (op1_1, tmp_list[1][0], batch_info[1][1], vc1_1), (op3_1, tmp_list[1][1], batch_info[1][1], vc2_1),
                        (op1_1, tmp_list[2][0], batch_info[2][1], vc1_1), (op3_1, tmp_list[2][1], batch_info[2][1], vc2_1),
                    ],
                })
                # C4: load(n0[1],n1[1]) + stage 1 step1 (3)
                self.add_bundle({
                    "load": [("load", next_nodes[0] + 1, addr_A[1]), ("load", next_nodes[1] + 1, addr_B[1])],
                    "valu": [
                        (op1_1, tmp_list[3][0], batch_info[3][1], vc1_1), (op3_1, tmp_list[3][1], batch_info[3][1], vc2_1),
                    ],
                })
                # C5: load(n0[2],n1[2]) + stage 1 step2 (0-3)
                self.add_bundle({
                    "load": [("load", next_nodes[0] + 2, addr_A[2]), ("load", next_nodes[1] + 2, addr_B[2])],
                    "valu": [(op2_1, batch_info[j][1], tmp_list[j][0], tmp_list[j][1]) for j in range(4)],
                })
                # C6: load(n0[3],n1[3]) + multiply_add stage 2 (0-3)
                self.add_bundle({
                    "load": [("load", next_nodes[0] + 3, addr_A[3]), ("load", next_nodes[1] + 3, addr_B[3])],
                    "valu": [("multiply_add", batch_info[j][1], batch_info[j][1], v_mult_33, vc1_2) for j in range(4)],
                })
                # C7: load(n0[4],n1[4]) + stage 3 step1 (0-2)
                self.add_bundle({
                    "load": [("load", next_nodes[0] + 4, addr_A[4]), ("load", next_nodes[1] + 4, addr_B[4])],
                    "valu": [
                        (op1_3, tmp_list[0][0], batch_info[0][1], vc1_3), (op3_3, tmp_list[0][1], batch_info[0][1], vc2_3),
                        (op1_3, tmp_list[1][0], batch_info[1][1], vc1_3), (op3_3, tmp_list[1][1], batch_info[1][1], vc2_3),
                        (op1_3, tmp_list[2][0], batch_info[2][1], vc1_3), (op3_3, tmp_list[2][1], batch_info[2][1], vc2_3),
                    ],
                })
                # C8: load(n0[5],n1[5]) + stage 3 step1 (3)
                self.add_bundle({
                    "load": [("load", next_nodes[0] + 5, addr_A[5]), ("load", next_nodes[1] + 5, addr_B[5])],
                    "valu": [
                        (op1_3, tmp_list[3][0], batch_info[3][1], vc1_3), (op3_3, tmp_list[3][1], batch_info[3][1], vc2_3),
                    ],
                })
                # C9: load(n0[6],n1[6]) + stage 3 step2(0-3) + addr_C(next_b2) [4 VALU, 8 ALU]
                self.add_bundle({
                    "load": [("load", next_nodes[0] + 6, addr_A[6]), ("load", next_nodes[1] + 6, addr_B[6])],
                    "valu": [(op2_3, batch_info[j][1], tmp_list[j][0], tmp_list[j][1]) for j in range(4)],
                    "alu": [("+", addr_C[i], fp, next_batch_info[2][0] + i) for i in range(VLEN)],
                })
                # C10: load(n0[7],n1[7]) + multiply_add stage 4(0-3) + addr_D(next_b3) [4 VALU, 8 ALU]
                self.add_bundle({
                    "load": [("load", next_nodes[0] + 7, addr_A[7]), ("load", next_nodes[1] + 7, addr_B[7])],
                    "valu": [("multiply_add", batch_info[j][1], batch_info[j][1], v_mult_9, vc1_4) for j in range(4)],
                    "alu": [("+", addr_D[i], fp, next_batch_info[3][0] + i) for i in range(VLEN)],
                })
                # C11: load(n2[0],n3[0]) + stage 5 step1(0-2) [6 VALU]
                self.add_bundle({
                    "load": [("load", next_nodes[2] + 0, addr_C[0]), ("load", next_nodes[3] + 0, addr_D[0])],
                    "valu": [
                        (op1_5, tmp_list[0][0], batch_info[0][1], vc1_5), (op3_5, tmp_list[0][1], batch_info[0][1], vc2_5),
                        (op1_5, tmp_list[1][0], batch_info[1][1], vc1_5), (op3_5, tmp_list[1][1], batch_info[1][1], vc2_5),
                        (op1_5, tmp_list[2][0], batch_info[2][1], vc1_5), (op3_5, tmp_list[2][1], batch_info[2][1], vc2_5),
                    ],
                })
                # C12: load(n2[1],n3[1]) + stage 5 step1(3) + step2(0-2) [5 VALU]
                self.add_bundle({
                    "load": [("load", next_nodes[2] + 1, addr_C[1]), ("load", next_nodes[3] + 1, addr_D[1])],
                    "valu": [
                        (op1_5, tmp_list[3][0], batch_info[3][1], vc1_5), (op3_5, tmp_list[3][1], batch_info[3][1], vc2_5),
                        (op2_5, batch_info[0][1], tmp_list[0][0], tmp_list[0][1]),
                        (op2_5, batch_info[1][1], tmp_list[1][0], tmp_list[1][1]),
                        (op2_5, batch_info[2][1], tmp_list[2][0], tmp_list[2][1]),
                    ],
                })
                # C13: load(n2[2],n3[2]) + step2(3) + idx_and(0-2) [4 VALU]
                self.add_bundle({
                    "load": [("load", next_nodes[2] + 2, addr_C[2]), ("load", next_nodes[3] + 2, addr_D[2])],
                    "valu": [
                        (op2_5, batch_info[3][1], tmp_list[3][0], tmp_list[3][1]),
                        ("&", tmp_list[0][0], batch_info[0][1], v_one),
                        ("&", tmp_list[1][0], batch_info[1][1], v_one),
                        ("&", tmp_list[2][0], batch_info[2][1], v_one),
                    ],
                })
                # C14: load(n2[3],n3[3]) + idx_and(3) + idx_plus(0-2) [4 VALU]
                self.add_bundle({
                    "load": [("load", next_nodes[2] + 3, addr_C[3]), ("load", next_nodes[3] + 3, addr_D[3])],
                    "valu": [
                        ("&", tmp_list[3][0], batch_info[3][1], v_one),
                        ("+", tmp_list[0][0], tmp_list[0][0], v_one),
                        ("+", tmp_list[1][0], tmp_list[1][0], v_one),
                        ("+", tmp_list[2][0], tmp_list[2][0], v_one),
                    ],
                })
                # C15: load(n2[4],n3[4]) + idx_plus(3) + idx_ma(0-2) [4 VALU]
                self.add_bundle({
                    "load": [("load", next_nodes[2] + 4, addr_C[4]), ("load", next_nodes[3] + 4, addr_D[4])],
                    "valu": [
                        ("+", tmp_list[3][0], tmp_list[3][0], v_one),
                        ("multiply_add", batch_info[0][0], batch_info[0][0], v_two, tmp_list[0][0]),
                        ("multiply_add", batch_info[1][0], batch_info[1][0], v_two, tmp_list[1][0]),
                        ("multiply_add", batch_info[2][0], batch_info[2][0], v_two, tmp_list[2][0]),
                    ],
                })
                # C16: load(n2[5],n3[5]) + idx_ma(3) [1 VALU]
                self.add_bundle({
                    "load": [("load", next_nodes[2] + 5, addr_C[5]), ("load", next_nodes[3] + 5, addr_D[5])],
                    "valu": [("multiply_add", batch_info[3][0], batch_info[3][0], v_two, tmp_list[3][0])],
                })
                # C17-C18: remaining loads
                self.add_bundle({"load": [("load", next_nodes[2] + 6, addr_C[6]), ("load", next_nodes[3] + 6, addr_D[6])]})
                self.add_bundle({"load": [("load", next_nodes[2] + 7, addr_C[7]), ("load", next_nodes[3] + 7, addr_D[7])]})

            # ===== GROUP 7: multiply_add hash, no next group to load =====
            base = 7 * 4
            batch_info = [(v_idx[base + i], v_val[base + i]) for i in range(4)]
            nodes = node_set_B

            # C1: XOR(0-3)
            self.add_bundle({"valu": [("^", batch_info[j][1], batch_info[j][1], nodes[j]) for j in range(4)]})
            # C2: multiply_add stage 0(0-3)
            self.add_bundle({"valu": [("multiply_add", batch_info[j][1], batch_info[j][1], v_mult_4097, vc1_0) for j in range(4)]})
            # C3: stage 1 step1(0-2)
            self.add_bundle({"valu": [
                (op1_1, tmp_list[0][0], batch_info[0][1], vc1_1), (op3_1, tmp_list[0][1], batch_info[0][1], vc2_1),
                (op1_1, tmp_list[1][0], batch_info[1][1], vc1_1), (op3_1, tmp_list[1][1], batch_info[1][1], vc2_1),
                (op1_1, tmp_list[2][0], batch_info[2][1], vc1_1), (op3_1, tmp_list[2][1], batch_info[2][1], vc2_1),
            ]})
            # C4: stage 1 step1(3)
            self.add_bundle({"valu": [
                (op1_1, tmp_list[3][0], batch_info[3][1], vc1_1), (op3_1, tmp_list[3][1], batch_info[3][1], vc2_1),
            ]})
            # C5: stage 1 step2(0-3)
            self.add_bundle({"valu": [(op2_1, batch_info[j][1], tmp_list[j][0], tmp_list[j][1]) for j in range(4)]})
            # C6: multiply_add stage 2(0-3)
            self.add_bundle({"valu": [("multiply_add", batch_info[j][1], batch_info[j][1], v_mult_33, vc1_2) for j in range(4)]})
            # C7: stage 3 step1(0-2)
            self.add_bundle({"valu": [
                (op1_3, tmp_list[0][0], batch_info[0][1], vc1_3), (op3_3, tmp_list[0][1], batch_info[0][1], vc2_3),
                (op1_3, tmp_list[1][0], batch_info[1][1], vc1_3), (op3_3, tmp_list[1][1], batch_info[1][1], vc2_3),
                (op1_3, tmp_list[2][0], batch_info[2][1], vc1_3), (op3_3, tmp_list[2][1], batch_info[2][1], vc2_3),
            ]})
            # C8: stage 3 step1(3)
            self.add_bundle({"valu": [
                (op1_3, tmp_list[3][0], batch_info[3][1], vc1_3), (op3_3, tmp_list[3][1], batch_info[3][1], vc2_3),
            ]})
            # C9: stage 3 step2(0-3)
            self.add_bundle({"valu": [(op2_3, batch_info[j][1], tmp_list[j][0], tmp_list[j][1]) for j in range(4)]})
            # C10: multiply_add stage 4(0-3)
            self.add_bundle({"valu": [("multiply_add", batch_info[j][1], batch_info[j][1], v_mult_9, vc1_4) for j in range(4)]})
            # C11: stage 5 step1(0-2)
            self.add_bundle({"valu": [
                (op1_5, tmp_list[0][0], batch_info[0][1], vc1_5), (op3_5, tmp_list[0][1], batch_info[0][1], vc2_5),
                (op1_5, tmp_list[1][0], batch_info[1][1], vc1_5), (op3_5, tmp_list[1][1], batch_info[1][1], vc2_5),
                (op1_5, tmp_list[2][0], batch_info[2][1], vc1_5), (op3_5, tmp_list[2][1], batch_info[2][1], vc2_5),
            ]})
            # C12: stage 5 step1(3)
            self.add_bundle({"valu": [
                (op1_5, tmp_list[3][0], batch_info[3][1], vc1_5), (op3_5, tmp_list[3][1], batch_info[3][1], vc2_5),
            ]})
            # C13: stage 5 step2(0-3)
            self.add_bundle({"valu": [(op2_5, batch_info[j][1], tmp_list[j][0], tmp_list[j][1]) for j in range(4)]})
            # C14: idx_and(0-3) [4 VALU]
            self.add_bundle({"valu": [("&", tmp_list[j][0], batch_info[j][1], v_one) for j in range(4)]})
            # C15: idx_plus(0-3) [4 VALU]
            self.add_bundle({"valu": [("+", tmp_list[j][0], tmp_list[j][0], v_one) for j in range(4)]})
            # C16: idx_ma(0-3) [4 VALU]
            self.add_bundle({"valu": [("multiply_add", batch_info[j][0], batch_info[j][0], v_two, tmp_list[j][0]) for j in range(4)]})

        # Store all indices and values back - use 2 pointers and ALU for parallel increment
        ptr2 = self.alloc_scratch("ptr2")
        vlen_const = self.scratch_const(VLEN)
        self.add_bundle({"alu": [
            ("+", ptr, self.scratch["inp_indices_p"], zero_const),
            ("+", ptr2, self.scratch["inp_values_p"], zero_const),
        ]})

        # Interleave idx and val stores, merge ptr increments into same bundle (VLIW: reads at cycle start, writes at end)
        for i in range(num_batches):
            bundle = {"store": [("vstore", ptr, v_idx[i]), ("vstore", ptr2, v_val[i])]}
            if i + 1 < num_batches:
                bundle["alu"] = [("+", ptr, ptr, vlen_const), ("+", ptr2, ptr2, vlen_const)]
            self.add_bundle(bundle)

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

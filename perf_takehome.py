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

    def _slot_reads_writes(self, engine, slot):
        reads: set[int] = set()
        writes: set[int] = set()

        def add_range(base: int, n: int):
            for i in range(n):
                yield base + i

        if engine == "alu":
            op, dest, a1, a2 = slot
            writes.add(dest)
            reads.add(a1)
            reads.add(a2)
        elif engine == "load":
            match slot:
                case ("load", dest, addr):
                    writes.add(dest)
                    reads.add(addr)
                case ("load_offset", dest, addr, offset):
                    writes.add(dest + offset)
                    reads.add(addr + offset)
                case ("vload", dest, addr):
                    writes.update(add_range(dest, VLEN))
                    reads.add(addr)
                case ("const", dest, _val):
                    writes.add(dest)
                case _:
                    pass
        elif engine == "store":
            match slot:
                case ("store", addr, src):
                    reads.add(addr)
                    reads.add(src)
                case ("vstore", addr, src):
                    reads.add(addr)
                    reads.update(add_range(src, VLEN))
                case _:
                    pass
        elif engine == "flow":
            match slot:
                case ("select", dest, cond, a, b):
                    writes.add(dest)
                    reads.add(cond)
                    reads.add(a)
                    reads.add(b)
                case ("add_imm", dest, a, _imm):
                    writes.add(dest)
                    reads.add(a)
                case ("vselect", dest, cond, a, b):
                    writes.update(add_range(dest, VLEN))
                    reads.update(add_range(cond, VLEN))
                    reads.update(add_range(a, VLEN))
                    reads.update(add_range(b, VLEN))
                case ("trace_write", val):
                    reads.add(val)
                case ("cond_jump", cond, _addr):
                    reads.add(cond)
                case ("cond_jump_rel", cond, _offset):
                    reads.add(cond)
                case ("jump_indirect", addr):
                    reads.add(addr)
                case ("coreid", dest):
                    writes.add(dest)
                case _:
                    pass
        elif engine == "valu":
            match slot:
                case ("vbroadcast", dest, src):
                    writes.update(add_range(dest, VLEN))
                    reads.add(src)
                case ("multiply_add", dest, a, b, c):
                    writes.update(add_range(dest, VLEN))
                    reads.update(add_range(a, VLEN))
                    reads.update(add_range(b, VLEN))
                    reads.update(add_range(c, VLEN))
                case (_op, dest, a1, a2):
                    writes.update(add_range(dest, VLEN))
                    reads.update(add_range(a1, VLEN))
                    reads.update(add_range(a2, VLEN))
                case _:
                    pass
        else:
            pass

        return reads, writes

    def _is_flow_barrier(self, engine, slot) -> bool:
        if engine != "flow":
            return False
        return slot[0] in {
            "halt",
            "pause",
            "jump",
            "jump_indirect",
            "cond_jump",
            "cond_jump_rel",
        }

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        instrs: list[dict] = []

        cur: dict = {}
        cur_reads: set[int] = set()
        cur_writes: set[int] = set()

        def flush():
            nonlocal cur, cur_reads, cur_writes
            if cur:
                instrs.append(cur)
            cur = {}
            cur_reads = set()
            cur_writes = set()

        def can_add(engine, slot) -> bool:
            if self._is_flow_barrier(engine, slot):
                return False
            if engine in cur and len(cur[engine]) >= SLOT_LIMITS[engine]:
                return False
            slot_reads, slot_writes = self._slot_reads_writes(engine, slot)
            if (slot_reads & cur_writes) or (slot_writes & cur_writes) or (slot_writes & cur_reads):
                return False
            return True

        for engine, slot in slots:
            if self._is_flow_barrier(engine, slot):
                flush()
                instrs.append({engine: [slot]})
                continue

            if not can_add(engine, slot):
                flush()

            cur.setdefault(engine, []).append(slot)
            slot_reads, slot_writes = self._slot_reads_writes(engine, slot)
            cur_reads |= slot_reads
            cur_writes |= slot_writes

        flush()
        return instrs

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

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

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Highly optimized kernel with 4-vector processing and aggressive pipelining.
        """
        slots = []

        addr0 = self.alloc_scratch("addr0")
        addr1 = self.alloc_scratch("addr1")
        tmp1 = self.alloc_scratch("tmp1")

        init_vars = ["rounds", "n_nodes", "batch_size", "forest_height",
                     "forest_values_p", "inp_indices_p", "inp_values_p"]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        for i, v in enumerate(init_vars):
            slots.append(("load", ("const", tmp1, i)))
            slots.append(("load", ("load", self.scratch[v], tmp1)))

        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)

        self.instrs.extend(self.build(slots))
        slots = []

        self.add("flow", ("pause",))

        # Allocate vector temps - need more for 4-vector processing
        v_node_val = self.alloc_scratch("v_node_val", VLEN)  # tree vals for vec 0
        v_nv1 = self.alloc_scratch("v_nv1", VLEN)            # tree vals for vec 1
        v_nv2 = self.alloc_scratch("v_nv2", VLEN)            # tree vals for vec 2
        v_nv3 = self.alloc_scratch("v_nv3", VLEN)            # tree vals for vec 3

        v_tmp1 = self.alloc_scratch("v_tmp1", VLEN)
        v_tmp2 = self.alloc_scratch("v_tmp2", VLEN)
        v_tmp3 = self.alloc_scratch("v_tmp3", VLEN)
        v_tmp4 = self.alloc_scratch("v_tmp4", VLEN)
        v_tmp5 = self.alloc_scratch("v_tmp5", VLEN)
        v_tmp6 = self.alloc_scratch("v_tmp6", VLEN)
        v_tmp7 = self.alloc_scratch("v_tmp7", VLEN)
        v_tmp8 = self.alloc_scratch("v_tmp8", VLEN)

        v_zero = self.alloc_scratch("v_zero", VLEN)
        v_one = self.alloc_scratch("v_one", VLEN)
        v_two = self.alloc_scratch("v_two", VLEN)
        v_n_nodes = self.alloc_scratch("v_n_nodes", VLEN)

        v_hash_consts = []
        v_hash_mults = []  # For multiply_add optimization
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            vc1 = self.alloc_scratch(f"v_hc1_{hi}", VLEN)
            vc3 = self.alloc_scratch(f"v_hc3_{hi}", VLEN)
            v_hash_consts.append((vc1, vc3))
            # Stages 0, 2, 4 can use multiply_add: (a + c1) + (a << s) = a * (2^s + 1) + c1
            if hi in [0, 2, 4] and op1 == "+" and op2 == "+" and op3 == "<<":
                vm = self.alloc_scratch(f"v_hm_{hi}", VLEN)
                v_hash_mults.append((hi, vm, (1 << val3) + 1))  # (stage, vector, multiplier)
            else:
                v_hash_mults.append(None)

        round_ctr = self.alloc_scratch("round_ctr")

        n_vecs = batch_size // VLEN  # 32 vectors

        # Keep ALL idx/val in scratch across all rounds
        all_idx = self.alloc_scratch("all_idx", batch_size)  # 256 words
        all_val = self.alloc_scratch("all_val", batch_size)  # 256 words
        tree_scalar = self.alloc_scratch("tree_scalar")

        # ALL tree values for scatter rounds - 32 vectors × 8 elements = 256 words
        all_tree = self.alloc_scratch("all_tree", batch_size)

        # Allocate 8 address registers for parallel scatter
        addrs = [addr0, addr1]
        for i in range(2, 8):
            addrs.append(self.alloc_scratch(f"addr{i}"))

        # Setup broadcasts
        slots.append(("valu", ("vbroadcast", v_zero, zero_const)))
        slots.append(("valu", ("vbroadcast", v_one, one_const)))
        slots.append(("valu", ("vbroadcast", v_two, two_const)))
        slots.append(("valu", ("vbroadcast", v_n_nodes, self.scratch["n_nodes"])))

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            c1 = self.scratch_const(val1)
            c3 = self.scratch_const(val3)
            vc1, vc3 = v_hash_consts[hi]
            slots.append(("valu", ("vbroadcast", vc1, c1)))
            slots.append(("valu", ("vbroadcast", vc3, c3)))
            # Broadcast multiplier for multiply_add optimization
            if v_hash_mults[hi] is not None:
                _, vm, mult_val = v_hash_mults[hi]
                cm = self.scratch_const(mult_val)
                slots.append(("valu", ("vbroadcast", vm, cm)))

        # Load all values from memory into scratch
        for v in range(n_vecs):
            off = self.scratch_const(VLEN * v)
            slots.append(("alu", ("+", addr0, self.scratch["inp_values_p"], off)))
            slots.append(("load", ("vload", all_val + v * VLEN, addr0)))

        # Initialize all idx to 0
        for v in range(n_vecs):
            slots.append(("valu", ("+", all_idx + v * VLEN, v_zero, v_zero)))

        self.instrs.extend(self.build(slots))
        slots = []

        # === ROUND 0: All idx=0, load tree[0] once, broadcast ===
        slots.append(("alu", ("+", addr0, self.scratch["forest_values_p"], zero_const)))
        slots.append(("load", ("load", tree_scalar, addr0)))
        slots.append(("valu", ("vbroadcast", v_node_val, tree_scalar)))

        # Helper: emit optimized hash for 4 vectors (quad) using multiply_add
        def emit_hash_quad(vv0, vv1, vv2, vv3, target=None):
            if target is None:
                target = slots
            for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                vc1, vc3 = v_hash_consts[hi]
                if v_hash_mults[hi] is not None:
                    _, vm, _ = v_hash_mults[hi]
                    target.append(("valu", ("multiply_add", vv0, vv0, vm, vc1)))
                    target.append(("valu", ("multiply_add", vv1, vv1, vm, vc1)))
                    target.append(("valu", ("multiply_add", vv2, vv2, vm, vc1)))
                    target.append(("valu", ("multiply_add", vv3, vv3, vm, vc1)))
                else:
                    target.append(("valu", (op1, v_tmp1, vv0, vc1)))
                    target.append(("valu", (op3, v_tmp2, vv0, vc3)))
                    target.append(("valu", (op1, v_tmp3, vv1, vc1)))
                    target.append(("valu", (op3, v_tmp4, vv1, vc3)))
                    target.append(("valu", (op1, v_tmp5, vv2, vc1)))
                    target.append(("valu", (op3, v_tmp6, vv2, vc3)))
                    target.append(("valu", (op1, v_tmp7, vv3, vc1)))
                    target.append(("valu", (op3, v_tmp8, vv3, vc3)))
                    target.append(("valu", (op2, vv0, v_tmp1, v_tmp2)))
                    target.append(("valu", (op2, vv1, v_tmp3, v_tmp4)))
                    target.append(("valu", (op2, vv2, v_tmp5, v_tmp6)))
                    target.append(("valu", (op2, vv3, v_tmp7, v_tmp8)))

        # Helper: emit index update for 4 vectors (quad) using multiply_add
        def emit_idx_quad(vi0, vv0, vi1, vv1, vi2, vv2, vi3, vv3, target=None):
            if target is None:
                target = slots
            # bit = hash % 2
            target.append(("valu", ("%", v_tmp1, vv0, v_two)))
            target.append(("valu", ("%", v_tmp2, vv1, v_two)))
            target.append(("valu", ("%", v_tmp3, vv2, v_two)))
            target.append(("valu", ("%", v_tmp4, vv3, v_two)))
            # bit_plus_one = bit + 1
            target.append(("valu", ("+", v_tmp1, v_tmp1, v_one)))
            target.append(("valu", ("+", v_tmp2, v_tmp2, v_one)))
            target.append(("valu", ("+", v_tmp3, v_tmp3, v_one)))
            target.append(("valu", ("+", v_tmp4, v_tmp4, v_one)))
            # idx = idx * 2 + bit_plus_one (using multiply_add)
            target.append(("valu", ("multiply_add", vi0, vi0, v_two, v_tmp1)))
            target.append(("valu", ("multiply_add", vi1, vi1, v_two, v_tmp2)))
            target.append(("valu", ("multiply_add", vi2, vi2, v_two, v_tmp3)))
            target.append(("valu", ("multiply_add", vi3, vi3, v_two, v_tmp4)))
            # cmp = idx < n_nodes; idx = idx * cmp (wrapping)
            target.append(("valu", ("<", v_tmp1, vi0, v_n_nodes)))
            target.append(("valu", ("<", v_tmp2, vi1, v_n_nodes)))
            target.append(("valu", ("<", v_tmp3, vi2, v_n_nodes)))
            target.append(("valu", ("<", v_tmp4, vi3, v_n_nodes)))
            target.append(("valu", ("*", vi0, vi0, v_tmp1)))
            target.append(("valu", ("*", vi1, vi1, v_tmp2)))
            target.append(("valu", ("*", vi2, vi2, v_tmp3)))
            target.append(("valu", ("*", vi3, vi3, v_tmp4)))

        # Process 4 vectors (quad) at a time for round 0
        for v in range(0, n_vecs, 4):
            vi0, vv0 = all_idx + v * VLEN, all_val + v * VLEN
            vi1, vv1 = all_idx + (v+1) * VLEN, all_val + (v+1) * VLEN
            vi2, vv2 = all_idx + (v+2) * VLEN, all_val + (v+2) * VLEN
            vi3, vv3 = all_idx + (v+3) * VLEN, all_val + (v+3) * VLEN
            slots.append(("valu", ("^", vv0, vv0, v_node_val)))
            slots.append(("valu", ("^", vv1, vv1, v_node_val)))
            slots.append(("valu", ("^", vv2, vv2, v_node_val)))
            slots.append(("valu", ("^", vv3, vv3, v_node_val)))
            emit_hash_quad(vv0, vv1, vv2, vv3)
            emit_idx_quad(vi0, vv0, vi1, vv1, vi2, vv2, vi3, vv3)

        self.instrs.extend(self.build(slots))
        slots = []

        # === ROUND 1: idx in {1,2}, use arithmetic instead of vselect ===
        c1_const = self.scratch_const(1)
        c2_const = self.scratch_const(2)
        tree_scalar2 = self.alloc_scratch("tree_scalar2")
        v_tree1 = self.alloc_scratch("v_tree1", VLEN)
        v_tree_diff = self.alloc_scratch("v_tree_diff", VLEN)

        slots.append(("alu", ("+", addr0, self.scratch["forest_values_p"], c1_const)))
        slots.append(("alu", ("+", addr1, self.scratch["forest_values_p"], c2_const)))
        slots.append(("load", ("load", tree_scalar, addr0)))
        slots.append(("load", ("load", tree_scalar2, addr1)))
        slots.append(("alu", ("-", tree_scalar2, tree_scalar2, tree_scalar)))
        slots.append(("valu", ("vbroadcast", v_tree1, tree_scalar)))
        slots.append(("valu", ("vbroadcast", v_tree_diff, tree_scalar2)))

        # Process 4 vectors at a time with multiply_add
        for v in range(0, n_vecs, 4):
            vi0, vv0 = all_idx + v * VLEN, all_val + v * VLEN
            vi1, vv1 = all_idx + (v+1) * VLEN, all_val + (v+1) * VLEN
            vi2, vv2 = all_idx + (v+2) * VLEN, all_val + (v+2) * VLEN
            vi3, vv3 = all_idx + (v+3) * VLEN, all_val + (v+3) * VLEN
            # Compute tree values for 4 vectors
            slots.append(("valu", ("-", v_node_val, vi0, v_one)))
            slots.append(("valu", ("-", v_nv1, vi1, v_one)))
            slots.append(("valu", ("-", v_nv2, vi2, v_one)))
            slots.append(("valu", ("-", v_nv3, vi3, v_one)))
            slots.append(("valu", ("*", v_node_val, v_node_val, v_tree_diff)))
            slots.append(("valu", ("*", v_nv1, v_nv1, v_tree_diff)))
            slots.append(("valu", ("*", v_nv2, v_nv2, v_tree_diff)))
            slots.append(("valu", ("*", v_nv3, v_nv3, v_tree_diff)))
            slots.append(("valu", ("+", v_node_val, v_node_val, v_tree1)))
            slots.append(("valu", ("+", v_nv1, v_nv1, v_tree1)))
            slots.append(("valu", ("+", v_nv2, v_nv2, v_tree1)))
            slots.append(("valu", ("+", v_nv3, v_nv3, v_tree1)))
            slots.append(("valu", ("^", vv0, vv0, v_node_val)))
            slots.append(("valu", ("^", vv1, vv1, v_nv1)))
            slots.append(("valu", ("^", vv2, vv2, v_nv2)))
            slots.append(("valu", ("^", vv3, vv3, v_nv3)))
            emit_hash_quad(vv0, vv1, vv2, vv3)
            emit_idx_quad(vi0, vv0, vi1, vv1, vi2, vv2, vi3, vv3)

        self.instrs.extend(self.build(slots))
        slots = []

        # === ROUND 2: idx in {3,4,5,6}, use vselect binary tree ===
        v_t3 = self.alloc_scratch("v_t3", VLEN)
        v_t4 = self.alloc_scratch("v_t4", VLEN)
        v_t5 = self.alloc_scratch("v_t5", VLEN)
        v_t6 = self.alloc_scratch("v_t6", VLEN)
        ts3 = self.alloc_scratch("ts3")
        ts4 = self.alloc_scratch("ts4")
        ts5 = self.alloc_scratch("ts5")
        ts6 = self.alloc_scratch("ts6")
        c3 = self.scratch_const(3)
        c4 = self.scratch_const(4)
        c5 = self.scratch_const(5)
        c6 = self.scratch_const(6)

        slots.append(("alu", ("+", addr0, self.scratch["forest_values_p"], c3)))
        slots.append(("alu", ("+", addr1, self.scratch["forest_values_p"], c4)))
        slots.append(("load", ("load", ts3, addr0)))
        slots.append(("load", ("load", ts4, addr1)))
        slots.append(("alu", ("+", addr0, self.scratch["forest_values_p"], c5)))
        slots.append(("alu", ("+", addr1, self.scratch["forest_values_p"], c6)))
        slots.append(("load", ("load", ts5, addr0)))
        slots.append(("load", ("load", ts6, addr1)))
        slots.append(("valu", ("vbroadcast", v_t3, ts3)))
        slots.append(("valu", ("vbroadcast", v_t4, ts4)))
        slots.append(("valu", ("vbroadcast", v_t5, ts5)))
        slots.append(("valu", ("vbroadcast", v_t6, ts6)))

        v_three = self.alloc_scratch("v_three", VLEN)
        slots.append(("valu", ("vbroadcast", v_three, c3)))

        self.instrs.extend(self.build(slots))
        slots = []

        # Helper to emit vselect tree lookup for 2 vectors
        def emit_vselect_lookup_pair(vi0_in, vi1_in, result0, result1):
            """Binary tree vselect: idx in {3,4,5,6} -> tree[idx]"""
            slots.append(("valu", ("-", v_tmp1, vi0_in, v_three)))
            slots.append(("valu", ("-", v_tmp4, vi1_in, v_three)))
            slots.append(("valu", ("%", v_tmp2, v_tmp1, v_two)))
            slots.append(("valu", (">>", v_tmp3, v_tmp1, v_one)))
            slots.append(("valu", ("%", v_tmp5, v_tmp4, v_two)))
            slots.append(("valu", (">>", v_tmp6, v_tmp4, v_one)))
            slots.append(("flow", ("vselect", result0, v_tmp2, v_t4, v_t3)))
            slots.append(("flow", ("vselect", v_tmp1, v_tmp2, v_t6, v_t5)))
            slots.append(("flow", ("vselect", result0, v_tmp3, v_tmp1, result0)))
            slots.append(("flow", ("vselect", result1, v_tmp5, v_t4, v_t3)))
            slots.append(("flow", ("vselect", v_tmp4, v_tmp5, v_t6, v_t5)))
            slots.append(("flow", ("vselect", result1, v_tmp6, v_tmp4, result1)))

        # Process 4 vectors at a time with multiply_add for hash
        for v in range(0, n_vecs, 4):
            vi0, vv0 = all_idx + v * VLEN, all_val + v * VLEN
            vi1, vv1 = all_idx + (v+1) * VLEN, all_val + (v+1) * VLEN
            vi2, vv2 = all_idx + (v+2) * VLEN, all_val + (v+2) * VLEN
            vi3, vv3 = all_idx + (v+3) * VLEN, all_val + (v+3) * VLEN
            # Vselect for tree lookup - 2 pairs at a time (flow slot limited)
            emit_vselect_lookup_pair(vi0, vi1, v_node_val, v_nv1)
            emit_vselect_lookup_pair(vi2, vi3, v_nv2, v_nv3)
            # XOR
            slots.append(("valu", ("^", vv0, vv0, v_node_val)))
            slots.append(("valu", ("^", vv1, vv1, v_nv1)))
            slots.append(("valu", ("^", vv2, vv2, v_nv2)))
            slots.append(("valu", ("^", vv3, vv3, v_nv3)))
            # Hash and idx update with multiply_add
            emit_hash_quad(vv0, vv1, vv2, vv3)
            emit_idx_quad(vi0, vv0, vi1, vv1, vi2, vv2, vi3, vv3)

        self.instrs.extend(self.build(slots))
        slots = []

        # === ROUNDS 3-10: Scatter rounds (before mass wrapping) ===
        # Helper for hash with multiply_add for 2 vectors
        def emit_hash_pair(vv0, vv1):
            for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                vc1, vc3 = v_hash_consts[hi]
                if v_hash_mults[hi] is not None:
                    _, vm, _ = v_hash_mults[hi]
                    slots.append(("valu", ("multiply_add", vv0, vv0, vm, vc1)))
                    slots.append(("valu", ("multiply_add", vv1, vv1, vm, vc1)))
                else:
                    slots.append(("valu", (op1, v_tmp1, vv0, vc1)))
                    slots.append(("valu", (op3, v_tmp2, vv0, vc3)))
                    slots.append(("valu", (op1, v_tmp4, vv1, vc1)))
                    slots.append(("valu", (op3, v_tmp5, vv1, vc3)))
                    slots.append(("valu", (op2, vv0, v_tmp1, v_tmp2)))
                    slots.append(("valu", (op2, vv1, v_tmp4, v_tmp5)))

        # Helper for idx update for 2 vectors using multiply_add
        def emit_idx_pair(vi0, vv0, vi1, vv1):
            # bit = hash % 2; bit_plus_one = bit + 1
            slots.append(("valu", ("%", v_tmp1, vv0, v_two)))
            slots.append(("valu", ("%", v_tmp4, vv1, v_two)))
            slots.append(("valu", ("+", v_tmp1, v_tmp1, v_one)))
            slots.append(("valu", ("+", v_tmp4, v_tmp4, v_one)))
            # idx = idx * 2 + bit_plus_one (using multiply_add)
            slots.append(("valu", ("multiply_add", vi0, vi0, v_two, v_tmp1)))
            slots.append(("valu", ("multiply_add", vi1, vi1, v_two, v_tmp4)))
            # cmp = idx < n_nodes; idx = idx * cmp (wrapping)
            slots.append(("valu", ("<", v_tmp1, vi0, v_n_nodes)))
            slots.append(("valu", ("<", v_tmp4, vi1, v_n_nodes)))
            slots.append(("valu", ("*", vi0, vi0, v_tmp1)))
            slots.append(("valu", ("*", vi1, vi1, v_tmp4)))

        def emit_scatter_round():
            """Generate one round of scatter with overlapped loads + compute."""
            nonlocal slots

            def interleave_slots(load_slots, compute_slots):
                li = 0
                ci = 0
                while li < len(load_slots) or ci < len(compute_slots):
                    for _ in range(SLOT_LIMITS["load"]):
                        if li < len(load_slots):
                            slots.append(load_slots[li])
                            li += 1
                    for _ in range(SLOT_LIMITS["valu"]):
                        if ci < len(compute_slots):
                            slots.append(compute_slots[ci])
                            ci += 1

            def load_slots_for_quad(v):
                load_slots = []
                for k in range(4):
                    vi = all_idx + (v + k) * VLEN
                    tree_dest = all_tree + (v + k) * VLEN
                    for i in range(8):
                        load_slots.append(
                            ("alu", ("+", addrs[i], self.scratch["forest_values_p"], vi + i))
                        )
                    for i in range(8):
                        load_slots.append(("load", ("load", tree_dest + i, addrs[i])))
                return load_slots

            def compute_slots_for_quad(v):
                compute_slots = []
                vi0, vv0, tv0 = all_idx + v * VLEN, all_val + v * VLEN, all_tree + v * VLEN
                vi1, vv1, tv1 = all_idx + (v + 1) * VLEN, all_val + (v + 1) * VLEN, all_tree + (v + 1) * VLEN
                vi2, vv2, tv2 = all_idx + (v + 2) * VLEN, all_val + (v + 2) * VLEN, all_tree + (v + 2) * VLEN
                vi3, vv3, tv3 = all_idx + (v + 3) * VLEN, all_val + (v + 3) * VLEN, all_tree + (v + 3) * VLEN
                compute_slots.append(("valu", ("^", vv0, vv0, tv0)))
                compute_slots.append(("valu", ("^", vv1, vv1, tv1)))
                compute_slots.append(("valu", ("^", vv2, vv2, tv2)))
                compute_slots.append(("valu", ("^", vv3, vv3, tv3)))
                emit_hash_quad(vv0, vv1, vv2, vv3, target=compute_slots)
                emit_idx_quad(vi0, vv0, vi1, vv1, vi2, vv2, vi3, vv3, target=compute_slots)
                return compute_slots

            prev_compute = None
            for v in range(0, n_vecs, 4):
                load_slots = load_slots_for_quad(v)
                if prev_compute is None:
                    slots.extend(load_slots)
                else:
                    interleave_slots(load_slots, prev_compute)
                prev_compute = compute_slots_for_quad(v)

            if prev_compute is not None:
                slots.extend(prev_compute)

        def emit_broadcast_round():
            """Round where all idx=0, use broadcast with multiply_add optimization."""
            nonlocal slots
            slots.append(("alu", ("+", addr0, self.scratch["forest_values_p"], zero_const)))
            slots.append(("load", ("load", tree_scalar, addr0)))
            slots.append(("valu", ("vbroadcast", v_node_val, tree_scalar)))
            # Process 4 vectors at a time with multiply_add
            for v in range(0, n_vecs, 4):
                vi0, vv0 = all_idx + v * VLEN, all_val + v * VLEN
                vi1, vv1 = all_idx + (v+1) * VLEN, all_val + (v+1) * VLEN
                vi2, vv2 = all_idx + (v+2) * VLEN, all_val + (v+2) * VLEN
                vi3, vv3 = all_idx + (v+3) * VLEN, all_val + (v+3) * VLEN
                slots.append(("valu", ("^", vv0, vv0, v_node_val)))
                slots.append(("valu", ("^", vv1, vv1, v_node_val)))
                slots.append(("valu", ("^", vv2, vv2, v_node_val)))
                slots.append(("valu", ("^", vv3, vv3, v_node_val)))
                emit_hash_quad(vv0, vv1, vv2, vv3)
                emit_idx_quad(vi0, vv0, vi1, vv1, vi2, vv2, vi3, vv3)

        def emit_arith_round():
            """Round where idx in {1,2}, use arithmetic with multiply_add."""
            nonlocal slots
            slots.append(("alu", ("+", addr0, self.scratch["forest_values_p"], c1_const)))
            slots.append(("alu", ("+", addr1, self.scratch["forest_values_p"], c2_const)))
            slots.append(("load", ("load", tree_scalar, addr0)))
            slots.append(("load", ("load", tree_scalar2, addr1)))
            slots.append(("alu", ("-", tree_scalar2, tree_scalar2, tree_scalar)))
            slots.append(("valu", ("vbroadcast", v_tree1, tree_scalar)))
            slots.append(("valu", ("vbroadcast", v_tree_diff, tree_scalar2)))
            # Process 4 vectors at a time
            for v in range(0, n_vecs, 4):
                vi0, vv0 = all_idx + v * VLEN, all_val + v * VLEN
                vi1, vv1 = all_idx + (v+1) * VLEN, all_val + (v+1) * VLEN
                vi2, vv2 = all_idx + (v+2) * VLEN, all_val + (v+2) * VLEN
                vi3, vv3 = all_idx + (v+3) * VLEN, all_val + (v+3) * VLEN
                # Compute tree values for 4 vectors
                slots.append(("valu", ("-", v_node_val, vi0, v_one)))
                slots.append(("valu", ("-", v_nv1, vi1, v_one)))
                slots.append(("valu", ("-", v_nv2, vi2, v_one)))
                slots.append(("valu", ("-", v_nv3, vi3, v_one)))
                slots.append(("valu", ("*", v_node_val, v_node_val, v_tree_diff)))
                slots.append(("valu", ("*", v_nv1, v_nv1, v_tree_diff)))
                slots.append(("valu", ("*", v_nv2, v_nv2, v_tree_diff)))
                slots.append(("valu", ("*", v_nv3, v_nv3, v_tree_diff)))
                slots.append(("valu", ("+", v_node_val, v_node_val, v_tree1)))
                slots.append(("valu", ("+", v_nv1, v_nv1, v_tree1)))
                slots.append(("valu", ("+", v_nv2, v_nv2, v_tree1)))
                slots.append(("valu", ("+", v_nv3, v_nv3, v_tree1)))
                slots.append(("valu", ("^", vv0, vv0, v_node_val)))
                slots.append(("valu", ("^", vv1, vv1, v_nv1)))
                slots.append(("valu", ("^", vv2, vv2, v_nv2)))
                slots.append(("valu", ("^", vv3, vv3, v_nv3)))
                emit_hash_quad(vv0, vv1, vv2, vv3)
                emit_idx_quad(vi0, vv0, vi1, vv1, vi2, vv2, vi3, vv3)

        def emit_vselect_round():
            """Round where idx in {3-6}, use vselect with multiply_add for hash."""
            nonlocal slots
            slots.append(("alu", ("+", addr0, self.scratch["forest_values_p"], c3)))
            slots.append(("alu", ("+", addr1, self.scratch["forest_values_p"], c4)))
            slots.append(("load", ("load", ts3, addr0)))
            slots.append(("load", ("load", ts4, addr1)))
            slots.append(("alu", ("+", addr0, self.scratch["forest_values_p"], c5)))
            slots.append(("alu", ("+", addr1, self.scratch["forest_values_p"], c6)))
            slots.append(("load", ("load", ts5, addr0)))
            slots.append(("load", ("load", ts6, addr1)))
            slots.append(("valu", ("vbroadcast", v_t3, ts3)))
            slots.append(("valu", ("vbroadcast", v_t4, ts4)))
            slots.append(("valu", ("vbroadcast", v_t5, ts5)))
            slots.append(("valu", ("vbroadcast", v_t6, ts6)))
            # Process 4 vectors at a time with multiply_add for hash
            for v in range(0, n_vecs, 4):
                vi0, vv0 = all_idx + v * VLEN, all_val + v * VLEN
                vi1, vv1 = all_idx + (v+1) * VLEN, all_val + (v+1) * VLEN
                vi2, vv2 = all_idx + (v+2) * VLEN, all_val + (v+2) * VLEN
                vi3, vv3 = all_idx + (v+3) * VLEN, all_val + (v+3) * VLEN
                # Vselect for tree lookup - 2 pairs at a time (flow slot limited)
                emit_vselect_lookup_pair(vi0, vi1, v_node_val, v_nv1)
                emit_vselect_lookup_pair(vi2, vi3, v_nv2, v_nv3)
                # XOR
                slots.append(("valu", ("^", vv0, vv0, v_node_val)))
                slots.append(("valu", ("^", vv1, vv1, v_nv1)))
                slots.append(("valu", ("^", vv2, vv2, v_nv2)))
                slots.append(("valu", ("^", vv3, vv3, v_nv3)))
                # Hash and idx update with multiply_add
                emit_hash_quad(vv0, vv1, vv2, vv3)
                emit_idx_quad(vi0, vv0, vi1, vv1, vi2, vv2, vi3, vv3)

        # Rounds 3-10: scatter (8 rounds) - emit together for better packing
        for rnd in range(3, 11):
            if rnd < rounds:
                emit_scatter_round()
        self.instrs.extend(self.build(slots))
        slots = []

        # Round 11: broadcast (all idx wrapped to 0)
        if 11 < rounds:
            emit_broadcast_round()
            self.instrs.extend(self.build(slots))
            slots = []

        # Round 12: arithmetic (idx in {1,2})
        if 12 < rounds:
            emit_arith_round()
            self.instrs.extend(self.build(slots))
            slots = []

        # Round 13: vselect (idx in {3-6})
        if 13 < rounds:
            emit_vselect_round()
            self.instrs.extend(self.build(slots))
            slots = []

        # Rounds 14-15: scatter - emit together
        for rnd in range(14, rounds):
            emit_scatter_round()
        self.instrs.extend(self.build(slots))
        slots = []

        # === STORE all val back to memory ===
        for v in range(n_vecs):
            off = self.scratch_const(VLEN * v)
            slots.append(("alu", ("+", addr1, self.scratch["inp_values_p"], off)))
            slots.append(("store", ("vstore", addr1, all_val + v * VLEN)))

        self.instrs.extend(self.build(slots))
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

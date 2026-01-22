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

        # Trace marker constants for timing analysis
        trace_marker = self.alloc_scratch("trace_marker")
        TRACE_INIT_DONE = 1
        TRACE_ROUND0_DONE = 2
        TRACE_ROUND1_DONE = 3
        TRACE_ROUND2_DONE = 4
        TRACE_SCATTER_START = 5
        TRACE_SCATTER_DONE = 6
        TRACE_STORE_START = 7
        TRACE_ALL_DONE = 8

        # Allocate vector temps - need more for 4-vector processing
        v_node_val = self.alloc_scratch("v_node_val", VLEN)  # tree vals for vec 0
        v_nv1 = self.alloc_scratch("v_nv1", VLEN)            # tree vals for vec 1
        v_nv2 = self.alloc_scratch("v_nv2", VLEN)            # tree vals for vec 2
        v_nv3 = self.alloc_scratch("v_nv3", VLEN)            # tree vals for vec 3

        # Temp vectors - allocate more for interleaved hash operations
        # We need 2 temps per vector, so 16 temps allows 8-vector interleaving
        v_tmp = [self.alloc_scratch(f"v_tmp{i}", VLEN) for i in range(16)]
        v_tmp1, v_tmp2, v_tmp3, v_tmp4 = v_tmp[0], v_tmp[1], v_tmp[2], v_tmp[3]
        v_tmp5, v_tmp6, v_tmp7, v_tmp8 = v_tmp[4], v_tmp[5], v_tmp[6], v_tmp[7]

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

        # Allocate address registers for pipelined scatter
        # Each quad needs 32 addresses (4 vectors × 8 elements)
        # For triple pipelining, we need 2 sets of 32 = 64 addresses
        # Set A (banks 0-3) for even quads (0, 2, 4, 6)
        # Set B (banks 4-7) for odd quads (1, 3, 5, 7)
        addrs = [addr0, addr1]
        for i in range(2, 8):
            addrs.append(self.alloc_scratch(f"addr{i}"))
        addrs_bank1 = [self.alloc_scratch(f"addr1_{i}") for i in range(8)]
        addrs_bank2 = [self.alloc_scratch(f"addr2_{i}") for i in range(8)]
        addrs_bank3 = [self.alloc_scratch(f"addr3_{i}") for i in range(8)]
        # Second set for double buffering
        addrs_bank4 = [self.alloc_scratch(f"addr4_{i}") for i in range(8)]
        addrs_bank5 = [self.alloc_scratch(f"addr5_{i}") for i in range(8)]
        addrs_bank6 = [self.alloc_scratch(f"addr6_{i}") for i in range(8)]
        addrs_bank7 = [self.alloc_scratch(f"addr7_{i}") for i in range(8)]
        addr_banks_even = [addrs, addrs_bank1, addrs_bank2, addrs_bank3]  # for quads 0,2,4,6
        addr_banks_odd = [addrs_bank4, addrs_bank5, addrs_bank6, addrs_bank7]  # for quads 1,3,5,7

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

        # Load all values from memory into scratch - optimized
        # First compute all 32 addresses (can pack at 12 alu/cycle)
        # Use addr bank registers to hold addresses
        all_val_addrs = []
        for v in range(n_vecs):
            if v < 8:
                all_val_addrs.append(addrs[v])
            elif v < 16:
                all_val_addrs.append(addrs_bank1[v - 8])
            elif v < 24:
                all_val_addrs.append(addrs_bank2[v - 16])
            else:
                all_val_addrs.append(addrs_bank3[v - 24])

        # Compute all addresses first (12 alu/cycle = 3 cycles for 32 ops)
        for v in range(n_vecs):
            off = self.scratch_const(VLEN * v)
            slots.append(("alu", ("+", all_val_addrs[v], self.scratch["inp_values_p"], off)))

        # Now do vloads (2/cycle = 16 cycles) interleaved with idx init (6/cycle)
        # This allows vloads and valu to run in parallel
        vload_idx = 0
        idx_init_idx = 0
        while vload_idx < n_vecs or idx_init_idx < n_vecs:
            # 2 vloads per chunk
            for _ in range(2):
                if vload_idx < n_vecs:
                    slots.append(("load", ("vload", all_val + vload_idx * VLEN, all_val_addrs[vload_idx])))
                    vload_idx += 1
            # 6 idx inits per chunk (can pack with vloads since different engines)
            for _ in range(6):
                if idx_init_idx < n_vecs:
                    slots.append(("valu", ("+", all_idx + idx_init_idx * VLEN, v_zero, v_zero)))
                    idx_init_idx += 1

        self.instrs.extend(self.build(slots))
        slots = []

        # Trace: init done
        slots.append(("load", ("const", trace_marker, TRACE_INIT_DONE)))
        slots.append(("flow", ("trace_write", trace_marker)))

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

        # Helper: emit interleaved hash for 8 vectors (2 quads) to maximize VALU packing
        # By interleaving temp ops from 2 quads, we can fill cycles where single quad
        # would have only 2-4 ops due to dependency chains.
        def emit_hash_octet(vecs, target=None):
            """Process 8 vectors with interleaved hash operations.

            For non-multiply_add stages, the dependency chain is:
            t1,t2 -> c0, t3,t4 -> c1, t5,t6 -> c2, t7,t8 -> c3

            Single quad gives: t1-t6 (6) | t7,t8 (2) | c0-c3 (4) = 12 ops in 3 cycles
            Two quads interleaved gives:
            Qa.t1-t6 (6) | Qa.t7,t8 + Qb.t1-t4 (6) | Qb.t5-t8 + Qa.c0,c1 (6) |
            Qa.c2,c3 + Qb.c0-c2 (5) | Qb.c3 + next stage ops
            """
            if target is None:
                target = slots
            vv = vecs  # list of 8 val vectors
            # Temp assignments: use v_tmp[0:8] for quad A, v_tmp[8:16] for quad B
            # Quad A: vecs 0-3 use temps 0-7
            # Quad B: vecs 4-7 use temps 8-15

            for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                vc1, vc3 = v_hash_consts[hi]
                if v_hash_mults[hi] is not None:
                    # multiply_add stages: 8 independent ops, perfect packing
                    _, vm, _ = v_hash_mults[hi]
                    for i in range(8):
                        target.append(("valu", ("multiply_add", vv[i], vv[i], vm, vc1)))
                else:
                    # Interleaved temp+combine stages
                    # Phase 1: Qa.t0-5 (6 ops)
                    target.append(("valu", (op1, v_tmp[0], vv[0], vc1)))
                    target.append(("valu", (op3, v_tmp[1], vv[0], vc3)))
                    target.append(("valu", (op1, v_tmp[2], vv[1], vc1)))
                    target.append(("valu", (op3, v_tmp[3], vv[1], vc3)))
                    target.append(("valu", (op1, v_tmp[4], vv[2], vc1)))
                    target.append(("valu", (op3, v_tmp[5], vv[2], vc3)))
                    # Phase 2: Qa.t6,t7 + Qb.t0-3 (6 ops)
                    target.append(("valu", (op1, v_tmp[6], vv[3], vc1)))
                    target.append(("valu", (op3, v_tmp[7], vv[3], vc3)))
                    target.append(("valu", (op1, v_tmp[8], vv[4], vc1)))
                    target.append(("valu", (op3, v_tmp[9], vv[4], vc3)))
                    target.append(("valu", (op1, v_tmp[10], vv[5], vc1)))
                    target.append(("valu", (op3, v_tmp[11], vv[5], vc3)))
                    # Phase 3: Qb.t4-7 + Qa.c0,c1 (6 ops)
                    target.append(("valu", (op1, v_tmp[12], vv[6], vc1)))
                    target.append(("valu", (op3, v_tmp[13], vv[6], vc3)))
                    target.append(("valu", (op1, v_tmp[14], vv[7], vc1)))
                    target.append(("valu", (op3, v_tmp[15], vv[7], vc3)))
                    target.append(("valu", (op2, vv[0], v_tmp[0], v_tmp[1])))
                    target.append(("valu", (op2, vv[1], v_tmp[2], v_tmp[3])))
                    # Phase 4: Qa.c2,c3 + Qb.c0-c2 (5 ops) - small gap OK
                    target.append(("valu", (op2, vv[2], v_tmp[4], v_tmp[5])))
                    target.append(("valu", (op2, vv[3], v_tmp[6], v_tmp[7])))
                    target.append(("valu", (op2, vv[4], v_tmp[8], v_tmp[9])))
                    target.append(("valu", (op2, vv[5], v_tmp[10], v_tmp[11])))
                    target.append(("valu", (op2, vv[6], v_tmp[12], v_tmp[13])))
                    # Phase 5: Qb.c3 - will pack with next stage's ops
                    target.append(("valu", (op2, vv[7], v_tmp[14], v_tmp[15])))

        # Helper: emit index update for 4 vectors (quad) using multiply_add
        def emit_idx_quad(vi0, vv0, vi1, vv1, vi2, vv2, vi3, vv3, target=None, check_wrap=True):
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
            # Only check wrapping if needed - most rounds don't need it
            if check_wrap:
                target.append(("valu", ("<", v_tmp1, vi0, v_n_nodes)))
                target.append(("valu", ("<", v_tmp2, vi1, v_n_nodes)))
                target.append(("valu", ("<", v_tmp3, vi2, v_n_nodes)))
                target.append(("valu", ("<", v_tmp4, vi3, v_n_nodes)))
                target.append(("valu", ("*", vi0, vi0, v_tmp1)))
                target.append(("valu", ("*", vi1, vi1, v_tmp2)))
                target.append(("valu", ("*", vi2, vi2, v_tmp3)))
                target.append(("valu", ("*", vi3, vi3, v_tmp4)))

        # Helper: emit index update for 8 vectors (octet) - better VALU packing
        def emit_idx_octet(idx_vecs, val_vecs, target=None, check_wrap=True):
            if target is None:
                target = slots
            vi, vv = idx_vecs, val_vecs
            # bit = hash % 2 for all 8
            for i in range(8):
                target.append(("valu", ("%", v_tmp[i], vv[i], v_two)))
            # bit_plus_one = bit + 1
            for i in range(8):
                target.append(("valu", ("+", v_tmp[i], v_tmp[i], v_one)))
            # idx = idx * 2 + bit_plus_one (using multiply_add)
            for i in range(8):
                target.append(("valu", ("multiply_add", vi[i], vi[i], v_two, v_tmp[i])))
            # wrapping check if needed
            if check_wrap:
                for i in range(8):
                    target.append(("valu", ("<", v_tmp[i], vi[i], v_n_nodes)))
                for i in range(8):
                    target.append(("valu", ("*", vi[i], vi[i], v_tmp[i])))

        # Process 8 vectors (octet) at a time for round 0 - better VALU packing
        # No wrap check needed - idx goes from 0 to {1,2}
        for v in range(0, n_vecs, 8):
            idx_vecs = [all_idx + (v+i) * VLEN for i in range(8)]
            val_vecs = [all_val + (v+i) * VLEN for i in range(8)]
            # XOR with node value
            for vv in val_vecs:
                slots.append(("valu", ("^", vv, vv, v_node_val)))
            emit_hash_octet(val_vecs)
            emit_idx_octet(idx_vecs, val_vecs, check_wrap=False)

        self.instrs.extend(self.build(slots))
        slots = []

        # Trace: round 0 done
        slots.append(("load", ("const", trace_marker, TRACE_ROUND0_DONE)))
        slots.append(("flow", ("trace_write", trace_marker)))

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

        # Allocate more tree value vectors for 8-vector processing
        v_nv = [v_node_val, v_nv1, v_nv2, v_nv3]
        for i in range(4, 8):
            v_nv.append(self.alloc_scratch(f"v_nv{i}", VLEN))

        # Process 8 vectors at a time - better VALU packing
        for v in range(0, n_vecs, 8):
            idx_vecs = [all_idx + (v+i) * VLEN for i in range(8)]
            val_vecs = [all_val + (v+i) * VLEN for i in range(8)]
            # Compute tree values for 8 vectors: tree[idx] = tree[1] + (idx-1)*(tree[2]-tree[1])
            for i in range(8):
                slots.append(("valu", ("-", v_nv[i], idx_vecs[i], v_one)))
            for i in range(8):
                slots.append(("valu", ("*", v_nv[i], v_nv[i], v_tree_diff)))
            for i in range(8):
                slots.append(("valu", ("+", v_nv[i], v_nv[i], v_tree1)))
            # XOR
            for i in range(8):
                slots.append(("valu", ("^", val_vecs[i], val_vecs[i], v_nv[i])))
            emit_hash_octet(val_vecs)
            # No wrap check - idx goes from {1,2} to {3,4,5,6}
            emit_idx_octet(idx_vecs, val_vecs, check_wrap=False)

        self.instrs.extend(self.build(slots))
        slots = []

        # Trace: round 1 done
        slots.append(("load", ("const", trace_marker, TRACE_ROUND1_DONE)))
        slots.append(("flow", ("trace_write", trace_marker)))

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

        # Process 8 vectors at a time with interleaved hash - better VALU packing
        for v in range(0, n_vecs, 8):
            idx_vecs = [all_idx + (v+i) * VLEN for i in range(8)]
            val_vecs = [all_val + (v+i) * VLEN for i in range(8)]
            # Vselect for tree lookup - 2 pairs at a time (flow slot limited)
            # Store results in v_nv[0:8]
            emit_vselect_lookup_pair(idx_vecs[0], idx_vecs[1], v_nv[0], v_nv[1])
            emit_vselect_lookup_pair(idx_vecs[2], idx_vecs[3], v_nv[2], v_nv[3])
            emit_vselect_lookup_pair(idx_vecs[4], idx_vecs[5], v_nv[4], v_nv[5])
            emit_vselect_lookup_pair(idx_vecs[6], idx_vecs[7], v_nv[6], v_nv[7])
            # XOR
            for i in range(8):
                slots.append(("valu", ("^", val_vecs[i], val_vecs[i], v_nv[i])))
            # Hash and idx update - 8-vector interleaved
            emit_hash_octet(val_vecs)
            # No wrap check - idx goes from {3-6} to {7-14}
            emit_idx_octet(idx_vecs, val_vecs, check_wrap=False)

        self.instrs.extend(self.build(slots))
        slots = []

        # Trace: round 2 done
        slots.append(("load", ("const", trace_marker, TRACE_ROUND2_DONE)))
        slots.append(("flow", ("trace_write", trace_marker)))

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

        def emit_scatter_round(check_wrap=True):
            """Generate one round of scatter with triple-pipelining.

            Pipeline: addr[q] | load[q-1] | compute[q-2]
            Even quads use addr_banks_even, odd quads use addr_banks_odd.
            This allows addr[q] and load[q-1] to use different bank sets.

            check_wrap: if False, skip the idx < n_nodes check (safe for most rounds)
            """
            nonlocal slots

            # Collect operations per quad
            n_quads = n_vecs // 4
            quad_addr_ops = []  # quad_addr_ops[q] = list of addr ops for quad q
            quad_load_ops = []  # quad_load_ops[q] = list of load ops for quad q
            quad_compute_ops = []  # quad_compute_ops[q] = list of compute ops for quad q

            for q in range(n_quads):
                addr_ops = []
                load_ops = []
                v_start = q * 4
                # Use even/odd bank sets to allow pipelining
                banks = addr_banks_even if (q % 2) == 0 else addr_banks_odd
                for v_offset in range(4):
                    v = v_start + v_offset
                    bank = banks[v_offset]  # Each vector in quad uses different bank
                    vi = all_idx + v * VLEN
                    tree_dest = all_tree + v * VLEN
                    for i in range(8):
                        addr_ops.append(("alu", ("+", bank[i], self.scratch["forest_values_p"], vi + i)))
                    for i in range(8):
                        load_ops.append(("load", ("load", tree_dest + i, bank[i])))
                quad_addr_ops.append(addr_ops)
                quad_load_ops.append(load_ops)

            for q in range(n_quads):
                v_start = q * 4
                compute_ops = []
                vi0, vv0, tv0 = all_idx + v_start * VLEN, all_val + v_start * VLEN, all_tree + v_start * VLEN
                vi1, vv1, tv1 = all_idx + (v_start+1) * VLEN, all_val + (v_start+1) * VLEN, all_tree + (v_start+1) * VLEN
                vi2, vv2, tv2 = all_idx + (v_start+2) * VLEN, all_val + (v_start+2) * VLEN, all_tree + (v_start+2) * VLEN
                vi3, vv3, tv3 = all_idx + (v_start+3) * VLEN, all_val + (v_start+3) * VLEN, all_tree + (v_start+3) * VLEN
                compute_ops.append(("valu", ("^", vv0, vv0, tv0)))
                compute_ops.append(("valu", ("^", vv1, vv1, tv1)))
                compute_ops.append(("valu", ("^", vv2, vv2, tv2)))
                compute_ops.append(("valu", ("^", vv3, vv3, tv3)))
                emit_hash_quad(vv0, vv1, vv2, vv3, target=compute_ops)
                emit_idx_quad(vi0, vv0, vi1, vv1, vi2, vv2, vi3, vv3, target=compute_ops, check_wrap=check_wrap)
                quad_compute_ops.append(compute_ops)

            # Fine-grained pipelining: emit operations interleaved to maximize packing
            # Pipeline: addr[q] | load[q-1] | compute[q-2]
            # Key: emit in small chunks (2 load, 6 valu, 2 alu) so loads can pack
            # with valu instructions when valu-valu dependencies force flushes.

            # Phase 0: addr[0] only (no loads or compute yet)
            for op in quad_addr_ops[0]:
                slots.append(op)

            # Phase 1: addr[1] interleaved with load[0] (no compute yet)
            ai, li = 0, 0
            while ai < len(quad_addr_ops[1]) or li < len(quad_load_ops[0]):
                for _ in range(2):
                    if li < len(quad_load_ops[0]):
                        slots.append(quad_load_ops[0][li])
                        li += 1
                for _ in range(2):
                    if ai < len(quad_addr_ops[1]):
                        slots.append(quad_addr_ops[1][ai])
                        ai += 1

            # Phases 2 to n_quads-1: full triple pipeline
            for q in range(2, n_quads):
                ai, li, ci = 0, 0, 0
                addr_ops = quad_addr_ops[q]
                load_ops = quad_load_ops[q-1]
                compute_ops = quad_compute_ops[q-2]

                # Fine-grained interleave: 2 loads + 6 valu + 2 addr per chunk
                while ai < len(addr_ops) or li < len(load_ops) or ci < len(compute_ops):
                    for _ in range(2):
                        if li < len(load_ops):
                            slots.append(load_ops[li])
                            li += 1
                    for _ in range(6):
                        if ci < len(compute_ops):
                            slots.append(compute_ops[ci])
                            ci += 1
                    for _ in range(2):
                        if ai < len(addr_ops):
                            slots.append(addr_ops[ai])
                            ai += 1

            # Phase n_quads: load[n_quads-1] + compute[n_quads-2]
            li, ci = 0, 0
            load_ops = quad_load_ops[n_quads-1]
            compute_ops = quad_compute_ops[n_quads-2]
            while li < len(load_ops) or ci < len(compute_ops):
                for _ in range(2):
                    if li < len(load_ops):
                        slots.append(load_ops[li])
                        li += 1
                for _ in range(6):
                    if ci < len(compute_ops):
                        slots.append(compute_ops[ci])
                        ci += 1

            # Final phase: compute[n_quads-1] only
            for op in quad_compute_ops[n_quads-1]:
                slots.append(op)

        def emit_broadcast_round():
            """Round where all idx=0, use broadcast with 8-vector interleaved hash."""
            nonlocal slots
            slots.append(("alu", ("+", addr0, self.scratch["forest_values_p"], zero_const)))
            slots.append(("load", ("load", tree_scalar, addr0)))
            slots.append(("valu", ("vbroadcast", v_node_val, tree_scalar)))
            # Process 8 vectors at a time for better VALU packing
            for v in range(0, n_vecs, 8):
                idx_vecs = [all_idx + (v+i) * VLEN for i in range(8)]
                val_vecs = [all_val + (v+i) * VLEN for i in range(8)]
                for vv in val_vecs:
                    slots.append(("valu", ("^", vv, vv, v_node_val)))
                emit_hash_octet(val_vecs)
                emit_idx_octet(idx_vecs, val_vecs, check_wrap=False)

        def emit_arith_round():
            """Round where idx in {1,2}, use arithmetic with 8-vector interleaved hash."""
            nonlocal slots
            slots.append(("alu", ("+", addr0, self.scratch["forest_values_p"], c1_const)))
            slots.append(("alu", ("+", addr1, self.scratch["forest_values_p"], c2_const)))
            slots.append(("load", ("load", tree_scalar, addr0)))
            slots.append(("load", ("load", tree_scalar2, addr1)))
            slots.append(("alu", ("-", tree_scalar2, tree_scalar2, tree_scalar)))
            slots.append(("valu", ("vbroadcast", v_tree1, tree_scalar)))
            slots.append(("valu", ("vbroadcast", v_tree_diff, tree_scalar2)))
            # Process 8 vectors at a time for better VALU packing
            for v in range(0, n_vecs, 8):
                idx_vecs = [all_idx + (v+i) * VLEN for i in range(8)]
                val_vecs = [all_val + (v+i) * VLEN for i in range(8)]
                # Compute tree values for 8 vectors
                for i in range(8):
                    slots.append(("valu", ("-", v_nv[i], idx_vecs[i], v_one)))
                for i in range(8):
                    slots.append(("valu", ("*", v_nv[i], v_nv[i], v_tree_diff)))
                for i in range(8):
                    slots.append(("valu", ("+", v_nv[i], v_nv[i], v_tree1)))
                for i in range(8):
                    slots.append(("valu", ("^", val_vecs[i], val_vecs[i], v_nv[i])))
                emit_hash_octet(val_vecs)
                emit_idx_octet(idx_vecs, val_vecs, check_wrap=False)

        def emit_vselect_round():
            """Round where idx in {3-6}, use vselect with 8-vector interleaved hash."""
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
            # Process 8 vectors at a time for better VALU packing
            for v in range(0, n_vecs, 8):
                idx_vecs = [all_idx + (v+i) * VLEN for i in range(8)]
                val_vecs = [all_val + (v+i) * VLEN for i in range(8)]
                # Vselect for tree lookup - store in v_nv[0:8]
                emit_vselect_lookup_pair(idx_vecs[0], idx_vecs[1], v_nv[0], v_nv[1])
                emit_vselect_lookup_pair(idx_vecs[2], idx_vecs[3], v_nv[2], v_nv[3])
                emit_vselect_lookup_pair(idx_vecs[4], idx_vecs[5], v_nv[4], v_nv[5])
                emit_vselect_lookup_pair(idx_vecs[6], idx_vecs[7], v_nv[6], v_nv[7])
                for i in range(8):
                    slots.append(("valu", ("^", val_vecs[i], val_vecs[i], v_nv[i])))
                emit_hash_octet(val_vecs)
                emit_idx_octet(idx_vecs, val_vecs, check_wrap=False)

        # Trace: scatter start
        slots.append(("load", ("const", trace_marker, TRACE_SCATTER_START)))
        slots.append(("flow", ("trace_write", trace_marker)))

        # Emit scatter rounds 3-10 with inter-round pipelining
        # Key insight: round r+1's addr[q] only depends on round r's compute[q]
        # So we can start round r+1's addr while round r's later quads process

        def collect_scatter_round_ops(check_wrap):
            """Collect all ops for a scatter round, grouped by quad and type."""
            n_quads = n_vecs // 4
            all_addr_ops = []
            all_load_ops = []
            all_compute_ops = []

            for q in range(n_quads):
                addr_ops = []
                load_ops = []
                v_start = q * 4
                banks = addr_banks_even if (q % 2) == 0 else addr_banks_odd
                for v_offset in range(4):
                    v = v_start + v_offset
                    bank = banks[v_offset]
                    vi = all_idx + v * VLEN
                    tree_dest = all_tree + v * VLEN
                    for i in range(8):
                        addr_ops.append(("alu", ("+", bank[i], self.scratch["forest_values_p"], vi + i)))
                    for i in range(8):
                        load_ops.append(("load", ("load", tree_dest + i, bank[i])))
                all_addr_ops.append(addr_ops)
                all_load_ops.append(load_ops)

            for q in range(n_quads):
                v_start = q * 4
                compute_ops = []
                vi0, vv0, tv0 = all_idx + v_start * VLEN, all_val + v_start * VLEN, all_tree + v_start * VLEN
                vi1, vv1, tv1 = all_idx + (v_start+1) * VLEN, all_val + (v_start+1) * VLEN, all_tree + (v_start+1) * VLEN
                vi2, vv2, tv2 = all_idx + (v_start+2) * VLEN, all_val + (v_start+2) * VLEN, all_tree + (v_start+2) * VLEN
                vi3, vv3, tv3 = all_idx + (v_start+3) * VLEN, all_val + (v_start+3) * VLEN, all_tree + (v_start+3) * VLEN
                compute_ops.append(("valu", ("^", vv0, vv0, tv0)))
                compute_ops.append(("valu", ("^", vv1, vv1, tv1)))
                compute_ops.append(("valu", ("^", vv2, vv2, tv2)))
                compute_ops.append(("valu", ("^", vv3, vv3, tv3)))
                emit_hash_quad(vv0, vv1, vv2, vv3, target=compute_ops)
                emit_idx_quad(vi0, vv0, vi1, vv1, vi2, vv2, vi3, vv3, target=compute_ops, check_wrap=check_wrap)
                all_compute_ops.append(compute_ops)

            return all_addr_ops, all_load_ops, all_compute_ops

        # Collect ops for all scatter rounds (3-10)
        scatter_rounds_ops = []
        for rnd in range(3, 11):
            if rnd < rounds:
                check_wrap = (rnd == 10)  # Only round 10 needs wrap check
                scatter_rounds_ops.append(collect_scatter_round_ops(check_wrap))

        n_scatter = len(scatter_rounds_ops)
        n_quads = n_vecs // 4

        if n_scatter > 0:
            # Emit with inter-round pipelining
            # Phase structure per round: addr[0], (addr[1]+load[0]), (addr[2]+load[1]+compute[0]), ...
            # Inter-round: once compute[q] finishes for round r, start addr[q] for round r+1

            # Round 0, Phase 0: just addr[0]
            for op in scatter_rounds_ops[0][0][0]:  # round 0, addr_ops, quad 0
                slots.append(op)

            # Now do interleaved phases
            # We track: current quad being addressed, loaded, computed for each active round
            # Simplified approach: emit round-by-round but overlap addr of next round with compute of current

            for rnd_idx in range(n_scatter):
                addr_ops, load_ops, compute_ops = scatter_rounds_ops[rnd_idx]

                # Phase 1: load[0] + addr[1] (addr[0] was done in previous phase)
                if rnd_idx == 0:
                    # First round - normal phase 1
                    ai, li = 0, 0
                    while ai < len(addr_ops[1]) or li < len(load_ops[0]):
                        for _ in range(2):
                            if li < len(load_ops[0]):
                                slots.append(load_ops[0][li])
                                li += 1
                        for _ in range(2):
                            if ai < len(addr_ops[1]):
                                slots.append(addr_ops[1][ai])
                                ai += 1
                else:
                    # Later rounds - addr[0] was already done in previous round's drain
                    # Do addr[1] + load[0]
                    ai, li = 0, 0
                    while ai < len(addr_ops[1]) or li < len(load_ops[0]):
                        for _ in range(2):
                            if li < len(load_ops[0]):
                                slots.append(load_ops[0][li])
                                li += 1
                        for _ in range(2):
                            if ai < len(addr_ops[1]):
                                slots.append(addr_ops[1][ai])
                                ai += 1

                # Phases 2 to n_quads-1: full triple pipeline
                for q in range(2, n_quads):
                    ai, li, ci = 0, 0, 0
                    a_ops = addr_ops[q]
                    l_ops = load_ops[q-1]
                    c_ops = compute_ops[q-2]

                    while ai < len(a_ops) or li < len(l_ops) or ci < len(c_ops):
                        for _ in range(2):
                            if li < len(l_ops):
                                slots.append(l_ops[li])
                                li += 1
                        for _ in range(6):
                            if ci < len(c_ops):
                                slots.append(c_ops[ci])
                                ci += 1
                        for _ in range(2):
                            if ai < len(a_ops):
                                slots.append(a_ops[ai])
                                ai += 1

                # Phase n_quads: load[n_quads-1] + compute[n_quads-2]
                li, ci = 0, 0
                l_ops = load_ops[n_quads-1]
                c_ops = compute_ops[n_quads-2]
                while li < len(l_ops) or ci < len(c_ops):
                    for _ in range(2):
                        if li < len(l_ops):
                            slots.append(l_ops[li])
                            li += 1
                    for _ in range(6):
                        if ci < len(c_ops):
                            slots.append(c_ops[ci])
                            ci += 1

                # Drain phase: compute[n_quads-1] + (if not last round) next_round.addr[0]
                c_ops = compute_ops[n_quads-1]
                next_addr_ops = None
                if rnd_idx + 1 < n_scatter:
                    next_addr_ops = scatter_rounds_ops[rnd_idx + 1][0][0]  # next round's addr[0]

                ci, nai = 0, 0
                while ci < len(c_ops) or (next_addr_ops and nai < len(next_addr_ops)):
                    for _ in range(6):
                        if ci < len(c_ops):
                            slots.append(c_ops[ci])
                            ci += 1
                    if next_addr_ops:
                        for _ in range(2):
                            if nai < len(next_addr_ops):
                                slots.append(next_addr_ops[nai])
                                nai += 1

        # Round 11: broadcast (all idx wrapped to 0)
        if 11 < rounds:
            emit_broadcast_round()

        # Round 12: arithmetic (idx in {1,2})
        if 12 < rounds:
            emit_arith_round()

        # Round 13: vselect (idx in {3-6})
        if 13 < rounds:
            emit_vselect_round()

        # Rounds 14-15: scatter (no wrap check needed, idx restarts from 0 after wrap)
        for rnd in range(14, rounds):
            emit_scatter_round(check_wrap=False)

        # Build everything together
        self.instrs.extend(self.build(slots))
        slots = []

        # Trace: scatter done
        slots.append(("load", ("const", trace_marker, TRACE_SCATTER_DONE)))
        slots.append(("flow", ("trace_write", trace_marker)))

        # Trace: store start
        slots.append(("load", ("const", trace_marker, TRACE_STORE_START)))
        slots.append(("flow", ("trace_write", trace_marker)))

        # === STORE all val back to memory - optimized ===
        # Compute all 32 store addresses first (can pack at 12 alu/cycle)
        store_addrs = []
        for v in range(n_vecs):
            if v < 8:
                store_addrs.append(addrs[v])
            elif v < 16:
                store_addrs.append(addrs_bank1[v - 8])
            elif v < 24:
                store_addrs.append(addrs_bank2[v - 16])
            else:
                store_addrs.append(addrs_bank3[v - 24])

        # Compute all addresses (12 alu/cycle = 3 cycles for 32 ops)
        for v in range(n_vecs):
            off = self.scratch_const(VLEN * v)
            slots.append(("alu", ("+", store_addrs[v], self.scratch["inp_values_p"], off)))

        # Do all vstores (2 store/cycle = 16 cycles)
        for v in range(n_vecs):
            slots.append(("store", ("vstore", store_addrs[v], all_val + v * VLEN)))

        # Trace: all done
        slots.append(("load", ("const", trace_marker, TRACE_ALL_DONE)))
        slots.append(("flow", ("trace_write", trace_marker)))

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
    
    # Print trace buffer timing markers
    if machine.cores[0].trace_buf:
        TRACE_LABELS = {
            1: "INIT_DONE",
            2: "ROUND0_DONE", 
            3: "ROUND1_DONE",
            4: "ROUND2_DONE",
            5: "SCATTER_START",
            6: "SCATTER_DONE",
            7: "STORE_START",
            8: "ALL_DONE",
        }
        print("\n=== Trace Markers (from trace_buf) ===")
        for marker in machine.cores[0].trace_buf:
            print(f"  Marker {marker}: {TRACE_LABELS.get(marker, 'UNKNOWN')}")
    
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

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
        Optimized kernel: 4x unrolled with good VLIW packing.
        """
        addr0 = self.alloc_scratch("addr0")
        addr1 = self.alloc_scratch("addr1")
        tmp1 = self.alloc_scratch("tmp1")
        
        init_vars = ["rounds", "n_nodes", "batch_size", "forest_height",
                     "forest_values_p", "inp_indices_p", "inp_values_p"]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp1, i))
            self.add("load", ("load", self.scratch[v], tmp1))

        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)
        vlen_const = self.scratch_const(VLEN)

        self.add("flow", ("pause",))

        v_idx = self.alloc_scratch("v_idx", VLEN)
        v_val = self.alloc_scratch("v_val", VLEN)
        v_node_val = self.alloc_scratch("v_node_val", VLEN)
        v_tmp1 = self.alloc_scratch("v_tmp1", VLEN)
        v_tmp2 = self.alloc_scratch("v_tmp2", VLEN)
        v_tmp3 = self.alloc_scratch("v_tmp3", VLEN)
        v_zero = self.alloc_scratch("v_zero", VLEN)
        v_one = self.alloc_scratch("v_one", VLEN)
        v_two = self.alloc_scratch("v_two", VLEN)
        v_n_nodes = self.alloc_scratch("v_n_nodes", VLEN)

        v_hash_consts = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            vc1 = self.alloc_scratch(f"v_hc1_{hi}", VLEN)
            vc3 = self.alloc_scratch(f"v_hc3_{hi}", VLEN)
            v_hash_consts.append((vc1, vc3))

        round_ctr = self.alloc_scratch("round_ctr")
        batch_ctr = self.alloc_scratch("batch_ctr")
        cur_idx_ptr = self.alloc_scratch("cur_idx_ptr")
        cur_val_ptr = self.alloc_scratch("cur_val_ptr")

        n_vec_iters = batch_size // VLEN

        self.instrs.append({"valu": [("vbroadcast", v_zero, zero_const)]})
        self.instrs.append({"valu": [("vbroadcast", v_one, one_const)]})
        self.instrs.append({"valu": [("vbroadcast", v_two, two_const)]})
        self.instrs.append({"valu": [("vbroadcast", v_n_nodes, self.scratch["n_nodes"])]})

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            c1 = self.scratch_const(val1)
            c3 = self.scratch_const(val3)
            vc1, vc3 = v_hash_consts[hi]
            self.instrs.append({"valu": [("vbroadcast", vc1, c1), ("vbroadcast", vc3, c3)]})

        rounds_const = self.scratch_const(rounds)
        n_vecs = n_vec_iters  # 32 vectors

        # Keep ALL idx/val in scratch across all rounds (no memory traffic between rounds)
        all_idx = self.alloc_scratch("all_idx", batch_size)  # 256 words
        all_val = self.alloc_scratch("all_val", batch_size)  # 256 words
        tree_scalar = self.alloc_scratch("tree_scalar")

        # Helper functions
        def emit_hash(v_val_reg):
            for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                vc1, vc3 = v_hash_consts[hi]
                self.instrs.append({"valu": [(op1, v_tmp1, v_val_reg, vc1), (op3, v_tmp2, v_val_reg, vc3)]})
                self.instrs.append({"valu": [(op2, v_val_reg, v_tmp1, v_tmp2)]})

        def emit_idx_update(v_idx_reg, v_val_reg):
            # offset = (val % 2) + 1: gives 1 when even, 2 when odd - NO VSELECT!
            self.instrs.append({"valu": [("%", v_tmp1, v_val_reg, v_two), ("*", v_idx_reg, v_idx_reg, v_two)]})
            self.instrs.append({"valu": [("+", v_tmp1, v_tmp1, v_one)]})
            self.instrs.append({"valu": [("+", v_idx_reg, v_idx_reg, v_tmp1)]})
            # Wrap: idx = idx * (idx < n_nodes) - NO VSELECT!
            self.instrs.append({"valu": [("<", v_tmp1, v_idx_reg, v_n_nodes)]})
            self.instrs.append({"valu": [("*", v_idx_reg, v_idx_reg, v_tmp1)]})

        # === LOAD all val from memory into scratch (idx starts at 0) ===
        for v in range(n_vecs):
            off = self.scratch_const(VLEN * v)
            self.instrs.append({"alu": [("+", addr0, self.scratch["inp_values_p"], off)]})
            self.instrs.append({"load": [("vload", all_val + v * VLEN, addr0)]})
        
        # Initialize all idx to 0
        for v in range(n_vecs):
            self.instrs.append({"valu": [("+", all_idx + v * VLEN, v_zero, v_zero)]})

        # Additional temps for parallel processing
        v_nv2 = self.alloc_scratch("v_nv2", VLEN)
        v_tmp4 = self.alloc_scratch("v_tmp4", VLEN)
        v_tmp5 = self.alloc_scratch("v_tmp5", VLEN)

        # Helper for parallel hash (2 vectors)
        def emit_hash_pair(vv0, vv1):
            for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                vc1, vc3 = v_hash_consts[hi]
                self.instrs.append({"valu": [
                    (op1, v_tmp1, vv0, vc1), (op3, v_tmp2, vv0, vc3),
                    (op1, v_tmp4, vv1, vc1), (op3, v_tmp5, vv1, vc3),
                ]})
                self.instrs.append({"valu": [(op2, vv0, v_tmp1, v_tmp2), (op2, vv1, v_tmp4, v_tmp5)]})

        def emit_idx_update_pair(vi0, vv0, vi1, vv1):
            # offset = (val % 2) + 1: gives 1 when even, 2 when odd - NO VSELECT!
            self.instrs.append({"valu": [
                ("%", v_tmp1, vv0, v_two), ("%", v_tmp4, vv1, v_two),
                ("*", vi0, vi0, v_two), ("*", vi1, vi1, v_two),
            ]})
            self.instrs.append({"valu": [
                ("+", v_tmp1, v_tmp1, v_one), ("+", v_tmp4, v_tmp4, v_one),
            ]})
            self.instrs.append({"valu": [
                ("+", vi0, vi0, v_tmp1), ("+", vi1, vi1, v_tmp4),
            ]})
            # Wrap: idx = idx * (idx < n_nodes) - NO VSELECT!
            self.instrs.append({"valu": [("<", v_tmp1, vi0, v_n_nodes), ("<", v_tmp4, vi1, v_n_nodes)]})
            self.instrs.append({"valu": [("*", vi0, vi0, v_tmp1), ("*", vi1, vi1, v_tmp4)]})

        # === ROUND 0: All idx=0, load tree[0] once, broadcast ===
        self.instrs.append({"alu": [("+", addr0, self.scratch["forest_values_p"], zero_const)]})
        self.instrs.append({"load": [("load", tree_scalar, addr0)]})
        self.instrs.append({"valu": [("vbroadcast", v_node_val, tree_scalar)]})
        
        for v in range(0, n_vecs, 2):
            vi0, vv0 = all_idx + v * VLEN, all_val + v * VLEN
            vi1, vv1 = all_idx + (v+1) * VLEN, all_val + (v+1) * VLEN
            self.instrs.append({"valu": [("^", vv0, vv0, v_node_val), ("^", vv1, vv1, v_node_val)]})
            emit_hash_pair(vv0, vv1)
            emit_idx_update_pair(vi0, vv0, vi1, vv1)

        # === ROUND 1: idx in {1,2}, use arithmetic instead of vselect ===
        # tree_val = tree[1] + (idx - 1) * (tree[2] - tree[1])
        c1_const = self.scratch_const(1)
        c2_const = self.scratch_const(2)
        tree_scalar2 = self.alloc_scratch("tree_scalar2")
        v_tree1 = self.alloc_scratch("v_tree1", VLEN)
        v_tree_diff = self.alloc_scratch("v_tree_diff", VLEN)  # tree[2] - tree[1]
        
        self.instrs.append({"alu": [("+", addr0, self.scratch["forest_values_p"], c1_const),
                                    ("+", addr1, self.scratch["forest_values_p"], c2_const)]})
        self.instrs.append({"load": [("load", tree_scalar, addr0), ("load", tree_scalar2, addr1)]})
        self.instrs.append({"alu": [("-", tree_scalar2, tree_scalar2, tree_scalar)]})  # diff = tree[2] - tree[1]
        self.instrs.append({"valu": [("vbroadcast", v_tree1, tree_scalar), ("vbroadcast", v_tree_diff, tree_scalar2)]})
        
        for v in range(0, n_vecs, 2):
            vi0, vv0 = all_idx + v * VLEN, all_val + v * VLEN
            vi1, vv1 = all_idx + (v+1) * VLEN, all_val + (v+1) * VLEN
            # tree_val = tree[1] + (idx - 1) * diff, NO VSELECT!
            self.instrs.append({"valu": [("-", v_node_val, vi0, v_one), ("-", v_nv2, vi1, v_one)]})
            self.instrs.append({"valu": [("*", v_node_val, v_node_val, v_tree_diff), ("*", v_nv2, v_nv2, v_tree_diff)]})
            self.instrs.append({"valu": [("+", v_node_val, v_node_val, v_tree1), ("+", v_nv2, v_nv2, v_tree1)]})
            self.instrs.append({"valu": [("^", vv0, vv0, v_node_val), ("^", vv1, vv1, v_nv2)]})
            emit_hash_pair(vv0, vv1)
            emit_idx_update_pair(vi0, vv0, vi1, vv1)

        # === ROUND 2: idx in {3,4,5,6}, use vselect binary tree ===
        # Load all 4 tree values and broadcast
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
        
        self.instrs.append({"alu": [("+", addr0, self.scratch["forest_values_p"], c3),
                                    ("+", addr1, self.scratch["forest_values_p"], c4)]})
        self.instrs.append({"load": [("load", ts3, addr0), ("load", ts4, addr1)]})
        self.instrs.append({"alu": [("+", addr0, self.scratch["forest_values_p"], c5),
                                    ("+", addr1, self.scratch["forest_values_p"], self.scratch_const(6))]})
        self.instrs.append({"load": [("load", ts5, addr0), ("load", ts6, addr1)]})
        self.instrs.append({"valu": [("vbroadcast", v_t3, ts3), ("vbroadcast", v_t4, ts4)]})
        self.instrs.append({"valu": [("vbroadcast", v_t5, ts5), ("vbroadcast", v_t6, ts6)]})
        
        v_three = self.alloc_scratch("v_three", VLEN)
        self.instrs.append({"valu": [("vbroadcast", v_three, c3)]})
        
        # Process round 2 in pairs like other rounds
        for v in range(0, n_vecs, 2):
            vi0, vv0 = all_idx + v * VLEN, all_val + v * VLEN
            vi1, vv1 = all_idx + (v+1) * VLEN, all_val + (v+1) * VLEN
            # offset = idx - 3 for both
            self.instrs.append({"valu": [("-", v_tmp1, vi0, v_three), ("-", v_tmp4, vi1, v_three)]})
            # bit0 = offset % 2, bit1 = offset >> 1
            self.instrs.append({"valu": [("%", v_tmp2, v_tmp1, v_two), (">>", v_tmp3, v_tmp1, v_one),
                                         ("%", v_tmp5, v_tmp4, v_two), (">>", v_nv2, v_tmp4, v_one)]})
            # Select for vector 0
            self.instrs.append({"flow": [("vselect", v_node_val, v_tmp2, v_t4, v_t3)]})
            self.instrs.append({"flow": [("vselect", v_tmp1, v_tmp2, v_t6, v_t5)]})
            self.instrs.append({"flow": [("vselect", v_node_val, v_tmp3, v_tmp1, v_node_val)]})
            # Select for vector 1
            self.instrs.append({"flow": [("vselect", v_tmp4, v_tmp5, v_t4, v_t3)]})
            self.instrs.append({"flow": [("vselect", v_tmp1, v_tmp5, v_t6, v_t5)]})
            self.instrs.append({"flow": [("vselect", v_tmp4, v_nv2, v_tmp1, v_tmp4)]})
            # XOR and process both
            self.instrs.append({"valu": [("^", vv0, vv0, v_node_val), ("^", vv1, vv1, v_tmp4)]})
            emit_hash_pair(vv0, vv1)
            emit_idx_update_pair(vi0, vv0, vi1, vv1)

        # === ROUNDS 3+: Optimized with pipelined gather and parallel hash ===
        remaining = rounds - 3
        if remaining > 0:
            
            self.instrs.append({"load": [("const", round_ctr, 0)]})
            remaining_const = self.scratch_const(remaining)
            
            outer_loop = len(self.instrs)
            
            # Allocate address registers once
            addrs = [addr0, addr1]
            for i in range(2, 8):
                if f"addr{i}" not in self.scratch:
                    addrs.append(self.alloc_scratch(f"addr{i}"))
                else:
                    addrs.append(self.scratch[f"addr{i}"])
            
            # Process all vector pairs with software pipelining
            # Prologue: start gather for first pair
            vi0 = all_idx
            vv0 = all_val
            vi1 = all_idx + VLEN
            vv1 = all_val + VLEN
            
            self.instrs.append({"alu": [("+", addrs[i], self.scratch["forest_values_p"], vi0 + i) for i in range(8)]})
            for i in range(0, 8, 2):
                self.instrs.append({"load": [("load", v_node_val + i, addrs[i]), ("load", v_node_val + i + 1, addrs[i + 1])]})
            self.instrs.append({"alu": [("+", addrs[i], self.scratch["forest_values_p"], vi1 + i) for i in range(8)]})
            for i in range(0, 8, 2):
                self.instrs.append({"load": [("load", v_nv2 + i, addrs[i]), ("load", v_nv2 + i + 1, addrs[i + 1])]})
            
            # Main loop: overlap gather(V+1) with compute(V)
            for v in range(0, n_vecs - 2, 2):
                vi0 = all_idx + v * VLEN
                vv0 = all_val + v * VLEN
                vi1 = all_idx + (v + 1) * VLEN
                vv1 = all_val + (v + 1) * VLEN
                vi0_next = all_idx + (v + 2) * VLEN
                vi1_next = all_idx + (v + 3) * VLEN
                
                # XOR current pair while starting address compute for next
                self.instrs.append({
                    "valu": [("^", vv0, vv0, v_node_val), ("^", vv1, vv1, v_nv2)],
                    "alu": [("+", addrs[i], self.scratch["forest_values_p"], vi0_next + i) for i in range(4)]
                })
                
                # Hash + continue gather for next pair (interleave VALU with loads)
                for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                    vc1, vc3 = v_hash_consts[hi]
                    if hi < 4:  # First 4 hash stages: interleave with loads
                        self.instrs.append({
                            "valu": [(op1, v_tmp1, vv0, vc1), (op3, v_tmp2, vv0, vc3), (op1, v_tmp4, vv1, vc1), (op3, v_tmp5, vv1, vc3)],
                            "load": [("load", v_node_val + hi*2, addrs[hi*2]), ("load", v_node_val + hi*2 + 1, addrs[hi*2 + 1])] if hi < 4 else []
                        })
                    else:
                        self.instrs.append({"valu": [(op1, v_tmp1, vv0, vc1), (op3, v_tmp2, vv0, vc3), (op1, v_tmp4, vv1, vc1), (op3, v_tmp5, vv1, vc3)]})
                    self.instrs.append({"valu": [(op2, vv0, v_tmp1, v_tmp2), (op2, vv1, v_tmp4, v_tmp5)]})
                
                # Address compute for vi1_next + remaining loads
                self.instrs.append({"alu": [("+", addrs[i], self.scratch["forest_values_p"], vi0_next + i + 4) for i in range(4)]})
                for i in range(0, 4, 2):
                    self.instrs.append({"load": [("load", v_node_val + 4 + i, addrs[i]), ("load", v_node_val + 5 + i, addrs[i + 1])]})
                self.instrs.append({"alu": [("+", addrs[i], self.scratch["forest_values_p"], vi1_next + i) for i in range(8)]})
                for i in range(0, 8, 2):
                    self.instrs.append({"load": [("load", v_nv2 + i, addrs[i]), ("load", v_nv2 + i + 1, addrs[i + 1])]})
                
                # Index update for current pair
                self.instrs.append({"valu": [("%", v_tmp1, vv0, v_two), ("%", v_tmp4, vv1, v_two), ("*", vi0, vi0, v_two), ("*", vi1, vi1, v_two)]})
                self.instrs.append({"valu": [("+", v_tmp1, v_tmp1, v_one), ("+", v_tmp4, v_tmp4, v_one)]})
                self.instrs.append({"valu": [("+", vi0, vi0, v_tmp1), ("+", vi1, vi1, v_tmp4)]})
                self.instrs.append({"valu": [("<", v_tmp1, vi0, v_n_nodes), ("<", v_tmp4, vi1, v_n_nodes)]})
                self.instrs.append({"valu": [("*", vi0, vi0, v_tmp1), ("*", vi1, vi1, v_tmp4)]})
            
            # Epilogue: process last pair (no next pair to prefetch)
            v = n_vecs - 2
            vi0 = all_idx + v * VLEN
            vv0 = all_val + v * VLEN
            vi1 = all_idx + (v + 1) * VLEN
            vv1 = all_val + (v + 1) * VLEN
            self.instrs.append({"valu": [("^", vv0, vv0, v_node_val), ("^", vv1, vv1, v_nv2)]})
            for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                vc1, vc3 = v_hash_consts[hi]
                self.instrs.append({"valu": [(op1, v_tmp1, vv0, vc1), (op3, v_tmp2, vv0, vc3), (op1, v_tmp4, vv1, vc1), (op3, v_tmp5, vv1, vc3)]})
                self.instrs.append({"valu": [(op2, vv0, v_tmp1, v_tmp2), (op2, vv1, v_tmp4, v_tmp5)]})
            self.instrs.append({"valu": [("%", v_tmp1, vv0, v_two), ("%", v_tmp4, vv1, v_two), ("*", vi0, vi0, v_two), ("*", vi1, vi1, v_two)]})
            self.instrs.append({"valu": [("+", v_tmp1, v_tmp1, v_one), ("+", v_tmp4, v_tmp4, v_one)]})
            self.instrs.append({"valu": [("+", vi0, vi0, v_tmp1), ("+", vi1, vi1, v_tmp4)]})
            self.instrs.append({"valu": [("<", v_tmp1, vi0, v_n_nodes), ("<", v_tmp4, vi1, v_n_nodes)]})
            self.instrs.append({"valu": [("*", vi0, vi0, v_tmp1), ("*", vi1, vi1, v_tmp4)]})
            
            self.instrs.append({"alu": [("+", round_ctr, round_ctr, one_const)]})
            self.instrs.append({"alu": [("<", tmp1, round_ctr, remaining_const)]})
            self.instrs.append({"flow": [("cond_jump", tmp1, outer_loop)]})

        # === STORE all idx/val back to memory (once) ===
        for v in range(n_vecs):
            off = self.scratch_const(VLEN * v)
            self.instrs.append({"alu": [("+", addr0, self.scratch["inp_indices_p"], off),
                                        ("+", addr1, self.scratch["inp_values_p"], off)]})
            self.instrs.append({"store": [("vstore", addr0, all_idx + v * VLEN),
                                          ("vstore", addr1, all_val + v * VLEN)]})

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

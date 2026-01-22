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
        Like reference_kernel2 but building actual instructions.
        Scalar implementation using only scalar ALU and load/store.
        """
        def add_bundle(alu=None, valu=None, load=None, store=None, flow=None):
            instr = {}
            if alu:
                instr["alu"] = alu
            if valu:
                instr["valu"] = valu
            if load:
                instr["load"] = load
            if store:
                instr["store"] = store
            if flow:
                instr["flow"] = flow
            if instr:
                self.instrs.append(instr)

        def alloc_vec(name):
            return self.alloc_scratch(name, VLEN)

        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")
        tmp_addr_idx = self.alloc_scratch("tmp_addr_idx")
        tmp_addr_val = self.alloc_scratch("tmp_addr_val")
        # Scratch space addresses
        init_vars = [
            "rounds",
            "n_nodes",
            "batch_size",
            "forest_height",
            "forest_values_p",
            "inp_indices_p",
            "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp1, i))
            self.add("load", ("load", self.scratch[v], tmp1))

        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)

        # Vector scratch registers
        idx_v = alloc_vec("idx_v")
        val_v = alloc_vec("val_v")
        node_v = alloc_vec("node_v")
        addr_node_v = alloc_vec("addr_node_v")
        tmp1_v = alloc_vec("tmp1_v")
        tmp2_v = alloc_vec("tmp2_v")
        cond_v = alloc_vec("cond_v")
        one_v = alloc_vec("one_v")
        two_v = alloc_vec("two_v")
        forest_base_v = alloc_vec("forest_base_v")
        n_nodes_v = alloc_vec("n_nodes_v")

        for bundle in (
            [("vbroadcast", one_v, one_const)],
            [("vbroadcast", two_v, two_const), ("vbroadcast", forest_base_v, self.scratch["forest_values_p"])],
            [("vbroadcast", n_nodes_v, self.scratch["n_nodes"])]
        ):
            add_bundle(valu=bundle)

        hash_const_vecs = []
        for hi, (_, val1, _, _, val3) in enumerate(HASH_STAGES):
            val1_const = self.scratch_const(val1, name=f"hash_val1_{hi}")
            val3_const = self.scratch_const(val3, name=f"hash_val3_{hi}")
            val1_vec = alloc_vec(f"hash_val1_{hi}_v")
            val3_vec = alloc_vec(f"hash_val3_{hi}_v")
            add_bundle(valu=[("vbroadcast", val1_vec, val1_const), ("vbroadcast", val3_vec, val3_const)])
            hash_const_vecs.append((val1_vec, val3_vec))

        # Pause instructions are matched up with yield statements in the reference
        # kernel to let you debug at intermediate steps. The testing harness in this
        # file requires these match up to the reference kernel's yields, but the
        # submission harness ignores them.
        self.add("flow", ("pause",))

        for round in range(rounds):
            for i in range(0, batch_size, VLEN):
                i_const = self.scratch_const(i)
                add_bundle(
                    alu=[
                        ("+", tmp_addr_idx, self.scratch["inp_indices_p"], i_const),
                        ("+", tmp_addr_val, self.scratch["inp_values_p"], i_const),
                    ]
                )
                add_bundle(
                    load=[
                        ("vload", idx_v, tmp_addr_idx),
                        ("vload", val_v, tmp_addr_val),
                    ]
                )
                add_bundle(valu=[("+", addr_node_v, forest_base_v, idx_v)])
                for offset in range(0, VLEN, 2):
                    add_bundle(
                        load=[
                            ("load_offset", node_v, addr_node_v, offset),
                            ("load_offset", node_v, addr_node_v, offset + 1),
                        ]
                    )
                add_bundle(valu=[("^", val_v, val_v, node_v)])

                for stage_idx, (op1, _, op2, op3, _) in enumerate(HASH_STAGES):
                    val1_vec, val3_vec = hash_const_vecs[stage_idx]
                    add_bundle(
                        valu=[
                            (op1, tmp1_v, val_v, val1_vec),
                            (op3, tmp2_v, val_v, val3_vec),
                        ]
                    )
                    add_bundle(valu=[(op2, val_v, tmp1_v, tmp2_v)])

                add_bundle(valu=[("&", tmp1_v, val_v, one_v), ("*", idx_v, idx_v, two_v)])
                add_bundle(valu=[("+", tmp1_v, tmp1_v, one_v)])
                add_bundle(valu=[("+", idx_v, idx_v, tmp1_v)])
                add_bundle(valu=[("<", cond_v, idx_v, n_nodes_v)])
                add_bundle(valu=[("*", idx_v, idx_v, cond_v)])

                add_bundle(store=[("vstore", tmp_addr_idx, idx_v), ("vstore", tmp_addr_val, val_v)])

        # Required to match with the yield in reference_kernel2
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

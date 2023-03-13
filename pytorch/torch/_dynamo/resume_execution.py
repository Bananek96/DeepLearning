import copy
import dataclasses
import sys
import types
from typing import Any, Dict, List, Optional, Tuple

from .bytecode_transformation import (
    create_call_function,
    create_instruction,
    create_jump_absolute,
    Instruction,
    transform_code_object,
    unique_id,
)
from .codegen import PyCodegen
from .utils import ExactWeakKeyDictionary

# taken from code.h in cpython
CO_OPTIMIZED = 0x0001
CO_NEWLOCALS = 0x0002
CO_VARARGS = 0x0004
CO_VARKEYWORDS = 0x0008
CO_NESTED = 0x0010
CO_GENERATOR = 0x0020
CO_NOFREE = 0x0040
CO_COROUTINE = 0x0080
CO_ITERABLE_COROUTINE = 0x0100
CO_ASYNC_GENERATOR = 0x0200


@dataclasses.dataclass(frozen=True)
class ReenterWith:
    stack_index: int = None
    target_values: Optional[Tuple] = None

    # If we do not want to destroy the stack, we can do the same thing as a
    # `SETUP_WITH` block, only that we store the context manager in a local_symbol
    def try_except(self, code_options, cleanup: List[Instruction]):
        load_args = []
        if self.target_values:
            load_args = [
                create_instruction(
                    "LOAD_CONST",
                    PyCodegen.get_const_index(code_options, val),
                    val,
                )
                for val in self.target_values
            ]
        ctx_name = unique_id(f"___context_manager_{self.stack_index}")
        if ctx_name not in code_options["co_varnames"]:
            code_options["co_varnames"] += (ctx_name,)
        for name in ["__enter__", "__exit__"]:
            if name not in code_options["co_names"]:
                code_options["co_names"] += (name,)

        except_jump_target = create_instruction("NOP")
        cleanup_complete_jump_target = create_instruction("NOP")

        setup_finally = [
            *load_args,
            create_instruction("CALL_FUNCTION", len(load_args)),
            create_instruction(
                "STORE_FAST", code_options["co_varnames"].index(ctx_name), ctx_name
            ),
            create_instruction(
                "LOAD_FAST", code_options["co_varnames"].index(ctx_name), ctx_name
            ),
            create_instruction("LOAD_METHOD", "__enter__"),
            create_instruction("CALL_METHOD", 0),
            create_instruction("POP_TOP"),
            create_instruction("SETUP_FINALLY", target=except_jump_target),
        ]

        reset = [
            create_instruction(
                "LOAD_FAST", code_options["co_varnames"].index(ctx_name), ctx_name
            ),
            create_instruction("LOAD_METHOD", "__exit__"),
            create_instruction(
                "LOAD_CONST", PyCodegen.get_const_index(code_options, None), None
            ),
            create_instruction("DUP_TOP"),
            create_instruction("DUP_TOP"),
            create_instruction("CALL_METHOD", 3),
            create_instruction("POP_TOP"),
        ]
        if sys.version_info < (3, 9):
            epilogue = [
                create_instruction("POP_BLOCK"),
                create_instruction("BEGIN_FINALLY"),
                except_jump_target,
                *reset,
                create_instruction("END_FINALLY"),
            ]
        else:
            epilogue = [
                create_instruction("POP_BLOCK"),
                *reset,
                create_instruction("JUMP_FORWARD", target=cleanup_complete_jump_target),
                except_jump_target,
                *reset,
                create_instruction("RERAISE"),
                cleanup_complete_jump_target,
            ]

        cleanup[:] = epilogue + cleanup
        return setup_finally

    def __call__(self, code_options, cleanup):
        load_args = []
        if self.target_values:
            load_args = [
                create_instruction(
                    "LOAD_CONST",
                    PyCodegen.get_const_index(code_options, val),
                    val,
                )
                for val in self.target_values
            ]
        if sys.version_info < (3, 9):
            with_cleanup_start = create_instruction("WITH_CLEANUP_START")
            begin_finally = create_instruction("BEGIN_FINALLY")
            cleanup[:] = [
                create_instruction("POP_BLOCK"),
                begin_finally,
                with_cleanup_start,
                create_instruction("WITH_CLEANUP_FINISH"),
                create_instruction("END_FINALLY"),
            ] + cleanup

            return [
                *load_args,
                create_instruction("CALL_FUNCTION", len(load_args)),
                create_instruction("SETUP_WITH", target=with_cleanup_start),
                create_instruction("POP_TOP"),
            ]
        elif sys.version_info < (3, 11):
            with_except_start = create_instruction("WITH_EXCEPT_START")
            pop_top_after_with_except_start = create_instruction("POP_TOP")

            cleanup_complete_jump_target = create_instruction("NOP")

            cleanup[:] = [
                create_instruction("POP_BLOCK"),
                create_instruction(
                    "LOAD_CONST", PyCodegen.get_const_index(code_options, None), None
                ),
                create_instruction("DUP_TOP"),
                create_instruction("DUP_TOP"),
                create_instruction("CALL_FUNCTION", 3),
                create_instruction("POP_TOP"),
                create_instruction("JUMP_FORWARD", target=cleanup_complete_jump_target),
                with_except_start,
                create_instruction(
                    "POP_JUMP_IF_TRUE", target=pop_top_after_with_except_start
                ),
                create_instruction("RERAISE"),
                pop_top_after_with_except_start,
                create_instruction("POP_TOP"),
                create_instruction("POP_TOP"),
                create_instruction("POP_EXCEPT"),
                create_instruction("POP_TOP"),
                cleanup_complete_jump_target,
            ] + cleanup

            return [
                *load_args,
                create_instruction("CALL_FUNCTION", len(load_args)),
                create_instruction("SETUP_WITH", target=with_except_start),
                create_instruction("POP_TOP"),
            ]
        else:
            pop_top_after_with_except_start = create_instruction("POP_TOP")
            cleanup_complete_jump_target = create_instruction("NOP")

            def create_load_none():
                return create_instruction(
                    "LOAD_CONST", PyCodegen.get_const_index(code_options, None), None
                )

            cleanup[:] = (
                [
                    create_load_none(),
                    create_load_none(),
                    create_load_none(),
                ]
                + create_call_function(2, False)
                + [
                    create_instruction("POP_TOP"),
                    create_instruction(
                        "JUMP_FORWARD", target=cleanup_complete_jump_target
                    ),
                    create_instruction("PUSH_EXC_INFO"),
                    create_instruction("WITH_EXCEPT_START"),
                    create_instruction(
                        "POP_JUMP_FORWARD_IF_TRUE",
                        target=pop_top_after_with_except_start,
                    ),
                    create_instruction("RERAISE", 2),
                    create_instruction("COPY", 3),
                    create_instruction("POP_EXCEPT"),
                    create_instruction("RERAISE", 1),
                    pop_top_after_with_except_start,
                    create_instruction("POP_EXCEPT"),
                    create_instruction("POP_TOP"),
                    create_instruction("POP_TOP"),
                    cleanup_complete_jump_target,
                ]
                + cleanup
            )

            return create_call_function(0, False) + [
                create_instruction("BEFORE_WITH"),
                create_instruction("POP_TOP"),
            ]


@dataclasses.dataclass
class ResumeFunctionMetadata:
    code: types.CodeType
    instructions: List[Instruction] = None


class ContinueExecutionCache:
    cache = ExactWeakKeyDictionary()
    generated_code_metadata = ExactWeakKeyDictionary()

    @classmethod
    def lookup(cls, code, lineno, *key):
        if code not in cls.cache:
            cls.cache[code] = dict()
        key = tuple(key)
        if key not in cls.cache[code]:
            cls.cache[code][key] = cls.generate(code, lineno, *key)
        return cls.cache[code][key]

    @classmethod
    def generate(
        cls,
        code,
        lineno,
        offset: int,
        nstack: int,
        argnames: List[str],
        setup_fns: List[ReenterWith],
        null_idxes: List[int],
    ):
        assert offset is not None
        assert not (
            code.co_flags
            & (CO_GENERATOR | CO_COROUTINE | CO_ITERABLE_COROUTINE | CO_ASYNC_GENERATOR)
        )
        assert code.co_flags & CO_OPTIMIZED
        if code in ContinueExecutionCache.generated_code_metadata:
            return cls.generate_based_on_original_code_object(
                code, lineno, offset, nstack, argnames, setup_fns, null_idxes
            )

        meta = ResumeFunctionMetadata(code)

        def update(instructions: List[Instruction], code_options: Dict[str, Any]):
            meta.instructions = copy.deepcopy(instructions)

            args = [f"___stack{i}" for i in range(nstack)]
            args.extend(v for v in argnames if v not in args)
            freevars = tuple(code_options["co_cellvars"] or []) + tuple(
                code_options["co_freevars"] or []
            )
            code_options["co_name"] = f"<resume in {code_options['co_name']}>"
            if sys.version_info >= (3, 11):
                code_options[
                    "co_qualname"
                ] = f"<graph break in {code_options['co_qualname']}>"
            code_options["co_firstlineno"] = lineno
            code_options["co_cellvars"] = tuple()
            code_options["co_freevars"] = freevars
            code_options["co_argcount"] = len(args)
            code_options["co_posonlyargcount"] = 0
            code_options["co_kwonlyargcount"] = 0
            code_options["co_varnames"] = tuple(
                args + [v for v in code_options["co_varnames"] if v not in args]
            )
            code_options["co_flags"] = code_options["co_flags"] & ~(
                CO_VARARGS | CO_VARKEYWORDS
            )
            # TODO probably need to update co_exceptiontable for python 3.11
            (target,) = [i for i in instructions if i.offset == offset]

            prefix = []
            cleanup = []
            hooks = {fn.stack_index: fn for fn in setup_fns}
            for i in range(nstack):
                prefix.append(create_instruction("LOAD_FAST", f"___stack{i}"))
                if i in hooks:
                    prefix.extend(hooks.pop(i)(code_options, cleanup))
            assert not hooks

            if sys.version_info >= (3, 11):
                for idx in null_idxes:
                    prefix.append(create_instruction("PUSH_NULL"))
                    prefix.extend(create_rot_n(idx))

            prefix.append(create_jump_absolute(target))

            # because the line number table monotonically increases from co_firstlineno
            # remove starts_line for any instructions before the graph break instruction
            # this will ensure the instructions after the break have the correct line numbers
            target_ind = int(target.offset / 2)
            for inst in instructions[0:target_ind]:
                inst.starts_line = None

            if cleanup:
                prefix.extend(cleanup)
                prefix.extend(cls.unreachable_codes(code_options))

            # TODO(jansel): add dead code elimination here
            instructions[:] = prefix + instructions

        new_code = transform_code_object(code, update)
        ContinueExecutionCache.generated_code_metadata[new_code] = meta
        return new_code

    @staticmethod
    def unreachable_codes(code_options):
        """Codegen a `raise None` to make analysis work for unreachable code"""
        return [
            create_instruction(
                "LOAD_CONST",
                argval=None,
                arg=PyCodegen.get_const_index(code_options, None),
            ),
            create_instruction("RAISE_VARARGS", 1),
        ]

    @classmethod
    def generate_based_on_original_code_object(cls, code, lineno, offset: int, *args):
        """
        This handles the case of generating a resume into code generated
        to resume something else.  We want to always generate starting
        from the original code object so that if control flow paths
        converge we only generated 1 resume function (rather than 2^n
        resume functions).
        """

        meta: ResumeFunctionMetadata = ContinueExecutionCache.generated_code_metadata[
            code
        ]
        new_offset = None

        def find_new_offset(
            instructions: List[Instruction], code_options: Dict[str, Any]
        ):
            nonlocal new_offset
            (target,) = [i for i in instructions if i.offset == offset]
            # match the functions starting at the last instruction as we have added a prefix
            (new_target,) = [
                i2
                for i1, i2 in zip(reversed(instructions), reversed(meta.instructions))
                if i1 is target
            ]
            assert target.opcode == new_target.opcode
            new_offset = new_target.offset

        transform_code_object(code, find_new_offset)
        return ContinueExecutionCache.lookup(meta.code, lineno, new_offset, *args)


"""
# partially finished support for with statements

def convert_locals_to_cells(
        instructions: List[Instruction],
        code_options: Dict[str, Any]):

    code_options["co_cellvars"] = tuple(
        var
        for var in code_options["co_varnames"]
        if var not in code_options["co_freevars"]
        and not var.startswith("___stack")
    )
    cell_and_free = code_options["co_cellvars"] + code_options["co_freevars"]
    for inst in instructions:
        if str(inst.argval).startswith("___stack"):
            continue
        elif inst.opname == "LOAD_FAST":
            inst.opname = "LOAD_DEREF"
        elif inst.opname == "STORE_FAST":
            inst.opname = "STORE_DEREF"
        elif inst.opname == "DELETE_FAST":
            inst.opname = "DELETE_DEREF"
        else:
            continue
        inst.opcode = dis.opmap[inst.opname]
        assert inst.argval in cell_and_free, inst.argval
        inst.arg = cell_and_free.index(inst.argval)

def patch_setup_with(
    instructions: List[Instruction],
    code_options: Dict[str, Any]
):
    nonlocal need_skip
    need_skip = True
    target_index = [
        idx for idx, i in enumerate(instructions) if i.offset == offset
    ][0]
    assert instructions[target_index].opname == "SETUP_WITH"
    convert_locals_to_cells(instructions, code_options)

    stack_depth_before = nstack + stack_effect(instructions[target_index].opcode,
                                               instructions[target_index].arg)

    inside_with = []
    inside_with_resume_at = None
    stack_depth = stack_depth_before
    idx = target_index + 1
    for idx in range(idx, len(instructions)):
        inst = instructions[idx]
        if inst.opname == "BEGIN_FINALLY":
            inside_with_resume_at = inst
            break
        elif inst.target is not None:
            unimplemented("jump from with not supported")
        elif inst.opname in ("BEGIN_FINALLY", "WITH_CLEANUP_START", "WITH_CLEANUP_FINISH", "END_FINALLY",
                             "POP_FINALLY", "POP_EXCEPT",
                             "POP_BLOCK", "END_ASYNC_FOR"):
            unimplemented("block ops not supported")
        inside_with.append(inst)
        stack_depth += stack_effect(inst.opcode, inst.arg)
    assert inside_with_resume_at

    instructions = [
        create_instruction("LOAD_FAST", f"___stack{i}") for i in range(nstack)
    ] + [
        create_instruction("SETUP_WITH", target=instructions[target_index].target)
        ... call the function ...
        unpack_tuple
    ] + [
        create_instruction("JUMP_ABSOLUTE", target=inside_with_resume_at)
    ]
"""
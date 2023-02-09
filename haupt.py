from dataclasses import dataclass
from enum import Enum, auto
from typing import List
import subprocess
import sys
import os

import colorama
from colorama import Fore, Style

colorama.init()


class Types(Enum):
    INT = auto()
    PTR = auto()
    STR = auto()


@dataclass
class Signature:
    inputs: List[Types]
    outputs: List[Types]


class OpSet(Enum):
    NOP = auto()
    DUP = auto()
    DROP = auto()
    ROT = auto()
    SWAP = auto()
    OVER = auto()
    ADD_INT = auto()
    SUB_INT = auto()
    MUL_INT = auto()
    DIV_INT = auto()
    MOD_INT = auto()
    ADD_PTR = auto()
    SUB_PTR = auto()
    PUSH_INT = auto()
    PUSH_STR = auto()
    PUSH_PTR = auto()
    PRINT_INT = auto()
    PRINT_STR = auto()
    LT = auto()
    GT = auto()
    EQ = auto()
    NEQ = auto()
    SET_INT = auto()
    GET_INT = auto()


class Keywords(Enum):
    IF = auto()
    ELSE = auto()
    DO = auto()
    END = auto()
    WHILE = auto()


COMMENT_CHAR = "#"


def get_instruction_location(instruction):
    return (f"{instruction[0]}:"
            f"{Fore.CYAN}{instruction[1]}:{instruction[2]}{Style.RESET_ALL}")


class ErrorTypes(Enum):
    NORMAL = auto()
    STACK = auto()


def print_compiler_error(err, info="", error_type=ErrorTypes.NORMAL):
    match error_type:
        case ErrorTypes.NORMAL:
            if len(info) == 0:
                print(f"{Fore.LIGHTRED_EX}Error: {err}{Style.RESET_ALL}")
            else:
                print(f"{Fore.LIGHTRED_EX}Error: {err}{Style.RESET_ALL}\n{info}")
        case ErrorTypes.STACK:
            if len(info) == 0:
                print(f"{Fore.LIGHTRED_EX}StackError: {err}{Style.RESET_ALL}")
            else:
                print(f"{Fore.LIGHTRED_EX}StackError: {err}{Style.RESET_ALL}\n{info}")

    exit(1)


def call_cmd(cmd: List, silenced=False):
    if not silenced:
        print("[CMD] " + " ".join(cmd))
    subprocess.call(cmd)


def get_lines(lines: List):
    return [(row, line) for (row, line) in enumerate(lines)]


def get_token_in_line(line: str):
    line = line.removesuffix("\n")
    if len(line) == 0:
        return []
    result = []
    parsed_token = []
    buffer = ""
    inside_str = False
    index = 0
    while index < len(line):
        char = line[index]
        if char == "\"":
            if not inside_str:
                assert len(buffer) == 0
                inside_str = True
                buffer += char
            else:
                inside_str = False
                buffer += char
                parsed_token.append(buffer)
                buffer = ""
        else:
            if inside_str:
                buffer += char
            elif char == " ":
                if len(buffer) > 0:
                    parsed_token.append(buffer)
                    buffer = ""
            else:
                buffer += char
        index += 1
    parsed_token.append(buffer)
    for col, token in enumerate(parsed_token):
        result.append((col, token))
    return result


def load_from_file(file_path: str):
    with open(file_path, "r") as f:
        return [(file_path, row + 1, col + 1, token)
                for (row, line) in get_lines(f.readlines())
                for (col, token) in get_token_in_line(line.split(COMMENT_CHAR)[0])]


def parse_op(op: str, memory: List):
    other_info = op
    op = op[3]
    assert len(OpSet) == 24, "Not all OP can be parsed yet"
    if (op.startswith("-") and op[1:].isdigit()) or op.isdigit():
        op = OpSet.PUSH_INT, (Types.INT, int(op))
    elif op.startswith("\"") and op.endswith("\""):
        op = OpSet.PUSH_STR, (Types.STR, op[1:-1])
    elif op == "rot":
        op = OpSet.ROT,
    elif op == "dup":
        op = OpSet.DUP,
    elif op == "swap":
        op = OpSet.SWAP,
    elif op == "over":
        op = OpSet.OVER,
    elif op == "drop":
        op = OpSet.DROP,
    elif op == "+":
        op = OpSet.ADD_INT,
    elif op == "-":
        op = OpSet.SUB_INT,
    elif op == "ptr+":
        op = OpSet.ADD_PTR,
    elif op == "ptr-":
        op = OpSet.SUB_PTR,
    elif op == "*":
        op = OpSet.MUL_INT,
    elif op == "/":
        op = OpSet.DIV_INT,
    elif op == '%':
        op = OpSet.MOD_INT,
    elif op == "puti":
        op = OpSet.PRINT_INT,
    elif op == "puts":
        op = OpSet.PRINT_STR,
    elif op == "==":
        op = OpSet.EQ,
    elif op == "!=":
        op = OpSet.NEQ,
    elif op == '>':
        op = OpSet.GT,
    elif op == '<':
        op = OpSet.LT,
    elif op == "!64":
        op = OpSet.SET_INT,
    elif op == "?64":
        op = OpSet.GET_INT,
    elif op == "mem":
        print_compiler_error("Unexpected `mem` found",
                             f"{get_instruction_location(other_info)}"
                             f" => `{op}`\n"
                             f"Memory block needs to be declared at the very top of the program.")
    else:
        if op == 'if':
            op = Keywords.IF,
        elif op == "do":
            op = Keywords.DO,
        elif op == "end":
            op = Keywords.END,
        elif op == "else":
            op = Keywords.ELSE,
        elif op == "while":
            op = Keywords.WHILE,
        else:
            variable = False
            for mem in memory:
                if op == mem[3]:
                    variable = True
                    break
            if variable:
                pass
                op = OpSet.PUSH_PTR, (Types.PTR, op)
            else:
                print_compiler_error(f"Unknown token in parse_op(op)!",
                                     f"{get_instruction_location(other_info)}"
                                     f" => `{op}`")
    return other_info[0], other_info[1], other_info[2], op


def parse_instructions(code: List):
    memory_found = False
    for op in code:
        if memory_found and op[3] == "mem":
            print_compiler_error("Multiple memory blocks found",
                                 "Right now only one memory block is supported\n")
        elif op[3] == "mem":
            memory_found = True

    memory = []
    if memory_found:
        memory_index = 0
        op = code[memory_index][3]
        if op == "mem":
            while code[memory_index][3] != "end":
                if memory_index == len(code) - 1:
                    print_compiler_error("Matching `end` not found!",
                                         f"{get_instruction_location(code[memory_index])}: "
                                         "`mem` block does not have matching `end`.")
                memory.append(code[memory_index])
                memory_index += 1
            memory.append(code[memory_index])

    new_code = []
    for op in code:
        if op not in memory:
            new_code.append(op)
    instructions = [parse_op(op, memory) for op in new_code]

    strings = []
    index = 0
    string_counter = 0
    while index < len(instructions):
        op = instructions[index]
        operation = op[3]
        if operation[0] == OpSet.PUSH_STR:
            typed_operand = operation[1]
            string = typed_operand[1]
            label = f"str_{string_counter}"
            file_name = op[0]
            row = op[1]
            col = op[2]
            oper = op[3]
            instructions[index] = (file_name, row, col, (oper[0], (Types.STR, label)))
            strings.append((label, string))
            string_counter += 1
        index += 1
    return instructions, memory, strings
    # return [parse_op(op) for op in code]


def cross_reference(instructions):
    index = 0
    stack = []
    assert len(Keywords) == 5, "Not all keywords can be cross-referenced yet"
    while index < len(instructions):
        op = instructions[index][3]
        if op[0] == Keywords.WHILE:
            stack.append((Keywords.WHILE, index))
        elif op[0] == Keywords.IF:
            stack.append((Keywords.IF, index))
        if op[0] == Keywords.DO:
            instr = stack.pop()
            assert instr[0] == Keywords.IF or instr[0] == Keywords.WHILE, "DO only supports IF-ELSE for now!"
            stack.append((Keywords.DO, instr, index))
        elif op[0] == Keywords.ELSE:
            instr = stack.pop()
            if instr[1][0] != Keywords.IF:
                print_compiler_error("Attempted to link `else` with non-if!",
                                     f"{get_instruction_location(instructions[index])}"
                                     f" => `{instr[1][0].name.lower()}` does not support `else`.")
            instructions[instr[2]] = (
                instructions[index][0], instructions[index][1], instructions[index][2],
                (instr[0], instr[1], index - instr[2]))
            stack.append((Keywords.ELSE, instr, index))
        elif op[0] == Keywords.END:
            instr = stack.pop()
            if instr[0] == Keywords.DO:
                instructions[instr[2]] = (
                    instructions[index][0], instructions[index][1], instructions[index][2],
                    (instr[0], instr[1], index - instr[2]))
                if instr[1][0] == Keywords.IF:
                    if len(instructions[index][3]) != 1:
                        print_compiler_error("Expected END block to only have one element",
                                             f"Found: {op}")
                    instructions[index] = (
                        instructions[index][0], instructions[index][1], instructions[index][2],
                        (op[0], 1))
                elif instr[1][0] == Keywords.WHILE:
                    instructions[index] = (
                        instructions[index][0], instructions[index][1], instructions[index][2],
                        (op[0], instructions[instr[2]][3][1][1] - index))
            elif instr[0] == Keywords.ELSE:
                instructions[instr[2]] = (
                    instructions[index][0], instructions[index][1], instructions[index][2],
                    (instr[0], index - instr[2]))
                instructions[index] = (
                    instructions[index][0], instructions[index][1], instructions[index][2],
                    (op[0], 1))
            else:
                assert False, "END only supports IF, ELSE and WHILE for now!"
        index += 1

    if len(stack) != 0:
        op_info = instructions[stack[-1][2]]
        print_compiler_error("Missing END",
                             f"{get_instruction_location(op_info)}"
                             f" => `{op_info[3][0].name.lower()}` has no matching end.")

    # contains all indexes which will be jumped to
    jmp_instr = []
    for i, op in enumerate(instructions):
        i = i + 1
        instr = op[3]
        if instr[0] == Keywords.ELSE:
            jmp_instr.append(instr[1] + i)
        elif instr[0] == Keywords.DO:
            jmp_instr.append(instr[2] + i)
        elif instr[0] == Keywords.END:
            if instr[1] != 1:
                jmp_instr.append(instr[1] + i)

    return instructions, jmp_instr


def type_check_program(instructions):
    assert len(OpSet) == 24, "Not all Operations are handled in type checking"
    assert len(Keywords) == 5, "Not all Keywords are handled in type checking"
    assert len(Types) == 3, "Not all Types are handled in type checking"
    stack = []
    keyword_stack = []
    for ip, instr in enumerate(instructions):
        op = instr[3]
        operator = op[0]
        if operator in OpSet:
            if operator == OpSet.PUSH_INT:
                stack.append(Types.INT)
            elif operator == OpSet.PUSH_PTR:
                stack.append(Types.PTR)
            elif operator == OpSet.PUSH_STR:
                stack.append(Types.STR)
            elif operator == OpSet.DROP:
                if len(stack) < 1:
                    print_compiler_error("Not enough operands for operation",
                                         f"{get_instruction_location(instr)}: {operator} expected 1 argument, found {len(stack)} instead: {stack}\n")
                else:
                    stack.pop()
            elif operator == OpSet.OVER:
                if len(stack) < 2:
                    print_compiler_error("Not enough operands for operation",
                                         f"{get_instruction_location(instr)}: {operator} expected 2 arguments, found {len(stack)} instead: {stack}\n")
                else:
                    type2 = stack.pop()
                    type1 = stack.pop()
                    stack.append(type1)
                    stack.append(type2)
                    stack.append(type1)
            elif operator == OpSet.DUP:
                if len(stack) < 1:
                    print_compiler_error("Not enough operands for operation",
                                         f"{get_instruction_location(instr)}: {operator} expected 1 argument, found {len(stack)} instead: {stack}\n")
                else:
                    type1 = stack.pop()
                    stack.append(type1)
                    stack.append(type1)
            elif operator == OpSet.ROT:
                if len(stack) < 3:
                    print_compiler_error("Not enough operands for operation",
                                         f"{get_instruction_location(instr)}: {operator} expected 3 arguments, found {len(stack)} instead: {stack}\n")
                else:
                    # type3 type2 type1
                    type3 = stack.pop()
                    type2 = stack.pop()
                    type1 = stack.pop()
                    stack.append(type2)
                    stack.append(type1)
                    stack.append(type3)
                    # type2 type1 type3
            elif operator == OpSet.SWAP:
                if len(stack) < 2:
                    print_compiler_error("Not enough operands for operation",
                                         f"{get_instruction_location(instr)}: {operator} expected 2 arguments, found {len(stack)} instead: {stack}\n")
                else:
                    # type2 type1
                    type2 = stack.pop()
                    type1 = stack.pop()
                    stack.append(type2)
                    stack.append(type1)
                    # type2 type1
            elif operator in [OpSet.ADD_INT, OpSet.SUB_INT, OpSet.MUL_INT, OpSet.DIV_INT, OpSet.MOD_INT] or operator in [OpSet.LT, OpSet.GT, OpSet.EQ, OpSet.NEQ]:
                if len(stack) < 2:
                    print_compiler_error("Not enough operands for operation",
                                         f"{get_instruction_location(instr)}: {operator} expected 2 arguments, found {len(stack)} instead: {stack}\n")
                else:
                    type2 = stack.pop()
                    type1 = stack.pop()
                    if type1 == Types.INT and type2 == Types.INT:
                        stack.append(Types.INT)
                    else:
                        print_compiler_error("Wrong types for operation",
                                             f"{get_instruction_location(instr)}: {operator} expected 2 Integers, found {type1} and {type2} instead.")
            elif operator in [OpSet.ADD_PTR, OpSet.SUB_PTR]:
                if len(stack) < 2:
                    print_compiler_error("Not enough operands for operation",
                                         f"{get_instruction_location(instr)}: {operator} expected 2 arguments, found {len(stack)} instead: {stack}\n")
                else:
                    type2 = stack.pop()
                    type1 = stack.pop()
                    if type1 == Types.PTR and type2 == Types.INT:
                        stack.append(Types.PTR)
                    else:
                        print_compiler_error("Wrong types for operation",
                                             f"{get_instruction_location(instr)}: {operator} expected {[Types.PTR, Types.INT]}, found {type1} and {type2} instead.")
            elif operator == OpSet.SET_INT:
                # value variable !64
                if len(stack) < 2:
                    print_compiler_error("Not enough operands for operation",
                                         f"{get_instruction_location(instr)}: {operator} expected 2 arguments, found {len(stack)} instead: {stack}\n")
                else:
                    type2 = stack.pop()
                    type1 = stack.pop()
                    if type1 == Types.INT and type2 == Types.PTR:
                        pass
                    else:
                        print_compiler_error("Wrong types for operation",
                                             f"{get_instruction_location(instr)}: {operator} expected {[Types.INT, Types.PTR]}, found {type1} and {type2} instead.")
            elif operator == OpSet.GET_INT:
                # variable ?64
                if len(stack) < 1:
                    print_compiler_error("Not enough operands for operation",
                                         f"{get_instruction_location(instr)}: {operator} expected 1 argument, found {len(stack)} instead: {stack}\n")
                else:
                    type1 = stack.pop()
                    if type1 == Types.PTR:
                        stack.append(Types.INT)
                    else:
                        print_compiler_error("Wrong types for operation",
                                             f"{get_instruction_location(instr)}: {operator} expected {[Types.PTR]}, found {type1} instead.")
            elif operator == OpSet.PRINT_INT:
                if len(stack) < 1:
                    print_compiler_error("Not enough operands for operation",
                                         f"{get_instruction_location(instr)}: {operator} expected 1 argument, found {len(stack)} instead: {stack}\n")
                else:
                    type1 = stack.pop()
                    if type1 == Types.INT:
                        pass
                    else:
                        print_compiler_error("Wrong types for operation",
                                             f"{get_instruction_location(instr)}: {operator} expected {[Types.INT]}, found {type1} instead.")
            elif operator == OpSet.PRINT_STR:
                if len(stack) < 1:
                    print_compiler_error("Not enough operands for operation",
                                         f"{get_instruction_location(instr)}: {operator} expected 1 argument, found {len(stack)} instead: {stack}\n")
                else:
                    type1 = stack.pop()
                    if type1 == Types.STR:
                        pass
                    else:
                        print_compiler_error("Wrong types for operation",
                                             f"{get_instruction_location(instr)}: {operator} expected {[Types.INT]}, found {type1} instead.")

            else:
                assert False, f"Not implemented type checking for {operator} yet"
        elif operator in Keywords:
            if operator in [Keywords.WHILE, Keywords.IF]:
                keyword_stack.append((ip, instr, stack.copy()))
            elif operator == Keywords.DO:
                pre_do = op[1][0]
                if pre_do in [Keywords.WHILE, Keywords.IF]:
                    if len(stack) < 1:
                        print_compiler_error("Not enough operands for operation",
                                             f"{get_instruction_location(instr)}: {operator} expected 1 argument, found {len(stack)} instead: {stack}\n")
                    else:
                        type1 = stack.pop()
                        if type1 != Types.INT:
                            print_compiler_error("Wrong types for operation",
                                                 f"{get_instruction_location(instr)}: {operator} expected {[Types.INT]}, found {type1} instead.")
                else:
                    assert False, f"{pre_do} type-checking for Keywords.DO not implemented yet"
            elif operator == Keywords.ELSE:
                block = keyword_stack.pop()
                # block_ip = block[0]
                block_instr = block[1]
                block_keyword = block_instr[3][0]
                block_stack = block[2]
                assert block_keyword == Keywords.IF, "This should never fail"
                keyword_stack.append((ip, instr, stack.copy(), block_instr, block_stack.copy()))
                stack = block_stack.copy()
            elif operator == Keywords.END:
                block = keyword_stack.pop()
                # block_ip = block[0]
                block_instr = block[1]
                block_keyword = block_instr[3][0]
                block_stack = block[2]
                if block_keyword == Keywords.WHILE or block_keyword == Keywords.IF:
                    if len(stack) < len(block_stack):
                        print_compiler_error("Stack modification error in type checking",
                                             f"{get_instruction_location(block_instr)}: `{block_keyword.name}` is not allowed to decrease the size of the stack.")
                    else:
                        pre_stack_len = len(block_stack)
                        post_stack_len = len(stack)
                        stack_len = pre_stack_len if pre_stack_len < post_stack_len else post_stack_len
                        for i in range(stack_len):
                            if stack[i] != block_stack[i]:
                                print_compiler_error("Stack modification error in type checking",
                                                     f"{get_instruction_location(block_instr)}: `{block_keyword.name}` is not allowed to modify the types on the stack.\n"
                                                     f"Before: {block_stack}\n"
                                                     f"After: {stack}")
                elif block_keyword == Keywords.ELSE:
                    if_block = block[3]
                    if_keyword = if_block[3][0]
                    assert if_keyword == Keywords.IF, "This should never fail"
                    if_stack = block[4]
                    len1 = len(if_stack)
                    len2 = len(block_stack)
                    len3 = len(stack)
                    if not (len1 == len2 == len3):
                        print_compiler_error("Stack modification error in type checking",
                                             f"{get_instruction_location(if_block)}: All `{if_keyword.name}`-branches must result in the same stack.\n"
                                             f"\tStack before IF: {if_stack}\n"
                                             f"\tStack before ELSE: {block_stack}\n"
                                             f"\tStack before END: {stack}")
                    else:
                        for i in range(len1):
                            if not (stack[i] == block_stack[i] == if_stack[i]):
                                print_compiler_error("Stack modification error in type checking",
                                                     f"{get_instruction_location(if_block)}: All `{if_keyword.name}`-branches must result in the same stack.\n"
                                                     f"\tStack before IF: {if_stack}\n"
                                                     f"\tStack before ELSE: {block_stack}\n"
                                                     f"\tStack before END: {stack}")

                else:
                    assert False, f"{block_keyword} not implemented yet in type-checking Keywords.END"
            else:
                print(keyword_stack)
                assert False, f"{operator} type-checking not implemented yet"
            # assert False, "Not implemented yet"
        else:
            print_compiler_error("Unreachable situation in type checking",
                                 f"Undefined operator {operator}\n"
                                 f"{get_instruction_location(instr)}")

    if len(stack) > 0:
        print_compiler_error("Unhandled Data on Stack",
                             f"There are still {len(stack)} item(s) on the stack:\n"
                             f"{stack}\n"
                             "Please make sure that the stack is empty after program is finished executing.",
                             ErrorTypes.STACK)
    return instructions


def evaluate_static_equations(instructions):
    # optimizes instructions like "10 2 1 3 * 4 5 * + - 2 * 7 - + 5 * 15"
    # by evaluating them in the pre-compiler phase and only pushing the result
    assert len(OpSet) == 24, "Make sure that `stack_op` in" \
                             "`evaluate_static_equations()` is up to date."
    # Last OP in our instruction set that is arithmetic
    # All Enum values less than that are available to be pre-evaluated
    new_code = []
    instr_stack = []
    for op in instructions:
        instr = op[3]
        if instr[0] in [OpSet.PUSH_INT, OpSet.ADD_INT, OpSet.SUB_INT, OpSet.MUL_INT, OpSet.DIV_INT, OpSet.MOD_INT]:
            instr_stack.append(op)
        else:
            if len(instr_stack) > 0:
                push_op = 0
                math_op = 0
                for s in instr_stack:
                    if s[3][0] == OpSet.PUSH_INT:
                        push_op += 1
                    else:
                        math_op += 1
                # print(push_op, math_op)
                if push_op != math_op + 1:
                    # Because all Math OP take pop 2 values and pop 1 value
                    # You need 1 more PUSH than MATH op
                    # If that's not the case, e.g. [*, *], just push them to output
                    # They might be used for variables or loops
                    for s in instr_stack:
                        new_code.append(s)
                    instr_stack = []
                else:
                    try:
                        stack = []
                        for s in instr_stack:
                            stack_op = s[3]
                            # print(f"  {s}")
                            if stack_op[0] == OpSet.PUSH_INT:
                                stack.append(stack_op[1])
                            elif stack_op[0] == OpSet.ADD_INT:
                                a = stack.pop()
                                b = stack.pop()
                                stack.append(a + b)
                            elif stack_op[0] == OpSet.SUB_INT:
                                a = stack.pop()
                                b = stack.pop()
                                stack.append(b - a)
                            elif stack_op[0] == OpSet.MUL_INT:
                                a = stack.pop()
                                b = stack.pop()
                                stack.append(a * b)
                            elif stack_op[0] == OpSet.DIV_INT:
                                a = stack.pop()
                                b = stack.pop()
                                stack.append(int(b / a))
                            elif stack_op[0] == OpSet.MOD_INT:
                                a = stack.pop()
                                b = stack.pop()
                                stack.append(int(b % a))
                        result = stack.pop()
                        # print("Result: " + str(result))
                        last_instr = instr_stack[-1]
                        # print(f"Last instruction: {last_instr}")
                        new_instr = (last_instr[0], last_instr[1], last_instr[2], (OpSet.PUSH_INT, result))
                        instr_stack = []
                        # print(f"Sequence breaker: {op}")
                        # print("*********************")
                        new_code.append(new_instr)
                    except IndexError:
                        for s in instr_stack:
                            new_code.append(s)
                        instr_stack = []
            new_code.append(op)

    return new_code


def tidy_memory(memory):
    memory = memory[1:][:-1]
    if len(memory) % 2 != 0:
        memory_as_string = ""
        for mem in memory:
            memory_as_string = memory_as_string + mem[3] + ", "
        if len(memory_as_string) > 0:
            memory_as_string = memory_as_string[:-2]
        print_compiler_error("Memory block is not aligned properly.",
                             "Expected memory block of form `var1 value1 var2 value2 ...`,\n"
                             "but memory size is not even.\n"
                             f"Found {len(memory)} elements: " +
                             memory_as_string)
    new_memory = []
    for i in range(0, len(memory), 2):
        new_memory.append((memory[i][3], memory[i + 1][3]))
    return new_memory


def simulate_code(instructions, mem):
    assert len(OpSet) == 19, "Not all OP can be simulated yet"
    stack = []
    memory = {}
    for m in mem:
        memory[m[0]] = 0
    index = 0
    while index < len(instructions):
        op = instructions[index][3]
        if op in memory:
            stack.append(op)
            index += 1
        elif op[0] == OpSet.PUSH:
            stack.append(op[1])
            index += 1
        elif op[0] == OpSet.ADD_INT:
            a = stack.pop()
            b = stack.pop()
            stack.append(a + b)
            index += 1
        elif op[0] == OpSet.SUB_INT:
            a = stack.pop()
            b = stack.pop()
            stack.append(b - a)
            index += 1
        elif op[0] == OpSet.MUL_INT:
            a = stack.pop()
            b = stack.pop()
            stack.append(a * b)
            index += 1
        elif op[0] == OpSet.DIV_INT:
            a = stack.pop()
            b = stack.pop()
            stack.append(int(b / a))
            index += 1
        elif op[0] == OpSet.MOD_INT:
            a = stack.pop()
            b = stack.pop()
            stack.append(int(b % a))
            index += 1
        elif op[0] == OpSet.PRINT_INT:
            print(stack.pop())
            index += 1
        elif op[0] == OpSet.IF or op[0] == OpSet.WHILE:
            index += 1
            pass
        elif op[0] == OpSet.SET_INT:
            variable = stack.pop()
            value = stack.pop()
            memory[variable] = value
            index += 1
        elif op[0] == OpSet.GET_INT:
            variable = stack.pop()
            value = memory[variable]
            stack.append(value)
            index += 1
        elif op[0] == OpSet.EQ:
            a = stack.pop()
            b = stack.pop()
            stack.append(int(a == b))
            index += 1
        elif op[0] == OpSet.NEQ:
            a = stack.pop()
            b = stack.pop()
            stack.append(int(a != b))
            index += 1
        elif op[0] == OpSet.LT:
            a = stack.pop()
            b = stack.pop()
            stack.append(int(b < a))
            index += 1
        elif op[0] == OpSet.GT:
            a = stack.pop()
            b = stack.pop()
            stack.append(int(b > a))
            index += 1
        elif op[0] == OpSet.DO:
            a = stack.pop()
            if a == 0:
                index += op[2]
            index += 1
        elif op[0] == OpSet.END:
            index += op[1]
        elif op[0] == OpSet.ELSE:
            index += op[1]
            index += 1
        else:
            print_compiler_error("Unknown operation!",
                                 f"Can't simulate unknown op {op}")


# Create Obj file: nasm -f win64 output.asm -o output.obj
# Link Obj together: golink /no /console /entry main output.obj MSVCRT.dll kernel32.dll
# Call Program: output.exe
def compile_code(program_name, instructions, memory, strings, labels, opt_flags: dict):
    assert len(OpSet) == 24, "Not all OP can be compiled yet"
    assert len(Keywords) == 5, "Not all Keywords can be compiled yet"
    silenced = opt_flags['-m']
    optimized = opt_flags['-o']
    keep_asm = opt_flags['-a']
    name = program_name.replace(".hpt", "")
    label_name = "instr"
    with open(name + ".tmp", "w") as output:
        output.write(f"  ; Generated code for {program_name}\n")
        output.write("default rel\n")
        output.write("\n")
        output.write("segment .data\n")
        output.write("  format_string db \"%lld\", 0xd, 0xa, 0\n")
        output.write("  true db 1\n"
                     "  false db 0\n")
        for label, value in strings:
            hex_value = ""
            char_index = 0
            while char_index < len(value):
                char = value[char_index]
                if char == "\\":
                    if char_index + 1 < len(value) and value[char_index + 1] == "n":
                        hex_value += "13, 10, "
                        char_index += 1
                else:
                    hex_value += str(ord(char)) + ", "
                char_index += 1
            hex_value += "0\n"
            output.write(f"  {label} db {hex_value}")
        output.write("\n")
        output.write("segment .bss\n")
        for var in memory:
            output.write(f"  {var[0]} resb {var[1]}\n")
        output.write("\n")
        output.write("segment .text\n"
                     "  global main\n"
                     "  extern ExitProcess\n"
                     "  extern printf\n")
        output.write("\n")
        output.write("main:\n")
        output.write("  push rbp\n"
                     "  mov rbp, rsp\n"
                     "  sub rsp, 32\n")
        for i, op in enumerate(instructions):
            op = op[3]
            if i in labels:
                output.write(f"{label_name}_{i}:\n")
            output.write(f"; -- {op} --\n")
            if op[0] in OpSet:
                if op[0] == OpSet.SWAP:
                    output.write(f"  pop rbx\n")
                    output.write(f"  pop rax\n")
                    output.write(f"  push rbx\n")
                    output.write(f"  push rax\n")
                elif op[0] == OpSet.ROT:
                    output.write(f"  pop rcx\n")
                    output.write(f"  pop rbx\n")
                    output.write(f"  pop rax\n")
                    output.write(f"  push rbx\n")
                    output.write(f"  push rcx\n")
                    output.write(f"  push rax\n")
                elif op[0] == OpSet.DUP:
                    output.write(f"  pop rax\n")
                    output.write(f"  push rax\n")
                    output.write(f"  push rax\n")
                elif op[0] == OpSet.DROP:
                    output.write(f"  pop rax\n")
                elif op[0] == OpSet.OVER:
                    output.write(f"  pop rbx\n")
                    output.write(f"  pop rax\n")
                    output.write(f"  push rax\n")
                    output.write(f"  push rbx\n")
                    output.write(f"  push rax\n")
                elif op[0] == OpSet.PUSH_INT:
                    value = op[1][1]
                    output.write(f"  mov rax, qword {value}\n")
                    output.write(f"  push rax\n")
                elif op[0] == OpSet.PUSH_PTR:
                    value = op[1][1]
                    output.write(f"  push {value}\n")
                elif op[0] == OpSet.PUSH_STR:
                    value = op[1][1]
                    output.write(f"  push {value}\n")
                elif op[0] == OpSet.ADD_INT or op[0] == OpSet.ADD_PTR:
                    output.write("  pop rax\n")
                    output.write("  pop rbx\n")
                    output.write("  add rax, rbx\n")
                    output.write("  push rax\n")
                elif op[0] == OpSet.SUB_INT or op[0] == OpSet.SUB_PTR:
                    output.write("  pop rax\n")
                    output.write("  pop rbx\n")
                    output.write("  sub rbx, rax\n")
                    output.write("  push rbx\n")
                elif op[0] == OpSet.MUL_INT:
                    output.write("  pop rax\n")
                    output.write("  pop rbx\n")
                    output.write("  mul rbx\n")
                    output.write("  push rax\n")
                elif op[0] == OpSet.DIV_INT:
                    output.write("  pop rbx\n")
                    output.write("  pop rax\n")
                    output.write("  mov rdx, 0\n")
                    output.write("  cqo\n")
                    output.write("  idiv rbx\n")
                    output.write("  push rax\n")
                elif op[0] == OpSet.MOD_INT:
                    output.write("  pop rbx\n")
                    output.write("  pop rax\n")
                    output.write("  mov rdx, 0\n")
                    output.write("  div rbx\n")
                    output.write("  push rdx\n")
                elif op[0] == OpSet.PRINT_INT:
                    output.write("  pop rax\n")
                    output.write("  push rbp\n"
                                 "  mov rbp, rsp\n"
                                 "  sub rsp, 32\n")
                    output.write("  mov rdx, rax\n")
                    output.write("  lea rcx, [format_string]\n")
                    output.write("  call printf\n")
                    output.write("  add rsp, 32\n"
                                 "  pop rbp\n")
                elif op[0] == OpSet.PRINT_STR:
                    output.write("  pop rax\n")
                    output.write("  push rbp\n"
                                 "  mov rbp, rsp\n"
                                 "  sub rsp, 32\n")
                    output.write("  lea rcx, [rax]\n")
                    output.write("  call printf\n")
                    output.write("  add rsp, 32\n"
                                 "  pop rbp\n")
                elif op[0] == OpSet.EQ:
                    output.write("  pop rax\n")
                    output.write("  pop rbx\n")
                    output.write("  sub rbx, rax\n")
                    output.write("  lahf\n")
                    output.write("  shr rax, 14\n")
                    output.write("  and rax, 1\n")
                    output.write("  push rax\n")
                elif op[0] == OpSet.NEQ:
                    output.write("  pop rax\n")
                    output.write("  pop rbx\n")
                    output.write("  sub rbx, rax\n")
                    output.write("  lahf\n")
                    output.write("  shr rax, 14\n")
                    output.write("  not rax\n")
                    output.write("  and rax, 1\n")
                    output.write("  push rax\n")
                elif op[0] == OpSet.LT:
                    output.write("  pop rax\n")
                    output.write("  pop rbx\n")
                    output.write("  sub rbx, rax\n")
                    output.write("  lahf\n")
                    output.write("  shr rax, 15\n")
                    output.write("  and rax, 1\n")
                    output.write("  push rax\n")
                elif op[0] == OpSet.GT:
                    output.write("  pop rbx\n")
                    output.write("  pop rax\n")
                    output.write("  sub rbx, rax\n")
                    output.write("  lahf\n")
                    output.write("  shr rax, 15\n")
                    output.write("  and rax, 1\n")
                    output.write("  push rax\n")
                elif op[0] == OpSet.SET_INT:
                    # value var set_int
                    output.write("  pop rax\n")
                    # rax contains var
                    output.write("  pop rbx\n")
                    # rbx contains val
                    output.write("  mov [rax], rbx\n")
                elif op[0] == OpSet.GET_INT:
                    # var get_int
                    output.write(f"  pop rax\n")
                    # rax contains ptr to var
                    output.write(f"  mov rbx, [rax]\n")
                    # rbx contains value of var
                    output.write("  push rbx\n")
                else:
                    assert False, f"Unreachable - This means that an operation can't be compiled yet, namely: {op[0]}"
            elif op[0] in Keywords:
                if op[0] == Keywords.IF or op[0] == Keywords.WHILE:
                    pass
                elif op[0] == Keywords.ELSE:
                    end_goal = i + op[1] + 1
                    output.write(f"  jmp {label_name}_{end_goal}\n")
                elif op[0] == Keywords.DO:
                    if op[1][0] == Keywords.IF or op[1][0] == Keywords.WHILE:
                        end_goal = i + op[2] + 1
                        output.write("  pop rax\n")
                        output.write("  test rax, rax\n")
                        output.write(f"  jz {label_name}_{end_goal}\n")
                    else:
                        print_compiler_error("Compiling DO not implemented yet")
                elif op[0] == Keywords.END:
                    if op[1] == 1:
                        pass
                    else:
                        end_goal = i + op[1] + 1
                        output.write(f"  jmp {label_name}_{end_goal}\n")
            else:
                print(f"`{op}` can't be compiled yet")
                exit(1)
        output.write("\n")
        output.write(f"{label_name}_{len(instructions)}:\n")
        output.write("  xor rcx, rcx\n")
        output.write("  call ExitProcess\n")

    if not silenced:
        print(f"[INFO] Generated {name}.tmp")

    if optimized:
        # registers = ["rax", "rbx", "rcx", "rdx", "rbp"]
        optimized = []
        not_done = []
        with open(f"{name}.tmp", "r") as unoptimized:
            content = unoptimized.readlines()
            unoptimized_line_count = len(content)
            content = [line.removesuffix("\n") for line in content]
            for i, line in enumerate(content):
                if line.startswith(";"):
                    optimized.append((i, line))
                else:
                    not_done.append((i, line))
            content = []
            for (i, op) in not_done:
                if not (op.startswith("  push") or op.startswith("  pop")):
                    optimized.append((i, op))
                else:
                    content.append((i, op))

            not_done = []
            content.sort(key=lambda tup: tup[0])
            for i in range(len(content)):
                index = content[i][0]
                op = content[i][1]
                instr = op.strip(" ").split(" ")
                if instr[0] != "push" and instr[0] != "pop":
                    optimized.append((index, op))
                else:
                    not_done.append((index, op))

            index = 0
            while index < len(not_done):
                curr_line = not_done[index]
                if index >= len(not_done) - 1:
                    optimized.append((curr_line[0], curr_line[1]))
                    index += 1
                else:
                    next_line = not_done[index + 1]
                    curr_line_index = curr_line[0]
                    next_line_index = next_line[0]
                    if next_line_index == curr_line_index + 2:
                        curr_line_instr = curr_line[1].strip(" ")
                        next_line_instr = next_line[1].strip(" ")
                        if curr_line_instr.startswith("push") and next_line_instr.startswith("pop"):
                            reg = curr_line_instr.split(" ")[-1]
                            if next_line_instr.endswith(reg):
                                # push rax
                                # pop  rax
                                # useless OP
                                pass
                            else:
                                optimized.append((curr_line_index, curr_line[1]))
                                optimized.append((next_line_index, next_line[1]))
                            index += 2
                        else:
                            optimized.append((curr_line_index, curr_line[1]))
                            optimized.append((next_line_index, next_line[1]))
                            index += 2
                    else:
                        optimized.append((curr_line_index, curr_line[1]))
                        index += 1

        optimized.sort(key=lambda tup: tup[0])

        # Replaces:
        #   ; -- (<OpSet.PUSH: 2>, value) --
        #   mov rax, qword value
        #   push rax
        #   ; -- variable --
        #   push var_name
        #   ; -- (<OpSet.SET_INT: 18>,) --
        #   pop rax
        #   pop rbx
        #   mov [rax], rbx
        # with:
        #   mov [var_name], qword value
        # TODO: Add this optimization
        # Replaces:
        #   ; -- var_name --
        #   push var_name
        #   ; -- (<OpSet.GET_INT: 19>,) --
        #   pop rax
        #   mov rbx, [rax]
        #   push rbx
        # with:
        #  push qword [var_name]
        # TODO: Add this optimization

        optimized_line_count = len(optimized)
        if not silenced:
            print(f"[INFO] Removed {unoptimized_line_count - optimized_line_count} "
                  f"lines of ASM due to optimization.")
        with open(f"{name}.asm", "w") as output:
            for (i, op) in optimized:
                output.write(op + "\n")
    else:
        with open(f"{name}.tmp", "r") as code:
            with open(f"{name}.asm", "w") as asm:
                asm.writelines(code.readlines())

    if not silenced:
        print(f"[INFO] Generated {name}.asm")
    os.remove(f"{name}.tmp")
    if not silenced:
        print(f"[INFO] Removed {name}.tmp")
    call_cmd(["nasm", "-f", "win64", f"{name}.asm", "-o", f"{name}.obj"], silenced)
    if not keep_asm:
        os.remove(f"{name}.asm")
        if not silenced:
            print(f"[INFO] Removed {name}.asm")
    call_cmd(["golink", "/no", "/console", "/entry", "main", f"{name}.obj", "MSVCRT.dll", "kernel32.dll"],
             silenced)
    os.remove(f"{name}.obj")
    if not silenced:
        print(f"[INFO] Removed {name}.obj")
        print(f"[CMD] Created {name}.exe")
    # call_cmd([f"{name}.exe"])


def shift(argv):
    return argv[0], argv[1:]


def get_help(flag):
    match flag:
        case '-h':
            return "Shows this help screen."
        case '-s':
            return "Simulates the given input code in Python."
        case '-c':
            return "Compiles the given input code and generates a single executable."
        case '-d':
            return "Debug Mode: Parses the input code, prints the instructions, then exits."
        case '-o':
            return "Optimize the generated code. Only works in combination with `-c`."
        case '-m':
            return "Mutes compilation command line output."
        case '-a':
            return "Keeps generated Assembly file after compilation."
        case _:
            return "[No description]"


def get_usage(program_name):
    return f"Usage: {program_name} [-h] <input.hpt> " \
           f"[-s | -c | -d] [-o, -m, -a]\n" \
           f"       If you need more help, run `{program_name} -h`"


def main():
    # TODO: Add Strings, Arrays, Functions
    flags = ['-h', '-s', '-c', '-d', '-o', '-m', '-a']
    exec_flags = flags[1:4]
    optional_flags = flags[4:]
    opt_flags = dict(zip(optional_flags, [False] * len(optional_flags)))
    program_name, sys.argv = shift(sys.argv)
    program_name = program_name.split("\\")[-1]
    if len(sys.argv) < 1:
        print_compiler_error("Not enough parameters!",
                             f"{get_usage(program_name)}\n")
    if sys.argv[0] == '-h':
        print(get_usage(program_name))
        for flag in flags:
            print(f"{flag}:\t" + get_help(flag))
        exit(0)
    if len(sys.argv) < 2:
        print_compiler_error("Not enough parameters!",
                             f"{get_usage(program_name)}\n")
    input_file, sys.argv = shift(sys.argv)
    if not input_file.endswith(".hpt"):
        print_compiler_error(f"File {input_file} does not end with `.hpt`!",
                             get_usage(program_name))
    code = ""
    try:
        code = load_from_file(input_file)
    except FileNotFoundError:
        print_compiler_error(f"File `{input_file} does not exist!",
                             get_usage(program_name))

    run_flag, sys.argv = shift(sys.argv)
    if run_flag not in exec_flags:
        print_compiler_error("Third Parameter has to be an execution flag!",
                             get_usage(program_name))

    instructions, memory, strings = parse_instructions(code)
    instructions, labels = cross_reference(instructions)
    instructions = type_check_program(instructions)
    instructions = evaluate_static_equations(instructions)

    memory = tidy_memory(memory)

    if len(sys.argv) > 0:
        opt_args = sys.argv
        for opt in opt_args:
            if opt not in optional_flags:
                print_compiler_error("Unknown Flag",
                                     f"Found `{opt}`. For valid flags run `{program_name} -h`")
            else:
                opt_flags[opt] = True

    if run_flag == '-s':
        simulate_code(instructions, memory)
    elif run_flag == '-c':
        compile_code(input_file, instructions, memory, strings, labels, opt_flags)
    elif run_flag == '-d':
        for i, mem in enumerate(memory):
            print(mem[0] + ": " + mem[1] + " bytes")
        print("*" * 50)
        for i, op in enumerate(instructions):
            print(i, op[3])
        exit(1)
    else:
        print(f"Unknown flag `{run_flag}`")
        print(get_usage(program_name))
        exit(1)


if __name__ == '__main__':
    main()

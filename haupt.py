from enum import Enum, auto
from typing import List
import subprocess
import sys
import os

import colorama
from colorama import Fore, Style

colorama.init()


class OpSet(Enum):
    NOP = auto()
    PUSH = auto()
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    MOD = auto()
    LT = auto()
    GT = auto()
    EQ = auto()
    NEQ = auto()
    PRINT = auto()
    IF = auto()
    ELSE = auto()
    DO = auto()
    END = auto()
    WHILE = auto()
    SET_INT = auto()
    GET_INT = auto()


COMMENT_CHAR = "#"


def get_instruction_location(instruction):
    return (f"{instruction[0]}:"
            f"{Fore.CYAN}{instruction[1]}:{instruction[2]}{Style.RESET_ALL}")


class ErrorTypes(Enum):
    NORMAL = auto()
    STACK = auto()


def print_error(err, info="", error_type=ErrorTypes.NORMAL):
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
    if line.startswith(COMMENT_CHAR):
        return []
    result = []
    line = line.split(" ")
    while '' in line:
        line.remove('')
    for col, token in enumerate(line):
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
    assert len(OpSet) == 19, "Not all OP can be parsed yet"
    if op == "+":
        op = OpSet.ADD,
    elif op == "-":
        op = OpSet.SUB,
    elif op == "*":
        op = OpSet.MUL,
    elif op == "/":
        op = OpSet.DIV,
    elif op == '%':
        op = OpSet.MOD,
    elif op == "print":
        op = OpSet.PRINT,
    elif (op.startswith("-") and op[1:].isdigit()) or op.isdigit():
        op = OpSet.PUSH, int(op)
    elif op == 'if':
        op = OpSet.IF,
    elif op == "==":
        op = OpSet.EQ,
    elif op == "!=":
        op = OpSet.NEQ,
    elif op == '>':
        op = OpSet.GT,
    elif op == '<':
        op = OpSet.LT,
    elif op == "do":
        op = OpSet.DO,
    elif op == "end":
        op = OpSet.END,
    elif op == "else":
        op = OpSet.ELSE,
    elif op == "while":
        op = OpSet.WHILE,
    elif op == "!64":
        op = OpSet.SET_INT,
    elif op == "?64":
        op = OpSet.GET_INT,
    elif op == "mem":
        print_error("Unexpected `mem` found",
                    f"{get_instruction_location(other_info)}"
                    f" => `{op}`\n"
                    f"Memory block needs to be declared at the very top of the program.")
    else:
        variable = False
        for mem in memory:
            if op == mem[3]:
                variable = True
                break
        if variable:
            pass
        else:
            print_error(f"Unknown token in parse_op(op)!",
                        f"{get_instruction_location(other_info)}"
                        f" => `{op}`")
    return other_info[0], other_info[1], other_info[2], op


def parse_instructions(code: List):
    memory_found = False
    for op in code:
        if memory_found and op[3] == "mem":
            print_error("Multiple memory blocks found",
                        "Right now only one memory block is supported\n")
        elif op[3] == "mem":
            memory_found = True
    memory = []
    memory_index = 0
    op = code[memory_index][3]
    if op == "mem":
        while code[memory_index][3] != "end":
            if memory_index == len(code) - 1:
                print_error("Matching `end` not found!",
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

    return instructions, memory
    # return [parse_op(op) for op in code]


# Format: OP: [PUSH_CNT, POP_CNT]
# Example: ADD: [1, 2] means that ADD pushes 1 operand to the stack and pops 2 operands.
push_pop_dict = {
    'NOP': [0, 0],
    'PUSH': [1, 0],
    'ADD': [1, 2],
    'SUB': [1, 2],
    'MUL': [1, 2],
    'DIV': [1, 2],
    'MOD': [1, 2],
    'LT': [1, 2],
    'GT': [1, 2],
    'EQ': [1, 2],
    'NEQ': [1, 2],
    'PRINT': [0, 1],
    'IF': [0, 0],
    'ELSE': [0, 0],
    'DO': [0, 1],
    'END': [0, 0],
    'WHILE': [0, 0],
    'SET_INT': [0, 2],
    'GET_INT': [1, 1],
}

assert len(OpSet) == len(push_pop_dict), "Not all instructions are included in `push_pop_dict`"


def evaluate_stack(instructions, memory):
    # goes over instructions, lazily evaluates stack size
    # if at any point any instruction does not have the operands it needs, print error
    stack_size = 0
    for instr in instructions:
        op = instr[3]
        variable = False
        for mem in memory:
            if op == mem[3]:
                variable = True
                break
        if not variable:
            stack_pop = push_pop_dict[op[0].name][1]
        else:
            stack_pop = 0
        if stack_size - stack_pop < 0:
            print_error("Stack underflow!",
                        f"{get_instruction_location(instr)}"
                        f" => `{op[0].name.lower()}` expected {stack_pop} argument(s), found {stack_size}",
                        ErrorTypes.STACK)
        stack_size -= stack_pop
        if not variable:
            stack_push = push_pop_dict[op[0].name][0]
        else:
            stack_push = 1
        stack_size += stack_push
    if stack_size > 0:
        print_error("Unhandled Data on Stack",
                    f"There are still {stack_size} item(s) on the stack.\n"
                    "Please make sure that the stack is empty after program is finished executing.",
                    ErrorTypes.STACK)
    return instructions


def evaluate_static_equations(instructions):
    # optimizes instructions like "10 2 1 3 * 4 5 * + - 2 * 7 - + 5 * 15"
    # by evaluating them in the pre-compiler phase and only pushing the result
    assert len(OpSet) == 19, "Make sure that `stack_op` in" \
                             "`evaluate_static_equations()` is up to date."
    # Last OP in our instruction set that is arithmetic
    # All Enum values less than that are available to be pre-evaluated
    last_arith = OpSet.MOD
    new_code = []
    instr_stack = []
    for op in instructions:
        instr = op[3]
        if not type(instr) == str and instr[0].value <= last_arith.value:
            instr_stack.append(op)
        else:
            if len(instr_stack) > 0:
                push_op = 0
                math_op = 0
                for s in instr_stack:
                    if s[3][0] == OpSet.PUSH:
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
                            if stack_op[0] == OpSet.PUSH:
                                stack.append(stack_op[1])
                            elif stack_op[0] == OpSet.ADD:
                                a = stack.pop()
                                b = stack.pop()
                                stack.append(a + b)
                            elif stack_op[0] == OpSet.SUB:
                                a = stack.pop()
                                b = stack.pop()
                                stack.append(b - a)
                            elif stack_op[0] == OpSet.MUL:
                                a = stack.pop()
                                b = stack.pop()
                                stack.append(a * b)
                            elif stack_op[0] == OpSet.DIV:
                                a = stack.pop()
                                b = stack.pop()
                                stack.append(int(b / a))
                            elif stack_op[0] == OpSet.MOD:
                                a = stack.pop()
                                b = stack.pop()
                                stack.append(int(b % a))
                        result = stack.pop()
                        # print("Result: " + str(result))
                        last_instr = instr_stack[-1]
                        # print(f"Last instruction: {last_instr}")
                        new_instr = (last_instr[0], last_instr[1], last_instr[2], (OpSet.PUSH, result))
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


def cross_reference(instructions):
    index = 0
    stack = []
    assert len(OpSet) == 19, "Not all OP can be cross-referenced yet"
    while index < len(instructions):
        op = instructions[index][3]
        if op[0] == OpSet.WHILE:
            stack.append((OpSet.WHILE, index))
        elif op[0] == OpSet.IF:
            stack.append((OpSet.IF, index))
        if op[0] == OpSet.DO:
            instr = stack.pop()
            assert instr[0] == OpSet.IF or instr[0] == OpSet.WHILE, "DO only supports IF-ELSE for now!"
            stack.append((OpSet.DO, instr, index))
        elif op[0] == OpSet.ELSE:
            instr = stack.pop()
            if instr[1][0] != OpSet.IF:
                print_error("Attempted to link `else` with non-if!",
                            f"{get_instruction_location(instructions[index])}"
                            f" => `{instr[1][0].name.lower()}` does not support `else`.")
            instructions[instr[2]] = (
                instructions[index][0], instructions[index][1], instructions[index][2],
                (instr[0], instr[1], index - instr[2]))
            stack.append((OpSet.ELSE, instr, index))
        elif op[0] == OpSet.END:
            instr = stack.pop()
            if instr[0] == OpSet.DO:
                instructions[instr[2]] = (
                    instructions[index][0], instructions[index][1], instructions[index][2],
                    (instr[0], instr[1], index - instr[2]))
                if instr[1][0] == OpSet.IF:
                    if len(instructions[index][3]) != 1:
                        print_error("Expected END block to only have one element",
                                    f"Found: {op}")
                    instructions[index] = (
                        instructions[index][0], instructions[index][1], instructions[index][2],
                        (op[0], 1))
                elif instr[1][0] == OpSet.WHILE:
                    instructions[index] = (
                        instructions[index][0], instructions[index][1], instructions[index][2],
                        (op[0], instructions[instr[2]][3][1][1] - index))
            elif instr[0] == OpSet.ELSE:
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
        print_error("Missing END",
                    f"{get_instruction_location(op_info)}"
                    f" => `{op_info[3][0].name.lower()}` has no matching end.")

    # contains all indexes which will be jumped to
    jmp_instr = []
    for i, op in enumerate(instructions):
        i = i + 1
        instr = op[3]
        if instr[0] == OpSet.ELSE:
            jmp_instr.append(instr[1] + i)
        elif instr[0] == OpSet.DO:
            jmp_instr.append(instr[2] + i)
        elif instr[0] == OpSet.END:
            if instr[1] != 1:
                jmp_instr.append(instr[1] + i)

    return instructions, jmp_instr


def tidy_memory(memory):
    memory = memory[1:][:-1]
    if len(memory) % 2 != 0:
        memory_as_string = ""
        for mem in memory:
            memory_as_string = memory_as_string + mem[3] + ", "
        if len(memory_as_string) > 0:
            memory_as_string = memory_as_string[:-2]
        print_error("Memory block is not aligned properly.",
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
        elif op[0] == OpSet.ADD:
            a = stack.pop()
            b = stack.pop()
            stack.append(a + b)
            index += 1
        elif op[0] == OpSet.SUB:
            a = stack.pop()
            b = stack.pop()
            stack.append(b - a)
            index += 1
        elif op[0] == OpSet.MUL:
            a = stack.pop()
            b = stack.pop()
            stack.append(a * b)
            index += 1
        elif op[0] == OpSet.DIV:
            a = stack.pop()
            b = stack.pop()
            stack.append(int(b / a))
            index += 1
        elif op[0] == OpSet.MOD:
            a = stack.pop()
            b = stack.pop()
            stack.append(int(b % a))
            index += 1
        elif op[0] == OpSet.PRINT:
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
            print_error("Unknown operation!",
                        f"Can't simulate unknown op {op}")


# Create Obj file: nasm -f win64 output.asm -o output.obj
# Link Obj together: golink /no /console /entry main output.obj MSVCRT.dll kernel32.dll
# Call Program: output.exe
def compile_code(instructions, memory, labels, opt_flags: dict):
    assert len(OpSet) == 19, "Not all OP can be compiled yet"
    silenced = opt_flags['-m']
    optimized = opt_flags['-o']
    name = "output"
    label_name = "instr"
    with open(name + ".tmp", "w") as output:
        output.write("bits 64\n")
        output.write("default rel\n")
        output.write("\n")
        output.write("segment .data\n")
        output.write("  format_string db \"%lld\", 0xd, 0xa, 0\n")
        output.write("  true db 1\n"
                     "  false db 0\n")
        output.write("\n")
        output.write("segment .bss\n")
        for var in memory:
            output.write(f"  {var[0]} resb {var[1]}\n")
        output.write("\n")
        output.write("segment .text\n"
                     "global main\n"
                     "extern ExitProcess\n"
                     "extern printf\n")
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
            if type(op) == str:
                variable = False
                for var, size in memory:
                    if var == op:
                        variable = True
                        break
                if not variable:
                    print_error("Unexpected token in compilation",
                                f"=> `{op}`.")
                else:
                    output.write(f"  push {op}\n")
            elif op[0] == OpSet.PUSH:
                output.write(f"  mov rax, qword {op[1]}\n")
                output.write(f"  push rax\n")
            elif op[0] == OpSet.ADD:
                output.write("  pop rax\n")
                output.write("  pop rbx\n")
                output.write("  add rax, rbx\n")
                output.write("  push rax\n")
            elif op[0] == OpSet.SUB:
                output.write("  pop rax\n")
                output.write("  pop rbx\n")
                output.write("  sub rbx, rax\n")
                output.write("  push rbx\n")
            elif op[0] == OpSet.MUL:
                output.write("  pop rax\n")
                output.write("  pop rbx\n")
                output.write("  mul rbx\n")
                output.write("  push rax\n")
            elif op[0] == OpSet.DIV:
                output.write("  pop rbx\n")
                output.write("  pop rax\n")
                output.write("  mov rdx, 0\n")
                output.write("  cqo\n")
                output.write("  idiv rbx\n")
                output.write("  push rax\n")
            elif op[0] == OpSet.MOD:
                output.write("  pop rbx\n")
                output.write("  pop rax\n")
                output.write("  mov rdx, 0\n")
                output.write("  div rbx\n")
                output.write("  push rdx\n")
            elif op[0] == OpSet.PRINT:
                output.write("  pop rdx\n")
                output.write("  lea rcx, [format_string]\n")
                output.write("  call printf\n")
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
            elif op[0] == OpSet.IF or op[0] == OpSet.WHILE:
                pass
            elif op[0] == OpSet.ELSE:
                end_goal = i + op[1] + 1
                output.write(f"  jmp {label_name}_{end_goal}\n")
            elif op[0] == OpSet.DO:
                if op[1][0] == OpSet.IF or op[1][0] == OpSet.WHILE:
                    end_goal = i + op[2] + 1
                    output.write("  pop rax\n")
                    output.write("  cmp rax, 0\n")
                    output.write(f"  je {label_name}_{end_goal}\n")
                else:
                    print_error("Compiling DO not implemented yet")
            elif op[0] == OpSet.END:
                if op[1] == 1:
                    pass
                else:
                    end_goal = i + op[1] + 1
                    output.write(f"  jmp {label_name}_{end_goal}\n")
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
                print(f"`{op}` can't be compiled yet")
                exit(1)
        output.write("\n")
        output.write(f"{label_name}_{len(instructions)}:\n")
        output.write("  xor rcx, rcx\n")
        output.write("  call ExitProcess\n")

    if not silenced:
        print(f"[INFO] Generated {name}.tmp")

    if optimized:
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
    call_cmd(["golink", "/no", "/console", "/entry", "main", f"{name}.obj", "MSVCRT.dll", "kernel32.dll"],
             silenced)
    if not silenced:
        print(f"[CMD] Created {name}.exe")
    # call_cmd([f"{name}.exe"])


def shift(argv):
    return argv[0], argv[1:]


def get_help(flag):
    match flag:
        case '-s':
            return "Simulates the given input code in Python."
        case '-c':
            return "Compiles the given input code and generates a single executable."
        case '-d':
            return "Debug Mode: Parses the input code, prints the instructions, then exits."
        case '-o':
            return "Optimize the generated code. Only works in combination with `-c`."
        case '-h':
            return "Shows this help screen."
        case '-m':
            return "Mutes compilation command line output."
        case _:
            return "[No description]"


def get_usage(program_name):
    return f"Usage: {program_name} [-h] <input.hpt> " \
           f"[-s | -c | -d] [-o, -m]\n" \
           f"       If you need more help, run `{program_name} -h`"


def main():
    # TODO: Add Strings, Arrays, Functions
    flags = ['-h', '-s', '-c', '-d', '-o', '-m']
    exec_flags = flags[1:4]
    optional_flags = flags[4:]
    opt_flags = dict(zip(optional_flags, [False] * len(optional_flags)))
    program_name, sys.argv = shift(sys.argv)
    program_name = program_name.split("\\")[-1]
    if len(sys.argv) < 1:
        print_error("Not enough parameters!",
                    f"{get_usage(program_name)}\n")
    if sys.argv[0] == '-h':
        print(get_usage(program_name))
        for flag in flags:
            print(f"{flag}:\t" + get_help(flag))
        exit(0)
    if len(sys.argv) < 2:
        print_error("Not enough parameters!",
                    f"{get_usage(program_name)}\n")
    input_file, sys.argv = shift(sys.argv)
    if not input_file.endswith(".hpt"):
        print_error(f"File {input_file} does not end with `.hpt`!",
                    get_usage(program_name))
    code = ""
    try:
        code = load_from_file(input_file)
    except FileNotFoundError:
        print_error(f"File `{input_file} does not exist!",
                    get_usage(program_name))

    run_flag, sys.argv = shift(sys.argv)
    if run_flag not in exec_flags:
        print_error("Third Parameter has to be an execution flag!",
                    get_usage(program_name))

    instructions, memory = parse_instructions(code)
    instructions = evaluate_stack(instructions, memory)
    instructions = evaluate_static_equations(instructions)
    instructions, labels = cross_reference(instructions)

    memory = tidy_memory(memory)

    if len(sys.argv) > 0:
        opt_args = sys.argv
        for opt in opt_args:
            if opt not in optional_flags:
                print_error("Unknown Flag",
                            f"Found `{opt}`. For valid flags run `{program_name} -h`")
            else:
                opt_flags[opt] = True

    if run_flag == '-s':
        simulate_code(instructions, memory)
    elif run_flag == '-c':
        compile_code(instructions, memory, labels, opt_flags)
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

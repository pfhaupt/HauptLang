from enum import Enum, auto
from typing import List
import subprocess
import sys

import colorama
from colorama import Fore, Style

colorama.init()


class OpSet(Enum):
    NOP = auto()
    GET_MEM = auto()
    SET_MEM = auto()
    PUSH = auto()
    PRINT = auto()
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    MOD = auto()
    IF = auto()
    ELSE = auto()
    DO = auto()
    END = auto()
    WHILE = auto()
    EQ = auto()
    NEQ = auto()
    LT = auto()
    GT = auto()


GET_MEM_CHAR = "&"
SET_MEM_CHAR = "$"
COMMENT_CHAR = "#"

memory = {}


def print_error(err, info=""):
    print(f"{Fore.LIGHTRED_EX}ERROR: {err}{Style.RESET_ALL}\n{info}")
    exit(1)


def call_cmd(cmd: List):
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
                for (col, token) in get_token_in_line(line)]


def parse_op(op: str):
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
    elif op.isdigit():
        op = OpSet.PUSH, int(op)
    elif op.startswith(SET_MEM_CHAR):
        op = OpSet.SET_MEM, op[len(SET_MEM_CHAR):]
    elif op.startswith(GET_MEM_CHAR):
        op = OpSet.GET_MEM, op[len(GET_MEM_CHAR):]
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
    else:
        print_error(f"Unknown token in parse_op(op)! Found",
                    f"{other_info[0]}:"
                    f"{Fore.CYAN}{other_info[1]}:{other_info[2]}{Style.RESET_ALL}"
                    f" => `{op}`")
    return other_info[0], other_info[1], other_info[2], op


def parse_instructions(code: List):
    return [parse_op(op) for op in code]


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
                            f"{instructions[index][0]}:"
                            f"{Fore.CYAN}{instructions[index][1]}:{instructions[index][2]}{Style.RESET_ALL}"
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
                    f"{op_info[0]}:"
                    f"{Fore.CYAN}{op_info[1]}:{op_info[2]}{Style.RESET_ALL}"
                    f" => `{op_info[3][0].name.lower()}` has no matching end.")

    return instructions


def simulate_code(instructions):
    assert len(OpSet) == 16, "Not all OP can be simulated yet"
    stack = []
    index = 0
    while index < len(instructions):
        op = instructions[index][3]
        if op[0] == OpSet.PUSH:
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
        elif op[0] == OpSet.PRINT:
            print(stack.pop())
            index += 1
        elif op[0] == OpSet.SET_MEM:
            memory[op[1]] = stack.pop()
            index += 1
        elif op[0] == OpSet.GET_MEM:
            stack.append(memory[op[1]])
            index += 1
        elif op[0] == OpSet.IF or op[0] == OpSet.WHILE:
            index += 1
            pass
        elif op[0] == OpSet.EQ:
            a = stack.pop()
            b = stack.pop()
            stack.append(int(a == b))
            index += 1
        elif op[0] == OpSet.LT:
            a = stack.pop()
            b = stack.pop()
            stack.append(int(b < a))
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


def get_vars(instructions):
    return {op[3][1] for op in instructions if op[3][0] == OpSet.SET_MEM}


# Create Obj file: nasm -f win64 output.asm -o output.obj
# Link Obj together: golink /no /console /entry main output.obj MSVCRT.dll kernel32.dll
# Call Program: output.exe
# You can combine those commands with && for a single cmd to do all of it
# Measure speed: powershell Measure-Command { .\output.exe }
def compile_code(instructions):
    assert len(OpSet) == 19, "Not all OP can be compiled yet"
    # TODO: Optimize PUSH/POP stuff
    used_vars = get_vars(instructions)

    name = "output"
    label_name = "instr"
    with open(name + ".asm", "w") as output:
        output.write("bits 64\n")
        output.write("default rel\n")
        output.write("segment .data\n")
        output.write("  format_string db \"%lld\", 0xd, 0xa, 0\n")
        output.write("  true db 1\n"
                     "  false db 0\n")
        for var in used_vars:
            output.write(f"  {var} dq 0\n")
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
            if len(op) == 1:
                print(i, op[0].name)
            else:
                print(i, op[0].name, op[1])
            output.write(f"{label_name}_{i}:\n")
            output.write(f"; -- {op} --\n")
            if op[0] == OpSet.PUSH:
                output.write(f"  mov rax, qword {op[1]}\n")
                output.write(f"  push rax\n")
                # output.write(f"  push rax\n")
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
                output.write("  pop rbx\n")
                output.write("  pop rax\n")
                output.write("  cmp rax, rbx\n")
                output.write("  pushf\n")
                output.write("  pop rax\n")
                output.write("  shr rax, 6\n")
                output.write("  and rax, 1\n")
                output.write("  push rax\n")
            elif op[0] == OpSet.NEQ:
                output.write("  pop rbx\n")
                output.write("  pop rax\n")
                output.write("  cmp rax, rbx\n")
                output.write("  pushf\n")
                output.write("  pop rax\n")
                output.write("  shr rax, 6\n")
                output.write("  not rax\n")
                output.write("  and rax, 1\n")
                output.write("  push rax\n")
            elif op[0] == OpSet.LT:
                output.write("  pop rbx\n")
                output.write("  pop rax\n")
                output.write("  cmp rax, rbx\n")
                output.write("  pushf\n")
                output.write("  pop rax\n")
                output.write("  shr rax, 7\n")
                output.write("  and rax, 1\n")
                output.write("  push rax\n")
            elif op[0] == OpSet.GT:
                output.write("  pop rbx\n")
                output.write("  pop rax\n")
                output.write("  cmp rax, rbx\n")
                output.write("  pushf\n")
                output.write("  pop rax\n")
                output.write("  shr rax, 7\n")
                output.write("  not rax\n")
                output.write("  and rax, 1\n")
                output.write("  push rax\n")
            elif op[0] == OpSet.IF or op[0] == OpSet.WHILE:
                pass
            elif op[0] == OpSet.ELSE:
                end_goal = i + op[1]
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
                    end_goal = i + op[1]
                    output.write(f"  jmp {label_name}_{end_goal}\n")
            elif op[0] == OpSet.SET_MEM:
                output.write("  pop rax\n")
                output.write(f"  mov [{op[1]}], rax\n")
            elif op[0] == OpSet.GET_MEM:
                output.write(f"  mov rax, [{op[1]}]\n")
                output.write("  push rax\n")
            else:
                print(f"{op} can't be compiled yet")
                exit(1)
        output.write("\n")
        output.write(f"{label_name}_{len(instructions)}:\n")
        output.write("  xor rax, rax\n")
        output.write("  call ExitProcess\n")
    print("File written")
    call_cmd(["nasm", "-f", "win64", f"{name}.asm", "-o", f"{name}.obj"])
    call_cmd(["golink", "/no", "/console", "/entry", "main", f"{name}.obj", "MSVCRT.dll", "kernel32.dll"])
    print(f"[CMD] Created {name}.exe")
    # call_cmd([f"{name}.exe"])


def shift(argv):
    return argv[0], argv[1:]


def main():
    # TODO: Add Strings, Arrays, Functions

    program_name, sys.argv = shift(sys.argv)
    program_name = program_name.split("\\")[-1]
    if sys.argv[0] == '-h':
        print(f"Usage: {program_name} <input.hpt> [-s | -c | -d]")
        # TODO: Add Flag description
        exit(0)
    if len(sys.argv) < 2:
        print_error("Not enough parameters!",
                    f"Usage: {program_name} <input.hpt> [-s | -c | -d]\n"
                    f"       If you need more help, run `{program_name} -h`")
    input_file, sys.argv = shift(sys.argv)
    if not input_file.endswith(".hpt"):
        print_error(f"File {input_file} does not end with `.hpt`!",
                    f"Usage: {program_name} <input.hpt> [-s | -c | -d]")
    code = ""
    try:
        code = load_from_file(input_file)
    except FileNotFoundError:
        print_error(f"File `{input_file} does not exist!",
                    f"Usage: {program_name} <input.hpt> [-s | -c | -d]")

    instructions = parse_instructions(code)
    # TODO: Optimize static equations,
    #       move them to pre-compiler
    instructions = cross_reference(instructions)

    run_flag, sys.argv = shift(sys.argv)
    if run_flag == '-s':
        print("Simulated Output: ")
        simulate_code(instructions)
    elif run_flag == '-c':
        print("Compiled Output: ")
        compile_code(instructions)
    elif run_flag == '-d':
        for i, op in enumerate(instructions):
            print(i, op)
        exit(1)
    else:
        print(f"Unknown flag `{run_flag}")
        print(f"Usage: {program_name} <input.hpt> [-s | -c | -d]")
        exit(1)


if __name__ == '__main__':
    main()

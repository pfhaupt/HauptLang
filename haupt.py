from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Union, Optional
import subprocess
import sys
import os

import colorama
from colorama import Fore, Style

colorama.init()


class Type(Enum):
    INT = auto()
    PTR = auto()
    STR = auto()


@dataclass
class Location:
    file: str
    row: int
    col: int

    def __str__(self):
        return f"{self.file}:{self.row}:{self.col}"


@dataclass
class Token:
    loc: Location
    name: str

    def __str__(self):
        return f"{self.loc}: `{self.name}`"


@dataclass
class Signature:
    inputs: List[Type]
    outputs: List[Type]


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


assert len(OpSet) == 24, "Not all Operations are in the lookup table yet!"
INTRINSIC_LOOKUP: dict[str, OpSet] = {
    "nop": OpSet.NOP,
    "dup": OpSet.DUP,
    "drop": OpSet.DROP,
    "rot": OpSet.ROT,
    "swap": OpSet.SWAP,
    "over": OpSet.OVER,
    "+": OpSet.ADD_INT,
    "-": OpSet.SUB_INT,
    "*": OpSet.MUL_INT,
    "/": OpSet.DIV_INT,
    "%": OpSet.MOD_INT,
    "ptr+": OpSet.ADD_PTR,
    "ptr-": OpSet.SUB_PTR,
    # "": OpSet.PUSH_INT,
    # "": OpSet.PUSH_STR,
    # "": OpSet.PUSH_PTR,
    "puti": OpSet.PRINT_INT,
    "puts": OpSet.PRINT_STR,
    "<": OpSet.LT,
    ">": OpSet.GT,
    "==": OpSet.EQ,
    "!=": OpSet.NEQ,
    "!64": OpSet.SET_INT,
    "?64": OpSet.GET_INT,
}


class Keyword(Enum):
    IF = auto()
    ELSE = auto()
    DO = auto()
    END = auto()
    WHILE = auto()
    MEMORY = auto()


assert len(Keyword) == 6, "Not all Keywords are in the lookup table yet!"
KEYWORD_LOOKUP: dict[str, Keyword] = {
    "if": Keyword.IF,
    "else": Keyword.ELSE,
    "do": Keyword.DO,
    "end": Keyword.END,
    "while": Keyword.WHILE,
    "memory": Keyword.MEMORY
}


@dataclass
class DataTuple:
    typ: Type
    value: Union[int, str]

    def __str__(self):
        return f"{self.typ} {self.value}"


@dataclass
class Operation:
    operation: Union[Keyword, OpSet]
    operand: Optional[DataTuple]

    def __str__(self):
        return f"{self.operation} {self.operand}" if self.operand else f"{self.operation}"


@dataclass
class Instruction:
    loc: Location
    word: Operation

    def __str__(self):
        return f"{self.loc}: {self.word}"


@dataclass
class Memory:
    loc: Location
    name: str
    size: int

    def __str__(self):
        return f"{self.loc}: {self.name} has a size of {self.size} bytes."


COMMENT_CHAR = "#"


def get_instruction_location(instruction):
    return (f"{instruction[0]}:"
            f"{Fore.CYAN}{instruction[1]}:{instruction[2]}{Style.RESET_ALL}")


class ErrorType(Enum):
    NORMAL = auto()
    STACK = auto()


def print_compiler_error(err, info="", error_type=ErrorType.NORMAL):
    match error_type:
        case ErrorType.NORMAL:
            if len(info) == 0:
                print(f"{Fore.LIGHTRED_EX}Error: {err}{Style.RESET_ALL}")
            else:
                print(f"{Fore.LIGHTRED_EX}Error: {err}{Style.RESET_ALL}\n{info}")
        case ErrorType.STACK:
            if len(info) == 0:
                print(f"{Fore.LIGHTRED_EX}StackError: {err}{Style.RESET_ALL}")
            else:
                print(f"{Fore.LIGHTRED_EX}StackError: {err}{Style.RESET_ALL}\n{info}")

    exit(1)


def call_cmd(cmd: List, silenced=False):
    if not silenced:
        print("[CMD] " + " ".join(cmd))
    subprocess.call(cmd)


def check_string(s):
    if len(s) == 0 or (not s[0].isalpha() and s[0] != "_"):
        return False
    for char in s[1:]:
        if not char.isalnum() and char not in ['_', '-']:
            return False
    return True


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
                assert len(buffer) == 0, f"Buffer is expected to be empty, found {buffer} instead"
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
    if len(buffer) > 0:
        parsed_token.append(buffer)
    for col, token in enumerate(parsed_token):
        result.append((col, token))
    return result


def load_from_file(file_path: str):
    with open(file_path, "r") as f:
        return [Token(loc=Location(file=file_path, row=row + 1, col=col + 1), name=token)
                for (row, line) in get_lines(f.readlines())
                for (col, token) in get_token_in_line(line.split(COMMENT_CHAR)[0])]


def parse_memory_block(code: List[Instruction], ip: int, global_memory: List[Memory]):
    # Syntax: memory variable byte_count end
    if len(code) - (ip + 1) < 3:
        # we need 4 words for each block
        # ip points at number 1
        print_compiler_error("Not enough words for memory block")
    ip += 1
    variable: Token = code[ip]
    variable_name: str = variable.name
    if not check_string(variable_name):
        if not variable_name[0].isalpha():
            error_descr = f"{variable.loc}: The name is only allowed to start with a letter or underscore\n" \
                          f"Found: {variable_name}"
        else:
            error_descr = f"{variable.loc}: The name can only consist of letters, numbers and [-, _]\n" \
                          f"Found: {variable_name}"
        print_compiler_error("Invalid name inside memory",
                             error_descr)
    elif variable_name in INTRINSIC_LOOKUP or variable_name in KEYWORD_LOOKUP:
        print_compiler_error("Invalid variable name",
                             f"{ variable.loc}: Name is reserved.\n"
                             f"Found: {variable_name}")
    else:
        for mem in global_memory:
            if variable_name == mem.name:
                print_compiler_error("Redefinition of memory block.",
                                     f"{variable.loc}: {variable_name} is already defined here:\n"
                                     f"  {mem.loc} with a size of {mem.size} bytes.")
                print("Oh no")
                exit(1)
        pass
        # print("valid name")
    ip += 1
    byte_count: Token = code[ip]
    byte_string: str = byte_count.name
    size = -1
    if byte_string.isdigit():
        size: int = int(byte_string)
        if size == 0:
            print_compiler_error("Invalid size inside memory",
                                 f"{byte_count.loc}: Memory size is expected to be greater than 0.\n"
                                 f"Found: {byte_string}")
        else:
            pass
            # print("valid size")
    elif byte_string.startswith("-") and byte_string[1:].isdigit():
        print_compiler_error("Invalid size inside memory",
                             f"{byte_count.loc}: Memory size is expected to be greater than 0.\n"
                             f"Found: {byte_string}")
    else:
        print_compiler_error("Invalid size inside memory",
                             f"{byte_count.loc}: Memory size is expected to be a number.\n"
                             f"Found: {byte_string}")
    ip += 1
    end_token: Token = code[ip]
    end_name: str = end_token.name
    if not end_name == "end":
        print_compiler_error("Unexpected word in parsing",
                             f"{end_token.loc} Expected `end`, found `{end_name}` instead.")
    else:
        pass
        # print("valid end")

    return ip, Memory(loc=variable.loc, name=variable_name, size=size)


def parse_instructions(code: List[Instruction]):
    global_memory: List[Memory] = []
    strings: List[tuple] = []
    instructions: List[Instruction] = []
    keyword_stack: List[tuple] = []
    jump_labels: List[DataTuple] = []

    ip = 0
    while ip < len(code):
        token: Token = code[ip]
        location: Location = token.loc
        name: str = token.name
        op: Instruction = None
        if name in INTRINSIC_LOOKUP:
            # print(f"Found intrinsic {name}")
            word = Operation(operation=INTRINSIC_LOOKUP[name], operand=None)
            op = Instruction(loc=location, word=word)
        elif name in KEYWORD_LOOKUP:
            # print(f"Found keyword {name}")
            if name == "memory":
                ip, memory_unit = parse_memory_block(code, ip, global_memory)
                global_memory.append(memory_unit)

                word = Operation(operation=KEYWORD_LOOKUP[name], operand=None)
                op = Instruction(loc=location, word=word)
            elif name == "while" or name == "if":
                keyword_stack.append((len(instructions), token))

                word = Operation(operation=KEYWORD_LOOKUP[name], operand=None)
                op = Instruction(loc=location, word=word)
            elif name == "do":
                if len(keyword_stack) < 1:
                    print_compiler_error("Lonely DO found.",
                                         f"{location}: Could not find matching if, while or proc before this `do`.")
                pre_do: tuple = keyword_stack.pop()
                pre_do_ip: int = pre_do[0]
                pre_do_keyword: Token = pre_do[1]
                if pre_do_keyword.name == "while" or pre_do_keyword.name == "if":
                    keyword_stack.append(((len(instructions), token), pre_do))
                else:
                    print_compiler_error("Unexpected keyword in parsing",
                                         f"{pre_do_keyword.loc}: Expected to be while or if.\n"
                                         f"Found: {pre_do_keyword.name}")

                word = Operation(operation=KEYWORD_LOOKUP[name], operand=None)
                op = Instruction(loc=location, word=word)
            elif name == "else":
                if len(keyword_stack) < 1:
                    print_compiler_error("Lonely ELSE found.",
                                         f"{location}: Could not find matching `do` before this `else`.")
                pre_else: tuple = keyword_stack.pop()
                pre_else_tuple: tuple = pre_else[0]
                pre_else_ip: int = pre_else_tuple[0]
                pre_do_tuple: tuple = pre_else[1]
                pre_do_ip: int = pre_do_tuple[0]
                pre_do_token: Token = pre_do_tuple[1]
                if pre_do_token.name != "if":
                    print_compiler_error("Unexpected keyword in parsing",
                                         f"{instructions[pre_do_ip].loc}: Expected to be if.\n"
                                         f"Found: {instructions[pre_do_ip].word}")
                instructions[pre_else_ip].word.operand = DataTuple(typ=Type.INT, value=len(instructions) - pre_else_ip)
                word = Operation(operation=KEYWORD_LOOKUP[name], operand=None)
                op = Instruction(loc=location, word=word)
                keyword_stack.append(((len(instructions), token), pre_else_tuple))
            elif name == "end":
                if len(keyword_stack) < 1:
                    print_compiler_error("Lonely END found.",
                                         f"{location}: Could not find matching `do` before this `end`.")
                pre_end: tuple = keyword_stack.pop()
                pre_end_tuple: tuple = pre_end[0]
                pre_end_ip: int = pre_end_tuple[0]
                pre_end_token: Token = pre_end_tuple[1]
                # pre_end_keyword: Token = pre_end[1]
                pre_end_operation: Keyword = instructions[pre_end_ip].word.operation
                pre_do_tuple: tuple = pre_end[1]
                pre_do_ip: int = pre_do_tuple[0]
                pre_do_token: Token = pre_do_tuple[1]
                # pre_do_keyword: Token = pre_do[1]
                pre_do_operation: Keyword = instructions[pre_do_ip].word.operation

                assert pre_end_operation == Keyword.DO or pre_end_operation == Keyword.ELSE, "This is a bug in the parsing step"
                if pre_end_operation == Keyword.DO:
                    instructions[pre_end_ip].word.operand = DataTuple(typ=Type.INT, value=len(instructions) - pre_end_ip)
                    if pre_do_operation == Keyword.IF:
                        word = Operation(operation=KEYWORD_LOOKUP[name], operand=DataTuple(typ=Type.INT, value=1))
                        op = Instruction(loc=location, word=word)
                    elif pre_do_operation == Keyword.WHILE:
                        word = Operation(operation=KEYWORD_LOOKUP[name], operand=DataTuple(typ=Type.INT, value=pre_do_ip - len(instructions)))
                        op = Instruction(loc=location, word=word)
                    else:
                        print_compiler_error("Unexpected keyword in parsing",
                                             f"{instructions[pre_do_ip].loc}: Expected to be while or if.\n"
                                             f"Found: {instructions[pre_do_ip].word}")
                elif pre_end_operation == Keyword.ELSE:
                    assert instructions[pre_do_ip].word.operation == Keyword.DO, "This is a bug in the parsing step"

                    instructions[pre_end_ip].word.operand = DataTuple(typ=Type.INT, value=len(instructions) - pre_end_ip)

                    word = Operation(operation=KEYWORD_LOOKUP[name], operand=DataTuple(typ=Type.INT, value=1))
                    op = Instruction(loc=location, word=word)
                else:
                    assert False, "Unreachable - This is a bug in the parsing step. END will always come after DO or ELSE"

            else:
                print_compiler_error("Parsing of keyword token not implemented!",
                                     f"{token} can't be parsed yet.")
        elif (name.startswith("-") and name[1:].isdigit()) or name.isdigit():
            # print(f"Found integer {name}")
            word = Operation(operation=OpSet.PUSH_INT, operand=DataTuple(typ=Type.INT, value=int(name)))
            op = Instruction(loc=location, word=word)
        elif name.startswith("\"") and name.endswith("\""):
            # print(f"Found string {name}")
            lbl: str = "str" + str(len(strings))
            strings.append((lbl, name[1:-1]))
            word = Operation(operation=OpSet.PUSH_STR, operand=DataTuple(typ=Type.STR, value=lbl))
            op = Instruction(loc=location, word=word)
        else:
            is_mem = False
            for mem in global_memory:
                if name == mem.name:
                    word = Operation(operation=OpSet.PUSH_PTR, operand=DataTuple(typ=Type.PTR, value=name))
                    op = Instruction(loc=location, word=word)
                    is_mem = True
                    break
            if not is_mem:
                print_compiler_error("Unknown Token in Parsing",
                                     f"{token} can't be parsed.")
        instructions.append(op)
        ip += 1

    for i, op in enumerate(instructions):
        operation: Operation = op.word.operation
        operand: DataTuple = op.word.operand
        if operation in [Keyword.ELSE, Keyword.DO]:
            jump_labels.append(DataTuple(typ=Type.INT, value=operand.value + i + 1))
        elif operation == Keyword.END:
            if operand.value != 1:
                jump_labels.append(DataTuple(typ=Type.INT, value=operand.value + i + 1))

    return instructions, global_memory, strings, jump_labels
    # return [parse_op(op) for op in code]


def type_check_program(instructions: List[Instruction]):
    assert len(OpSet) == 24, "Not all Operations are handled in type checking"
    assert len(Keyword) == 6, "Not all Keywords are handled in type checking"
    assert len(Type) == 3, "Not all Type are handled in type checking"
    stack: List[Type] = []
    stack_checkpoint: List[tuple] = []
    keyword_stack: List[tuple] = []

    for ip, op in enumerate(instructions):
        location: Location = op.loc
        word: Operation = op.word
        operation: Union[Keyword, OpSet] = word.operation
        if operation in OpSet:
            if operation == OpSet.PUSH_INT:
                stack.append(Type.INT)
            elif operation == OpSet.PUSH_PTR:
                stack.append(Type.PTR)
            elif operation == OpSet.PUSH_STR:
                stack.append(Type.STR)
            elif operation == OpSet.DROP:
                if len(stack) < 1:
                    print_compiler_error("Not enough operands for operation",
                                         f"{location}: {operation} expected 1 argument, found {len(stack)} instead: {stack}\n")
                else:
                    stack.pop()
            elif operation == OpSet.OVER:
                if len(stack) < 2:
                    print_compiler_error("Not enough operands for operation",
                                         f"{location}: {operation} expected 2 arguments, found {len(stack)} instead: {stack}\n")
                else:
                    type2 = stack.pop()
                    type1 = stack.pop()
                    stack.append(type1)
                    stack.append(type2)
                    stack.append(type1)
            elif operation == OpSet.DUP:
                if len(stack) < 1:
                    print_compiler_error("Not enough operands for operation",
                                         f"{location}: {operation} expected 1 argument, found {len(stack)} instead: {stack}\n")
                else:
                    type1 = stack.pop()
                    stack.append(type1)
                    stack.append(type1)
            elif operation == OpSet.ROT:
                if len(stack) < 3:
                    print_compiler_error("Not enough operands for operation",
                                         f"{location}: {operation} expected 3 arguments, found {len(stack)} instead: {stack}\n")
                else:
                    # type3 type2 type1
                    type3 = stack.pop()
                    type2 = stack.pop()
                    type1 = stack.pop()
                    stack.append(type2)
                    stack.append(type1)
                    stack.append(type3)
                    # type2 type1 type3
            elif operation == OpSet.SWAP:
                if len(stack) < 2:
                    print_compiler_error("Not enough operands for operation",
                                         f"{location}: {operation} expected 2 arguments, found {len(stack)} instead: {stack}\n")
                else:
                    # type2 type1
                    type2 = stack.pop()
                    type1 = stack.pop()
                    stack.append(type2)
                    stack.append(type1)
                    # type2 type1
            elif operation in [OpSet.ADD_INT, OpSet.SUB_INT, OpSet.MUL_INT, OpSet.DIV_INT, OpSet.MOD_INT] or operation in [OpSet.LT, OpSet.GT, OpSet.EQ, OpSet.NEQ]:
                if len(stack) < 2:
                    print_compiler_error("Not enough operands for operation",
                                         f"{location}: {operation} expected 2 arguments, found {len(stack)} instead: {stack}\n")
                else:
                    type2 = stack.pop()
                    type1 = stack.pop()
                    if type1 == Type.INT and type2 == Type.INT:
                        stack.append(Type.INT)
                    else:
                        print_compiler_error("Wrong types for operation",
                                             f"{location}: {operation} expected 2 Integers, found {type1} and {type2} instead.")
            elif operation in [OpSet.ADD_PTR, OpSet.SUB_PTR]:
                if len(stack) < 2:
                    print_compiler_error("Not enough operands for operation",
                                         f"{location}: {operation} expected 2 arguments, found {len(stack)} instead: {stack}\n")
                else:
                    type2 = stack.pop()
                    type1 = stack.pop()
                    if type1 == Type.PTR and type2 == Type.INT:
                        stack.append(Type.PTR)
                    else:
                        print_compiler_error("Wrong types for operation",
                                             f"{location}: {operation} expected {[Type.PTR, Type.INT]}, found {type1} and {type2} instead.")
            elif operation == OpSet.SET_INT:
                # value variable !64
                if len(stack) < 2:
                    print_compiler_error("Not enough operands for operation",
                                         f"{location}: {operation} expected 2 arguments, found {len(stack)} instead: {stack}\n")
                else:
                    type2 = stack.pop()
                    type1 = stack.pop()
                    if type1 == Type.INT and type2 == Type.PTR:
                        pass
                    else:
                        print_compiler_error("Wrong types for operation",
                                             f"{location}: {operation} expected {[Type.INT, Type.PTR]}, found {type1} and {type2} instead.")
            elif operation == OpSet.GET_INT:
                # variable ?64
                if len(stack) < 1:
                    print_compiler_error("Not enough operands for operation",
                                         f"{location}: {operation} expected 1 argument, found {len(stack)} instead: {stack}\n")
                else:
                    type1 = stack.pop()
                    if type1 == Type.PTR:
                        stack.append(Type.INT)
                    else:
                        print_compiler_error("Wrong types for operation",
                                             f"{location}: {operation} expected {[Type.PTR]}, found {type1} instead.")
            elif operation == OpSet.PRINT_INT:
                if len(stack) < 1:
                    print_compiler_error("Not enough operands for operation",
                                         f"{location}: {operation} expected 1 argument, found {len(stack)} instead: {stack}\n")
                else:
                    type1 = stack.pop()
                    if type1 == Type.INT:
                        pass
                    else:
                        print_compiler_error("Wrong types for operation",
                                             f"{location}: {operation} expected {[Type.INT]}, found {type1} instead.")
            elif operation == OpSet.PRINT_STR:
                if len(stack) < 1:
                    print_compiler_error("Not enough operands for operation",
                                         f"{location}: {operation} expected 1 argument, found {len(stack)} instead: {stack}\n")
                else:
                    type1 = stack.pop()
                    if type1 == Type.STR:
                        pass
                    else:
                        print_compiler_error("Wrong types for operation",
                                             f"{location}: {operation} expected {[Type.INT]}, found {type1} instead.")
            else:
                assert False, f"Not implemented type checking for {operation} yet"
        elif operation in Keyword:
            # assert False, "Type checking Keyword not refactored yet"
            if operation == Keyword.MEMORY:
                pass
            elif operation in [Keyword.WHILE, Keyword.IF]:
                stack_checkpoint.append((ip, op, stack.copy()))
                keyword_stack.append((ip, operation))
            elif operation == Keyword.DO:
                pre_do = keyword_stack.pop()
                pre_do_ip = pre_do[0]
                pre_do_keyword = pre_do[1]
                if pre_do_keyword in [Keyword.WHILE, Keyword.IF]:
                    if len(stack) < 1:
                        print_compiler_error("Not enough operands for operation",
                                             f"{location}: {operation} expected 1 argument, found {len(stack)} instead: {stack}\n")
                    else:
                        type1 = stack.pop()
                        if type1 != Type.INT:
                            print_compiler_error("Wrong types for operation",
                                                 f"{location}: {operation} expected {[Type.INT]}, found {type1} instead.")
                else:
                    assert False, f"{pre_do_keyword} type-checking for Keyword.DO not implemented yet"
            elif operation == Keyword.ELSE:
                block: tuple = stack_checkpoint.pop()
                # block_ip = block[0]
                block_instr: Instruction = block[1]
                block_keyword: Operation = block_instr.word.operation
                block_stack = block[2]
                assert block_keyword == Keyword.IF, "This should never fail"
                stack_checkpoint.append((ip, op, stack.copy(), block_instr, block_stack.copy()))
                stack = block_stack.copy()
            elif operation == Keyword.END:
                block: tuple = stack_checkpoint.pop()
                # block_ip = block[0]
                block_instr: Instruction = block[1]
                block_keyword: Operation = block_instr.word.operation
                block_stack = block[2]
                if block_keyword == Keyword.WHILE or block_keyword == Keyword.IF:
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
                elif block_keyword == Keyword.ELSE:
                    if_block: Instruction = block[3]
                    if_keyword: Operation = if_block.word.operation
                    assert if_keyword == Keyword.IF, "This should never fail"
                    if_stack = block[4]
                    len1 = len(if_stack)
                    len2 = len(block_stack)
                    len3 = len(stack)
                    if not (len1 == len2 == len3):
                        print_compiler_error("Stack modification error in type checking",
                                             f"{if_block.loc}: All `{if_keyword.name}`-branches must result in the same stack.\n"
                                             f"\tStack before IF: {if_stack}\n"
                                             f"\tStack before ELSE: {block_stack}\n"
                                             f"\tStack before END: {stack}")
                    else:
                        for i in range(len1):
                            if not (stack[i] == block_stack[i] == if_stack[i]):
                                print_compiler_error("Stack modification error in type checking",
                                                     f"{if_block.loc}: All `{if_keyword.name}`-branches must result in the same stack.\n"
                                                     f"\tStack before IF: {if_stack}\n"
                                                     f"\tStack before ELSE: {block_stack}\n"
                                                     f"\tStack before END: {stack}")

                else:
                    assert False, f"{block_keyword} not implemented yet in type-checking Keyword.END"
            else:
                print(stack_checkpoint)
                assert False, f"{operation} type-checking not implemented yet"
        else:
            assert False, "Unreachable - This might be a bug in parsing"

    if len(stack) > 0:
        print_compiler_error("Unhandled Data on Stack",
                             f"There are still {len(stack)} item(s) on the stack:\n"
                             f"{stack}\n"
                             "Please make sure that the stack is empty after program is finished executing.",
                             ErrorType.STACK)
    return instructions


def evaluate_static_equations(instructions: List[Instruction]):
    # optimizes instructions like "10 2 1 3 * 4 5 * + - 2 * 7 - + 5 * 15"
    # by evaluating them in the pre-compiler phase and only pushing the result
    assert len(OpSet) == 24, "Make sure that `stack_op` in" \
                             "`evaluate_static_equations()` is up to date."
    # Last OP in our instruction set that is arithmetic
    # All Enum values less than that are available to be pre-evaluated
    new_code = []
    instr_stack = []
    for op in instructions:
        instr = op.word.operation
        if instr in [OpSet.PUSH_INT, OpSet.ADD_INT, OpSet.SUB_INT, OpSet.MUL_INT, OpSet.DIV_INT, OpSet.MOD_INT]:
            instr_stack.append(op)
        else:
            if len(instr_stack) > 0:
                push_op = 0
                math_op = 0
                for s in instr_stack:
                    if s.word.operation == OpSet.PUSH_INT:
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
                            stack_op = s.word.operation
                            # print(f"  {s}")
                            if stack_op == OpSet.PUSH_INT:
                                stack.append(s.word.operand.value)
                            elif stack_op == OpSet.ADD_INT:
                                a = stack.pop()
                                b = stack.pop()
                                stack.append(a + b)
                            elif stack_op == OpSet.SUB_INT:
                                a = stack.pop()
                                b = stack.pop()
                                stack.append(b - a)
                            elif stack_op == OpSet.MUL_INT:
                                a = stack.pop()
                                b = stack.pop()
                                stack.append(a * b)
                            elif stack_op == OpSet.DIV_INT:
                                a = stack.pop()
                                b = stack.pop()
                                stack.append(int(b / a))
                            elif stack_op == OpSet.MOD_INT:
                                a = stack.pop()
                                b = stack.pop()
                                stack.append(int(b % a))
                        result = stack.pop()
                        # print("Result: " + str(result))
                        last_instr = instr_stack[-1]
                        # print(f"Last instruction: {last_instr}")
                        new_instr = Instruction(loc=last_instr.loc, word=Operation(operation=OpSet.PUSH_INT, operand=DataTuple(typ=Type.INT, value=result)))
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


# Create Obj file: nasm -f win64 output.asm -o output.obj
# Link Obj together: golink /no /console /entry main output.obj MSVCRT.dll kernel32.dll
# Call Program: output.exe
def compile_code(program_name: str, instructions: List[Instruction], memory: List[Memory], strings: List[tuple], labels: List[DataTuple], opt_flags: dict):
    assert len(OpSet) == 24, "Not all OP can be compiled yet"
    assert len(Keyword) == 6, "Not all Keywords can be compiled yet"
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
            output.write(f"  {var.name} resb {var.size}\n")
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
            location: Location = op.loc
            word: Operation = op.word
            operation: Union[Keyword, OpSet] = word.operation
            operand: Optional[Type] = word.operand
            for jmp in labels:
                if i == jmp.value:
                    output.write(f"{label_name}_{i}:\n")
            output.write(f"; -- {op} --\n")
            if operation in OpSet:
                if operation == OpSet.SWAP:
                    output.write(f"  pop rbx\n")
                    output.write(f"  pop rax\n")
                    output.write(f"  push rbx\n")
                    output.write(f"  push rax\n")
                elif operation == OpSet.ROT:
                    output.write(f"  pop rcx\n")
                    output.write(f"  pop rbx\n")
                    output.write(f"  pop rax\n")
                    output.write(f"  push rbx\n")
                    output.write(f"  push rcx\n")
                    output.write(f"  push rax\n")
                elif operation == OpSet.DUP:
                    output.write(f"  pop rax\n")
                    output.write(f"  push rax\n")
                    output.write(f"  push rax\n")
                elif operation == OpSet.DROP:
                    output.write(f"  pop rax\n")
                elif operation == OpSet.OVER:
                    output.write(f"  pop rbx\n")
                    output.write(f"  pop rax\n")
                    output.write(f"  push rax\n")
                    output.write(f"  push rbx\n")
                    output.write(f"  push rax\n")
                elif operation == OpSet.PUSH_INT:
                    output.write(f"  mov rax, qword {operand.value}\n")
                    output.write(f"  push rax\n")
                elif operation == OpSet.PUSH_PTR:
                    output.write(f"  push {operand.value}\n")
                elif operation == OpSet.PUSH_STR:
                    output.write(f"  push {operand.value}\n")
                elif operation == OpSet.ADD_INT or operation == OpSet.ADD_PTR:
                    output.write("  pop rax\n")
                    output.write("  pop rbx\n")
                    output.write("  add rax, rbx\n")
                    output.write("  push rax\n")
                elif operation == OpSet.SUB_INT or operation == OpSet.SUB_PTR:
                    output.write("  pop rax\n")
                    output.write("  pop rbx\n")
                    output.write("  sub rbx, rax\n")
                    output.write("  push rbx\n")
                elif operation == OpSet.MUL_INT:
                    output.write("  pop rax\n")
                    output.write("  pop rbx\n")
                    output.write("  mul rbx\n")
                    output.write("  push rax\n")
                elif operation == OpSet.DIV_INT:
                    output.write("  pop rbx\n")
                    output.write("  pop rax\n")
                    output.write("  mov rdx, 0\n")
                    output.write("  cqo\n")
                    output.write("  idiv rbx\n")
                    output.write("  push rax\n")
                elif operation == OpSet.MOD_INT:
                    output.write("  pop rbx\n")
                    output.write("  pop rax\n")
                    output.write("  mov rdx, 0\n")
                    output.write("  div rbx\n")
                    output.write("  push rdx\n")
                elif operation == OpSet.PRINT_INT:
                    output.write("  pop rax\n")
                    output.write("  push rbp\n"
                                 "  mov rbp, rsp\n"
                                 "  sub rsp, 32\n")
                    output.write("  mov rdx, rax\n")
                    output.write("  lea rcx, [format_string]\n")
                    output.write("  call printf\n")
                    output.write("  add rsp, 32\n"
                                 "  pop rbp\n")
                elif operation == OpSet.PRINT_STR:
                    output.write("  pop rax\n")
                    output.write("  push rbp\n"
                                 "  mov rbp, rsp\n"
                                 "  sub rsp, 32\n")
                    output.write("  lea rcx, [rax]\n")
                    output.write("  call printf\n")
                    output.write("  add rsp, 32\n"
                                 "  pop rbp\n")
                elif operation == OpSet.EQ:
                    output.write("  pop rax\n")
                    output.write("  pop rbx\n")
                    output.write("  sub rbx, rax\n")
                    output.write("  lahf\n")
                    output.write("  shr rax, 14\n")
                    output.write("  and rax, 1\n")
                    output.write("  push rax\n")
                elif operation == OpSet.NEQ:
                    output.write("  pop rax\n")
                    output.write("  pop rbx\n")
                    output.write("  sub rbx, rax\n")
                    output.write("  lahf\n")
                    output.write("  shr rax, 14\n")
                    output.write("  not rax\n")
                    output.write("  and rax, 1\n")
                    output.write("  push rax\n")
                elif operation == OpSet.LT:
                    output.write("  pop rax\n")
                    output.write("  pop rbx\n")
                    output.write("  sub rbx, rax\n")
                    output.write("  lahf\n")
                    output.write("  shr rax, 15\n")
                    output.write("  and rax, 1\n")
                    output.write("  push rax\n")
                elif operation == OpSet.GT:
                    output.write("  pop rbx\n")
                    output.write("  pop rax\n")
                    output.write("  sub rbx, rax\n")
                    output.write("  lahf\n")
                    output.write("  shr rax, 15\n")
                    output.write("  and rax, 1\n")
                    output.write("  push rax\n")
                elif operation == OpSet.SET_INT:
                    # value var set_int
                    output.write("  pop rax\n")
                    # rax contains var
                    output.write("  pop rbx\n")
                    # rbx contains val
                    output.write("  mov [rax], rbx\n")
                elif operation == OpSet.GET_INT:
                    # var get_int
                    output.write(f"  pop rax\n")
                    # rax contains ptr to var
                    output.write(f"  mov rbx, [rax]\n")
                    # rbx contains value of var
                    output.write("  push rbx\n")
                else:
                    assert False, f"Unreachable - This means that an operation can't be compiled yet, namely: {operation}"
            elif operation in Keyword:
                if operation == Keyword.IF or operation == Keyword.WHILE:
                    pass
                elif operation == Keyword.ELSE:
                    assert op.word.operand.typ == Type.INT, "This could be a bug while parsing."
                    end_goal = i + op.word.operand.value + 1
                    output.write(f"  jmp {label_name}_{end_goal}\n")
                elif operation == Keyword.DO:
                    assert op.word.operand.typ == Type.INT, "This could be a bug while parsing."
                    end_goal = i + op.word.operand.value + 1
                    output.write("  pop rax\n")
                    output.write("  test rax, rax\n")
                    output.write(f"  jz {label_name}_{end_goal}\n")
                elif operation == Keyword.END:
                    assert op.word.operand.typ == Type.INT, "This could be a bug while parsing."
                    if op.word.operand.value == 1:
                        pass
                    else:
                        end_goal = i + op.word.operand.value + 1
                        output.write(f"  jmp {label_name}_{end_goal}\n")
            else:
                print_compiler_error(f"Compilation failed",
                                     f"at {location}: {operation} can't be compiled yet.")
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
           f"[-c | -d] [-o, -m, -a]\n" \
           f"       If you need more help, run `{program_name} -h`"


def main():
    # TODO: Add Strings, Arrays, Functions
    flags = ['-h', '-c', '-d', '-o', '-m', '-a']
    exec_flags = flags[1:3]
    optional_flags = flags[3:]
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

    instructions, memory, strings, labels = parse_instructions(code)
    instructions = type_check_program(instructions)
    instructions = evaluate_static_equations(instructions)

    if len(sys.argv) > 0:
        opt_args = sys.argv
        for opt in opt_args:
            if opt not in optional_flags:
                print_compiler_error("Unknown Flag",
                                     f"Found `{opt}`. For valid flags run `{program_name} -h`")
            else:
                opt_flags[opt] = True

    if run_flag == '-c':
        compile_code(input_file, instructions, memory, strings, labels, opt_flags)
    elif run_flag == '-d':
        for i, mem in enumerate(memory):
            print(f"{mem.name}: {mem.size} bytes")
        print("*" * 50)
        for i, op in enumerate(instructions):
            print(i, op.word)
        exit(1)
    else:
        print(f"Unknown flag `{run_flag}`")
        print(get_usage(program_name))
        exit(1)


if __name__ == '__main__':
    main()

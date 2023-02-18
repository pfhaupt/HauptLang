from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Union, Optional
import subprocess
import sys
import os

import colorama
from colorama import Fore, Style

colorama.init()

COMMENT_CHAR = "#"
PROC_IO_SEP = "->"
PARSE_COUNT = -1
GLOBAL_MEM_CAP = 640_008  # +8 because Null-PTR
LOCAL_MEM_CAP = 32_000


class Type(Enum):
    INT = auto()
    CHAR = auto()
    INT_PTR = auto()
    BYTE_PTR = auto()
    PROC_PTR = auto()


assert len(Type) == 5, "Not all Types are in the lookup table yet!"
TYPES_LOOKUP: dict[str, Type] = {
    "int": Type.INT,
    "byte": Type.CHAR,
    "int-ptr": Type.INT_PTR,
    "byte-ptr": Type.BYTE_PTR,
    "proc-ptr": Type.PROC_PTR,
}


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

    def __str__(self):
        result = ""
        if len(self.inputs) == 0:
            result = "None"
        else:
            for t in self.inputs:
                result += t.name.lower() + ", "
            result = result[:-2]
        result += f" {PROC_IO_SEP} "
        if len(self.outputs) == 0:
            result = "None"
        else:
            for t in self.outputs:
                result += t.name.lower() + ", "
            result = result[:-2]
        return f"{result}"


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
    PUSH_CHAR = auto()
    PRINT_INT = auto()
    PRINT_STR = auto()
    LT = auto()
    GT = auto()
    EQ = auto()
    NEQ = auto()
    SET_INT = auto()
    GET_INT = auto()
    SET_BYTE = auto()
    GET_BYTE = auto()
    CALL_PROC = auto()
    PREP_PROC = auto()
    RET_PROC = auto()
    PUSH_GLOBAL_MEM = auto()
    PUSH_LOCAL_MEM = auto()
    CAST_CHAR = auto()
    CAST_INT = auto()
    CAST_PTR = auto()


assert len(OpSet) == 35, "Not all Operations are in the lookup table yet!"
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
    "!int": OpSet.SET_INT,
    "?int": OpSet.GET_INT,
    "!byte": OpSet.SET_BYTE,
    "?byte": OpSet.GET_BYTE,
    "cast(byte)": OpSet.CAST_CHAR,
    "cast(int)": OpSet.CAST_INT,
    "cast(ptr)": OpSet.CAST_PTR,
}


class Keyword(Enum):
    IF = auto()
    ELSE = auto()
    ELIF = auto()
    DO = auto()
    END = auto()
    WHILE = auto()
    MEMORY = auto()
    PROC = auto()
    INCLUDE = auto()
    CONSTANT = auto()


KEYWORD_LOOKUP: dict[str, Keyword] = {
    "if": Keyword.IF,
    "else": Keyword.ELSE,
    "elif": Keyword.ELIF,
    "do": Keyword.DO,
    "end": Keyword.END,
    "while": Keyword.WHILE,
    "memory": Keyword.MEMORY,
    "proc": Keyword.PROC,
    "include": Keyword.INCLUDE,
    "const": Keyword.CONSTANT,
}
assert len(Keyword) == len(KEYWORD_LOOKUP), "Not all Keywords are in the lookup table yet!"


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
    typ: Type
    start: int
    size: int

    def __str__(self):
        return f"{self.name}: {self.typ} with a start of mem+{self.start} and a size of {self.size}"


@dataclass
class Procedure:
    name: str
    start: int
    end: int
    local_mem: List[Memory]
    mem_size: int
    signature: Signature
    type_checked: bool

    def __str__(self):
        return f"Procedure {self.name}: {self.signature}"


@dataclass
class Program:
    name: str
    instructions: List[Instruction]
    procedures: dict[str, Procedure]
    memory: List[Memory]
    strings: List[tuple]
    labels: List[DataTuple]
    included_files: List[Token]

    def __str__(self):
        result = f"Program name:\n  {self.name}\n"
        result += "*" * 25 + "\n"
        result += "Global Memory:\n"
        for mem in self.memory:
            result += "  " + str(mem) + "\n"
        result += "*" * 25 + "\n"
        result += "Included Files:\n"
        for included_file in self.included_files:
            result += "  " + str(included_file) + "\n"
        result += "*" * 25 + "\n"
        result += "Procedures:\n"
        for proc in self.procedures:
            result += "  " + str(self.procedures[proc]) + "\n"
        result += "*" * 25 + "\n"
        result += "Instructions:\n"
        for instr in self.instructions:
            result += "  " + str(instr.word) + "\n"
        result += "Strings: \n"
        for s in self.strings:
            result += "  " + str(s) + "\n"
        result += "*" * 25 + "\n"
        result += "Jump labels: \n"
        for lbl in self.labels:
            result += "  " + str(lbl) + "\n"
        return result

    def __len__(self):
        result = 0
        for attr in self.__dict__:
            if not attr.startswith("_"):
                result += 1
        return result


class ErrorType(Enum):
    NORMAL = auto()
    STACK = auto()


@dataclass
class KeywordParsingInfo:
    ip: int
    info: Token
    pre_info = None

    def __init__(self, i: int, inf: Token, pre=None):
        self.ip = i
        self.info = inf
        self.pre_info = pre  # always typeof KeywordParsingInfo


@dataclass
class Constant:
    location: Location
    name: str
    content: DataTuple


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
        if not char.isalnum() and char != '_':
            return False
    return True


def check_name_availability(variable: Token, code: List[Token], constants: List[Constant], global_memory: List[Memory], procedures: dict[str, Procedure]):
    variable_name: str = variable.name
    if variable_name in TYPES_LOOKUP or variable_name in KEYWORD_LOOKUP or variable_name in INTRINSIC_LOOKUP:
        print_compiler_error("Name conflict.",
                             f"{variable.loc}: `{variable_name}` is a reserved keyword.")
    for mem in global_memory:
        if variable_name == mem.name:
            print_compiler_error("Name conflict.",
                                 f"{variable.loc}: `{variable_name}` is already defined here:\n"
                                 f"  {mem.loc} with a size of {mem.size} bytes.")
    for const in constants:
        if variable_name == const.name:
            print_compiler_error("Name conflict.",
                                 f"{variable.loc}: `{variable_name}` is already defined here: {const.location}")
    for proc in procedures:
        if variable_name == proc:
            loc: Location = code[procedures[proc].start].loc
            print_compiler_error("Name conflict.",
                                 f"{variable.loc}: `{variable_name}` is already defined here: {loc}")


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
                if len(buffer) > 0:
                    parsed_token.append(buffer)
                    buffer = ""
                else:
                    buffer += char
                inside_str = True
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
                if char == COMMENT_CHAR:
                    break
                else:
                    buffer += char
        index += 1
    if inside_str:
        print_compiler_error("Unmatched quotes",
                             "There's a missing quote somewhere in the source code.")
    if len(buffer) > 0:
        parsed_token.append(buffer)
    for col, token in enumerate(parsed_token):
        result.append((col, token))
    return result


def load_from_file(file_path: str):
    with open(file_path, "r") as f:
        return [Token(loc=Location(file=file_path, row=row + 1, col=col + 1), name=token)
                for (row, line) in get_lines(f.readlines())
                for (col, token) in get_token_in_line(line)]


def offset_memory(size: int):
    global MEM_PTR
    tmp: int = MEM_PTR
    MEM_PTR += size
    return tmp


def parse_memory_block(code: List[Token], ip: int, global_memory: List[Memory], constants: List[Constant], procedures: dict[str, Procedure], mem_ptr: int):
    # Syntax: memory type name [optional byte_count] end
    if len(code) - ip < 4:
        # we need 4 words for each block
        # ip points at number 1
        print_compiler_error("Not enough words for memory block")
    ip += 1
    type_token: Token = code[ip]
    type_name: str = type_token.name
    if type_name not in TYPES_LOOKUP:
        print_compiler_error("Unexpected Token for `memory` definition",
                             f"{type_token.loc}: Undefined type `{type_name}`")
    memory_type: Type = TYPES_LOOKUP[type_name]

    ip += 1
    name_token: Token = code[ip]
    name_name: str = name_token.name
    check_name_availability(name_token, code, constants, global_memory, procedures)
    if not check_string(name_name):
        if not name_name[0].isalpha() and name_name[0] != "_":
            error_descr = f"{name_token.loc}: The name is only allowed to start with a letter or underscore\n" \
                          f"Found: {name_name}"
        else:
            error_descr = f"{name_token.loc}: The name can only consist of letters, numbers and `_`\n" \
                          f"Found: {name_name}"
        print_compiler_error("Invalid name for `memory` definition",
                             error_descr)

    memory_location: Location = name_token.loc
    memory_name: str = name_name

    ip += 1
    next_token: Token = code[ip]
    next_name: str = next_token.name
    memory_size: int = 1
    if not next_name == "end":
        # optimistic that there is an integer
        if not next_name.isdigit():
            is_const = False
            for const in constants:
                if const.name == next_name:
                    if const.content.typ != Type.INT:
                        print_compiler_error("BLA BLA ERROR ERROR")
                    is_const = True
                    memory_size = const.content.value
            if not is_const:
                print_compiler_error("Invalid size for memory region",
                                     f"{next_token.loc}: Expected number, got `{next_name}`")
        else:
            memory_size = int(next_name)
            if memory_size <= 0:
                print_compiler_error("Invalid size for memory region",
                                     f"{next_token.loc}: Expected number greater than zero, got {memory_size}.")
        ip += 1
    if memory_type == Type.CHAR:
        memory_size *= 1
    else:
        memory_size *= 8

    tmp_ptr = mem_ptr
    mem_ptr += memory_size
    return ip, mem_ptr, Memory(loc=memory_location, name=memory_name, typ=memory_type, size=memory_size, start=tmp_ptr)
    # Syntax: memory type name [optional byte_count] end

    # if len(code) - ip < 4:
    #     # we need 4 words for each block
    #     # ip points at number 1
    #     print_compiler_error("Not enough words for memory block")
    # ip += 1
    # variable: Token = code[ip]
    # variable_name: str = variable.name
    # if not check_string(variable_name):
    #     if not variable_name[0].isalpha() and variable_name[0] != "_":
    #         error_descr = f"{variable.loc}: The name is only allowed to start with a letter or underscore\n" \
    #                       f"Found: {variable_name}"
    #     else:
    #         error_descr = f"{variable.loc}: The name can only consist of letters, numbers and `_`\n" \
    #                       f"Found: {variable_name}"
    #     print_compiler_error("Invalid name inside memory",
    #                          error_descr)
    # elif variable_name in INTRINSIC_LOOKUP or variable_name in KEYWORD_LOOKUP:
    #     print_compiler_error("Invalid variable name",
    #                          f"{variable.loc}: Name is reserved.\n"
    #                          f"Found: {variable_name}")
    # else:
    #     check_name_availability(variable, code, constants, global_memory, procedures)
    #     # print("valid name")
    # ip += 1
    # byte_count: Token = code[ip]
    # byte_string: str = byte_count.name
    # size = -1
    # if byte_string.isdigit():
    #     size: int = int(byte_string)
    #     if size == 0:
    #         print_compiler_error("Invalid size inside memory",
    #                              f"{byte_count.loc}: Memory size is expected to be greater than 0.\n"
    #                              f"Found: {byte_string}")
    #     else:
    #         pass
    #         # print("valid size")
    # elif byte_string.startswith("-") and byte_string[1:].isdigit():
    #     print_compiler_error("Invalid size inside memory",
    #                          f"{byte_count.loc}: Memory size is expected to be greater than 0.\n"
    #                          f"Found: {byte_string}")
    # else:
    #     is_const = False
    #     for const in constants:
    #         if byte_string == const.name:
    #             if const.content.typ != Type.INT:
    #                 print_compiler_error("Wrong type for constant memory size",
    #                                      f"{byte_count.loc}: Expected type integer, got {const.content.typ}")
    #             size: int = int(const.content.value)
    #             is_const = True
    #             break
    #     if is_const:
    #         if size == 0:
    #             print_compiler_error("Invalid size inside memory",
    #                                  f"{byte_count.loc}: Memory size is expected to be greater than 0.\n"
    #                                  f"Found: {byte_string}")
    #         else:
    #             pass
    #     else:
    #         print_compiler_error("Invalid size inside memory",
    #                              f"{byte_count.loc}: Memory size is expected to be a number.\n"
    #                          f"Found: {byte_string}")
    # ip += 1
    # end_token: Token = code[ip]
    # end_name: str = end_token.name
    # if not end_name == "end":
    #     print_compiler_error("Unexpected word in parsing",
    #                          f"{end_token.loc} Expected `end`, found `{end_name}` instead.")
    # else:
    #     pass
    #     # print("valid end")
    #
    # return ip, Memory(loc=variable.loc, name=variable_name, content=DataTuple(typ=Type.INT, value=size))


def parse_procedure_signature(code: List[Token], ip: int, global_memory: List[Memory], constants: List[Constant], procedures: dict[str, Procedure]):
    if len(code) - ip < 4:
        # we need *at minimum* 4 words for a valid procedure: proc name do end
        # ip points at number 1
        print_compiler_error("Not enough words for procedure signature")
    ip += 1
    proc_name_token: Token = code[ip]
    proc_name = proc_name_token.name
    check_name_availability(proc_name_token, code, constants, global_memory, procedures)
    # print(proc_name_token)
    ip += 1
    next_token: Token = code[ip]
    inputs: List[Type] = []
    outputs: List[Type] = []
    found_sep = False
    sep_loc: Location = None

    types = "[" + ", ".join(TYPES_LOOKUP) + "]"
    while next_token.name != "do":
        if not found_sep:
            # haven't found PROC_IO_SEP yet, we are parsing inputs
            if next_token.name == PROC_IO_SEP:
                found_sep = True
                sep_loc = next_token.loc
            elif next_token.name in TYPES_LOOKUP:
                inputs.append(TYPES_LOOKUP[next_token.name])
            else:
                print_compiler_error("Invalid word found",
                                     f"{next_token.loc}: Found {next_token.name}, expected one of {types}")
        else:
            if next_token.name == PROC_IO_SEP:
                print_compiler_error("Unexpected word",
                                     f"{next_token.loc}: Expected only one separator in procedure signature, found two.\n"
                                     f"Separator already found here: {sep_loc}")
            elif next_token.name in TYPES_LOOKUP:
                outputs.append(TYPES_LOOKUP[next_token.name])
            else:
                print_compiler_error("Invalid word found",
                                     f"{next_token.loc}: Found {next_token.name}, expected one of {types}")

        ip += 1
        if ip >= len(code):
            print_compiler_error("Missing `do` for procedure",
                                 f"{proc_name_token.loc}: Missing `do` for procedure `{proc_name_token.name}`")
        next_token: Token = code[ip]

    proc_signature: Signature = Signature(inputs=inputs, outputs=outputs)
    ip -= 1
    return ip, proc_name, proc_signature


def parse_constant_block(code: List[Token], ip: int, global_memory: List[Memory], constants: List[Constant], procedures: dict[str, Procedure]):
    if len(code) - ip < 5:
        # we need *at minimum* 5 words for a valid constant: const type name value end
        # ip points at number 1
        print_compiler_error("Not enough words for constant definition")

    ip += 1
    type_token: Token = code[ip]
    type_name: str = type_token.name
    if type_name not in TYPES_LOOKUP:
        print_compiler_error("Unexpected Token for `const` definition",
                             f"{type_token.loc}: Undefined type `{type_name}`")
    if type_name == "ptr":
        print_compiler_error("Illegal type for `const` definition",
                             f"{type_token.loc}: Constants can't be of type Pointer.")
    const_type: Type = TYPES_LOOKUP[type_name]
    ip += 1
    name_token: Token = code[ip]
    name_name: str = name_token.name
    check_name_availability(name_token, code, constants, global_memory, procedures)
    if not check_string(name_name):
        if not name_name[0].isalpha() and name_name[0] != "_":
            error_descr = f"{name_token.loc}: The name is only allowed to start with a letter or underscore\n" \
                          f"Found: {name_name}"
        else:
            error_descr = f"{name_token.loc}: The name can only consist of letters, numbers and `_`\n" \
                          f"Found: {name_name}"
        print_compiler_error("Invalid name for `const` definition",
                             error_descr)

    const_location: Location = name_token.loc
    const_name: str = name_name

    if const_type == Type.INT:
        stack = []
        ip += 1
        next_token: Token = code[ip]
        while next_token.name != "end":
            name: str = next_token.name
            if (name.startswith("-") and name[1:].isdigit()) or name.isdigit():
                stack.append(DataTuple(typ=Type.INT, value=int(next_token.name)))
            elif name == "+":
                if len(stack) < 2:
                    print_compiler_error("Not enough operands for `const` evaluation",
                                         f"{next_token.loc}: `{next_token.name}` expected 2 operands.")
                op2 = stack.pop()
                op1 = stack.pop()
                stack.append(DataTuple(typ=Type.INT, value=op1.value + op2.value))
            elif name == "-":
                if len(stack) < 2:
                    print_compiler_error("Not enough operands for `const` evaluation",
                                         f"{next_token.loc}: `{next_token.name}` expected 2 operands.")
                op2 = stack.pop()
                op1 = stack.pop()
                stack.append(DataTuple(typ=Type.INT, value=op1.value - op2.value))
            elif name == "*":
                if len(stack) < 2:
                    print_compiler_error("Not enough operands for `const` evaluation",
                                         f"{next_token.loc}: `{next_token.name}` expected 2 operands.")
                op2 = stack.pop()
                op1 = stack.pop()
                stack.append(DataTuple(typ=Type.INT, value=op1.value * op2.value))
            elif name == "/":
                if len(stack) < 2:
                    print_compiler_error("Not enough operands for `const` evaluation",
                                         f"{next_token.loc}: `{next_token.name}` expected 2 operands.")
                op2 = stack.pop()
                op1 = stack.pop()
                if op2.value == 0:
                    print_compiler_error("Division by Zero in `const` evaluation",
                                         f"{next_token.loc}: Second operand was Zero.")
                stack.append(DataTuple(typ=Type.INT, value=int(op1.value / op2.value)))
            elif name == "%":
                if len(stack) < 2:
                    print_compiler_error("Not enough operands for `const` evaluation",
                                         f"{next_token.loc}: `{next_token.name}` expected 2 operands.")
                op2 = stack.pop()
                op1 = stack.pop()
                stack.append(DataTuple(typ=Type.INT, value=op1.value % op2.value))
            else:
                is_const = False
                for const in constants:
                    if name == const.name:
                        stack.append(const.content)
                        is_const = True
                        break
                if not is_const:
                    print_compiler_error("Undefined token in `const` evaluation",
                                         f"{next_token.loc}: `{next_token.name} can't be evaluated in `const`-blocks.")

            ip += 1
            if ip >= len(code):
                print_compiler_error("Could not find matching `end` for `const` block.",
                                     f"{next_token.loc}: Expected `end` keyword, reached end of file.")
            next_token: Token = code[ip]
        if len(stack) > 1:
            print_compiler_error("Too many elements in the stack after `const` evaluation",
                                 f"{next_token.loc}: Expected to have a single element on the stack, got {len(stack)}:\n"
                                 f"{stack}")
        elif len(stack) == 0:
            print_compiler_error("Not enough elements in the stack after `const` evaluation",
                                 f"{next_token.loc}: Expected to have a single element on the stack, got {len(stack)}:\n"
                                 f"{stack}")
        value = stack.pop().value
        const_value: int = value
        ip -= 1
    # elif const_type == Type.STR:
    #     ip += 1
    #     value_token: Token = code[ip]
    #     value_name: str = value_token.name
    #     if not (value_name.startswith("\"") and value_name.endswith("\"")):
    #         print_compiler_error("Invalid value for `const` definition",
    #                              f"{value_token.loc}: Expected string, found `{value_name}`")
    #     const_value: str = value_name[1:-1]
    else:
        assert False, "Unreachable"

    ip += 1
    end_token: Token = code[ip]
    end_name: str = end_token.name
    if not end_name == "end":
        print_compiler_error("Unexpected Token",
                             f"{end_token.loc}: Expected `end`, found `{end_name}`")

    return ip, Constant(location=const_location, name=const_name, content=DataTuple(typ=const_type, value=const_value))


def parse_instructions(code: List[Token], opt_flags, program_name: str, pre_included_files: List[Token]):
    global PARSE_COUNT
    PARSE_COUNT += 1
    global_memory: List[Memory] = []
    procedures: dict[str, Procedure] = {}
    strings: List[tuple] = []
    instructions: List[Instruction] = []
    jump_labels: List[DataTuple] = []
    included_files: List[Token] = pre_included_files
    constants: List[Constant] = []

    inside_proc: bool = False
    current_proc: str = ""

    keyword_stack: List[KeywordParsingInfo] = []

    GLOBAL_MEM_PTR = 8
    LOCAL_MEM_PTR = 0

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
                # print(GLOBAL_MEM_PTR, LOCAL_MEM_PTR)
                if inside_proc:
                    ip, LOCAL_MEM_PTR, memory_unit = parse_memory_block(code, ip, global_memory, constants, procedures, LOCAL_MEM_PTR)
                    if LOCAL_MEM_PTR > LOCAL_MEM_CAP:
                        print_compiler_error("Out of memory",
                                             f"{location}: Attempted to reserve more memory than available.\n"
                                             f"You can only use {LOCAL_MEM_CAP} bytes per procedure, but attempted to use {LOCAL_MEM_PTR}.")
                    assert current_proc in procedures, "This might be a bug in parsing"
                    word = Operation(operation=OpSet.NOP,
                                     operand=DataTuple(typ=Type.INT, value=len(procedures[current_proc].local_mem)))
                    op = Instruction(loc=location, word=word)

                    procedures[current_proc].local_mem.append(memory_unit)
                    procedures[current_proc].mem_size += memory_unit.size
                else:
                    LOCAL_MEM_PTR = 0
                    ip, GLOBAL_MEM_PTR, memory_unit = parse_memory_block(code, ip, global_memory, constants, procedures, GLOBAL_MEM_PTR)
                    if GLOBAL_MEM_PTR > GLOBAL_MEM_CAP:
                        print_compiler_error("Out of memory",
                                             f"{location}: Attempted to reserve more memory than available.\n"
                                             f"You can only use {GLOBAL_MEM_CAP - 8} bytes, but attempted to use {GLOBAL_MEM_PTR - 8}.")
                    word = Operation(operation=OpSet.NOP,
                                     operand=DataTuple(typ=Type.INT, value=len(global_memory)))
                    op = Instruction(loc=location, word=word)
                    global_memory.append(memory_unit)
            elif name == "proc":
                if inside_proc:
                    print_compiler_error("Unexpected word",
                                         f"{location}: You can't define procedures inside other procedures.")
                ip, proc_name, proc_signature = parse_procedure_signature(code, ip, global_memory, constants, procedures)

                proc_ip = len(instructions) + 1
                procedure: Procedure = Procedure(name=proc_name, signature=proc_signature, local_mem=[], mem_size=0,
                                                 start=proc_ip, end=proc_ip, type_checked=False)
                procedures[proc_name] = procedure
                inside_proc = True
                current_proc = proc_name
                info: KeywordParsingInfo = KeywordParsingInfo(len(instructions), token, None)
                keyword_stack.append(info)

                word = Operation(operation=KEYWORD_LOOKUP[name], operand=DataTuple(typ=Type.PROC_PTR, value=proc_name))
                op = Instruction(loc=location, word=word)
            elif name == "const":
                ip, const_unit = parse_constant_block(code, ip, global_memory, constants, procedures)
                constants.append(const_unit)
                word = Operation(operation=KEYWORD_LOOKUP[name], operand=DataTuple(typ=const_unit.content.typ, value=const_unit.content.value))
                op = Instruction(loc=location, word=word)
            elif name == "while" or name == "if":
                info: KeywordParsingInfo = KeywordParsingInfo(len(instructions), token, None)
                keyword_stack.append(info)

                word = Operation(operation=KEYWORD_LOOKUP[name], operand=None)
                op = Instruction(loc=location, word=word)
            elif name == "do":
                if len(keyword_stack) < 1:
                    print_compiler_error("Lonely DO found.",
                                         f"{location}: Could not find matching if, elif, while or proc before this `do`.")
                pre_do: KeywordParsingInfo = keyword_stack.pop()
                pre_do_ip: int = pre_do.ip
                pre_do_keyword: Token = pre_do.info
                if pre_do_keyword.name == "while" or pre_do_keyword.name == "if" or pre_do_keyword.name == "elif":
                    word = Operation(operation=KEYWORD_LOOKUP[name], operand=None)
                    info: KeywordParsingInfo = KeywordParsingInfo(len(instructions), token, pre_do)
                    keyword_stack.append(info)
                elif pre_do_keyword.name == "proc":
                    word = Operation(operation=OpSet.PREP_PROC, operand=DataTuple(typ=Type.PROC_PTR, value=current_proc))
                    info: KeywordParsingInfo = KeywordParsingInfo(len(instructions), token, pre_do)
                    keyword_stack.append(info)
                else:
                    word = None
                    print_compiler_error("Unexpected keyword in parsing",
                                         f"{pre_do_keyword.loc}: Expected to be while, if, elif or proc.\n"
                                         f"Found: {pre_do_keyword.name}")

                op = Instruction(loc=location, word=word)
            elif name == "else" or name == "elif":
                if len(keyword_stack) < 1:
                    print_compiler_error("Lonely ELSE found.",
                                         f"{location}: Could not find matching `do` before this `else`.")
                pre_else: KeywordParsingInfo = keyword_stack.pop()
                pre_else_ip: int = pre_else.ip
                pre_do: KeywordParsingInfo = pre_else.pre_info
                assert pre_do is not None, "Uhh"
                pre_do_ip: int = pre_do.ip
                pre_do_token: Token = pre_do.info
                instructions[pre_else_ip].word.operand = DataTuple(typ=Type.INT, value=len(instructions) - pre_else_ip)
                if pre_do_token.name == "if":
                    pass
                elif pre_do_token.name == "elif":
                    instructions[pre_do_ip].word.operand = DataTuple(typ=Type.INT, value=len(instructions) - pre_do_ip - 1)
                else:
                    print_compiler_error("Unexpected keyword in parsing",
                                         f"{instructions[pre_do_ip].loc}: Expected to be if.\n"
                                         f"Found: {instructions[pre_do_ip].word}")
                word = Operation(operation=KEYWORD_LOOKUP[name], operand=None)
                op = Instruction(loc=location, word=word)
                info: KeywordParsingInfo = KeywordParsingInfo(len(instructions), token, pre_else)
                keyword_stack.append(info)
            elif name == "end":
                if len(keyword_stack) < 1:
                    print_compiler_error("Lonely END found.",
                                         f"{location}: Could not find matching `do` before this `end`.")
                pre_end: KeywordParsingInfo = keyword_stack.pop()
                pre_end_ip: int = pre_end.ip
                pre_end_token: Token = pre_end.info
                pre_end_operation: Keyword = instructions[pre_end_ip].word.operation
                pre_do: KeywordParsingInfo = pre_end.pre_info
                assert pre_do is not None, "Uhh"
                pre_do_ip: int = pre_do.ip
                pre_do_token: Token = pre_do.info
                pre_do_operation: Keyword = instructions[pre_do_ip].word.operation

                assert pre_end_operation == OpSet.PREP_PROC or pre_end_operation == Keyword.DO or pre_end_operation == Keyword.ELSE, "This might be a bug in the parsing step"
                if pre_end_operation == OpSet.PREP_PROC:
                    assert current_proc in procedures, "This might be a bug in parsing"
                    procedures[current_proc].end = len(instructions)
                    word = Operation(operation=OpSet.RET_PROC,
                                     operand=DataTuple(typ=Type.INT, value=procedures[current_proc].mem_size))
                    op = Instruction(loc=location, word=word)
                    inside_proc = False
                elif pre_end_operation == Keyword.DO:
                    instructions[pre_end_ip].word.operand = DataTuple(typ=Type.INT,
                                                                      value=len(instructions) - pre_end_ip)
                    if pre_do_operation == Keyword.IF:
                        word = Operation(operation=KEYWORD_LOOKUP[name], operand=DataTuple(typ=Type.INT, value=1))
                        op = Instruction(loc=location, word=word)
                    elif pre_do_operation == Keyword.ELIF:
                        word = Operation(operation=KEYWORD_LOOKUP[name], operand=DataTuple(typ=Type.INT, value=1))
                        op = Instruction(loc=location, word=word)
                        instructions[pre_do_ip].word.operand = DataTuple(typ=Type.INT,
                                                                         value=len(instructions) - pre_do_ip)
                    elif pre_do_operation == Keyword.WHILE:
                        word = Operation(operation=KEYWORD_LOOKUP[name],
                                         operand=DataTuple(typ=Type.INT, value=pre_do_ip - len(instructions)))
                        op = Instruction(loc=location, word=word)
                    else:
                        print_compiler_error("Unexpected keyword in parsing",
                                             f"{instructions[pre_do_ip].loc}: Expected to be while or if.\n"
                                             f"Found: {instructions[pre_do_ip].word}")
                elif pre_end_operation == Keyword.ELSE:
                    assert instructions[pre_do_ip].word.operation == Keyword.DO, "This is a bug in the parsing step"

                    instructions[pre_end_ip].word.operand = DataTuple(typ=Type.INT,
                                                                      value=len(instructions) - pre_end_ip)

                    word = Operation(operation=KEYWORD_LOOKUP[name], operand=DataTuple(typ=Type.INT, value=1))
                    op = Instruction(loc=location, word=word)
                else:
                    assert False, "Unreachable - This is a bug in the parsing step. END will always come after DO or ELSE"
            elif name == "include":
                if not (ip + 1 < len(code)):
                    print_compiler_error("Unexpected EOF",
                                         f"{location}: Expected a name for `include` keyword, found end of file.")
                ip += 1
                include_token: Token = code[ip]
                include_name: str = include_token.name
                if not (include_name.startswith("\"") and include_name.endswith("\"")):
                    print_compiler_error("Unexpected token in parsing",
                                         f"{location}: The file name is expected to be surrounded with quotes.\n"
                                         f"Found: {include_name}")
                include_name: str = include_name[1:-1]
                include_file_path: str = os.path.join(os.getcwd(), include_name)
                file_exists: bool = os.path.exists(include_file_path)
                if not file_exists:
                    print_compiler_error("File does not exist",
                                         f"{include_token.loc}: `include` could not find file `{include_name}`.")
                else:
                    if not include_name.endswith(".hpt"):
                        print_compiler_error("Unexpected file ending",
                                             f"{location}: Attempted to include non-Haupt program `{include_name}`.")
                    if include_name == program_name:
                        print_compiler_error("Attempted to include original program",
                                             f"{include_token.loc}: You can't include the original program into itself.")
                    for included in included_files:
                        if include_name == included.name:
                            print_compiler_error("File inclusion error",
                                                 f"{location}: Already imported file `{include_name}` here: {included.loc}")
                    if not opt_flags["-m"]:
                        print("[INFO] Including " + include_name)
                    included_files.append(Token(loc=include_token.loc, name=include_name))
                    include_program: Program = parse_source_code(include_name, opt_flags, program_name, included_files)
                    include_program.name = include_name
                    # print(include_program)
                    # Append all instructions
                    instructions.extend(include_program.instructions)
                    # Append all procedures
                    for proc in include_program.procedures:
                        if proc in procedures:
                            other_loc = include_program.instructions[include_program.procedures[proc].start].loc
                            def_loc = instructions[procedures[proc].start].loc
                            print_compiler_error("Procedure redefinition in `include`",
                                                 f"{location}: `include` failed. Reason: \n"
                                                 f"{other_loc}: {proc} is already defined here: {def_loc}")
                        else:
                            procedures[proc] = include_program.procedures[proc]
                    # Append the memory
                    for mem in include_program.memory:
                        for g_mem in global_memory:
                            if mem.name == g_mem.name:
                                other_loc = mem.loc
                                def_loc = g_mem.loc
                                print_compiler_error("Procedure redefinition in `include`",
                                                     f"{location}: `include` failed. Reason: \n"
                                                     f"{other_loc}: {mem.name} is already defined here: {def_loc}")

                        global_memory.append(mem)
                    # Append the labels
                    jump_labels.extend(include_program.labels)
                    # Append the strings
                    for include_string in include_program.strings:
                        for existing_string in strings:
                            assert include_string[0] != existing_string[0], "This might be a bug caused by wrong PARSE_COUNT"
                        strings.append(include_string)
                    # len - 1 because we don't care about the name
                    assert len(include_program) - 1 == 6, "Not all Program attributes are included in `include` parsing yet"
                ip += 1
                continue
            else:
                print_compiler_error("Parsing of keyword token not implemented!",
                                     f"{token} can't be parsed yet.")
        elif (name.startswith("-") and name[1:].isdigit()) or name.isdigit():
            # print(f"Found integer {name}")
            word = Operation(operation=OpSet.PUSH_INT, operand=DataTuple(typ=Type.INT, value=int(name)))
            op = Instruction(loc=location, word=word)
        elif name.startswith("\"") and name.endswith("\""):
            # print(f"Found string {name}")
            name = name[1:-1]
            exists = False
            lbl: str = "str_" + str(PARSE_COUNT) + "_" + str(len(strings))
            for l, s in strings:
                if s == name:
                    exists = True
                    lbl = l
                    break
            if not exists:
                strings.append((lbl, name))
            word = Operation(operation=OpSet.PUSH_STR, operand=DataTuple(typ=Type.BYTE_PTR, value=lbl))
            op = Instruction(loc=location, word=word)
        elif name.startswith("\'") and name.endswith("\'"):
            name = name[1:-1]
            assert len(name) == 1
            word = Operation(operation=OpSet.PUSH_CHAR, operand=DataTuple(typ=Type.CHAR, value=ord(name)))
            op = Instruction(loc=location, word=word)
        else:
            is_global_mem = False
            for mem in global_memory:
                if name == mem.name:
                    if mem.typ == Type.INT:
                        word = Operation(operation=OpSet.PUSH_GLOBAL_MEM, operand=DataTuple(typ=Type.INT_PTR, value=mem.start))
                    elif mem.typ == Type.CHAR:
                        word = Operation(operation=OpSet.PUSH_GLOBAL_MEM, operand=DataTuple(typ=Type.BYTE_PTR, value=mem.start))
                    else:
                        assert False, "Unreachable"
                    op = Instruction(loc=location, word=word)
                    is_global_mem = True
                    break

            is_local_mem = False
            if inside_proc:
                assert current_proc is not None, "This might be a bug in parsing"
                curr_proc: Procedure = procedures[current_proc]
                for mem in curr_proc.local_mem:
                    if name == mem.name:
                        if mem.typ == Type.INT:
                            word = Operation(operation=OpSet.PUSH_LOCAL_MEM, operand=DataTuple(typ=Type.INT_PTR, value=mem.start))
                        elif mem.typ == Type.CHAR:
                            word = Operation(operation=OpSet.PUSH_LOCAL_MEM, operand=DataTuple(typ=Type.BYTE_PTR, value=mem.start))
                        else:
                            assert False, "Unreachable"
                        op = Instruction(loc=location, word=word)
                        is_local_mem = True
                        break

            is_proc = False
            if name in procedures:
                jmp_ip = procedures[name].start
                word = Operation(operation=OpSet.CALL_PROC, operand=DataTuple(typ=Type.INT, value=jmp_ip))
                op = Instruction(loc=location, word=word)
                is_proc = True

            is_const = False
            for const in constants:
                if name == const.name:
                    typ = const.content.typ
                    value = const.content.value
                    if typ == Type.INT:
                        operation = OpSet.PUSH_INT
                    # elif typ == Type.STR:
                    #     operation = OpSet.PUSH_STR
                    #     exists = False
                    #     lbl: str = "str_" + str(PARSE_COUNT) + "_" + str(len(strings))
                    #     for l, s in strings:
                    #         if s == value:
                    #             exists = True
                    #             lbl = l
                    #             break
                    #     if not exists:
                    #         strings.append((lbl, value))
                    #     value = lbl
                    else:
                        assert False, "This might be a bug in parsing const blocks"
                    word = Operation(operation=operation, operand=DataTuple(typ=typ, value=value))
                    op = Instruction(loc=location, word=word)
                    is_const = True
                    break

            if not is_global_mem and not is_local_mem and not is_proc and not is_const:
                print_compiler_error("Unknown Token in Parsing",
                                     f"{token} can't be parsed.")
        instructions.append(op)
        ip += 1

    for i, op in enumerate(instructions):
        operation: Operation = op.word.operation
        operand: DataTuple = op.word.operand
        if operation in [Keyword.ELSE, Keyword.DO, Keyword.ELIF]:
            jump_labels.append(DataTuple(typ=Type.INT, value=operand.value + i + 1))
        elif operation == Keyword.END:
            if operand.value != 1:
                jump_labels.append(DataTuple(typ=Type.INT, value=operand.value + i + 1))

    for proc in procedures:
        # print(proc)
        jump_labels.append(DataTuple(typ=Type.INT, value=procedures[proc].start))
        jump_labels.append(DataTuple(typ=Type.INT, value=procedures[proc].end + 1))
    #
    # for i, lbl in enumerate(jump_labels):
    #     print(i, lbl)
    # exit(1)
    return Program(name=program_name, instructions=instructions, procedures=procedures, memory=global_memory, strings=strings, labels=jump_labels, included_files=included_files)
    # return [parse_op(op) for op in code]


def type_check_program(instructions: List[Instruction], procedures: dict[str, Procedure]):
    assert len(Type) == 5, "Not all Type are handled in type checking"
    stack: List[Type] = []
    stack_checkpoint: List[tuple] = []
    keyword_stack: List[tuple] = []

    for ip, op in enumerate(instructions):
        location: Location = op.loc
        word: Operation = op.word
        operation: Union[Keyword, OpSet] = word.operation
        if operation in OpSet:
            if operation == OpSet.NOP:
                pass
            elif operation == OpSet.PUSH_INT:
                stack.append(Type.INT)
            elif operation == OpSet.PUSH_CHAR:
                stack.append(Type.CHAR)
            elif operation == OpSet.PUSH_PTR:
                stack.append(word.operand.typ)
            elif operation == OpSet.PUSH_STR:
                stack.append(Type.INT)
                stack.append(Type.BYTE_PTR)
            elif operation == OpSet.DROP:
                if len(stack) < 1:
                    print_compiler_error("Not enough operands for operation",
                                         f"{location}: {operation} expected 1 argument, found {len(stack)} instead: {stack}")
                else:
                    stack.pop()
            elif operation == OpSet.OVER:
                if len(stack) < 2:
                    print_compiler_error("Not enough operands for operation",
                                         f"{location}: {operation} expected 2 arguments, found {len(stack)} instead: {stack}")
                else:
                    type2 = stack.pop()
                    type1 = stack.pop()
                    stack.append(type1)
                    stack.append(type2)
                    stack.append(type1)
            elif operation == OpSet.DUP:
                if len(stack) < 1:
                    print_compiler_error("Not enough operands for operation",
                                         f"{location}: {operation} expected 1 argument, found {len(stack)} instead: {stack}")
                else:
                    type1 = stack.pop()
                    stack.append(type1)
                    stack.append(type1)
            elif operation == OpSet.ROT:
                if len(stack) < 3:
                    print_compiler_error("Not enough operands for operation",
                                         f"{location}: {operation} expected 3 arguments, found {len(stack)} instead: {stack}")
                else:
                    # type1 type2 type3
                    type3 = stack.pop()
                    type2 = stack.pop()
                    type1 = stack.pop()
                    stack.append(type2)
                    stack.append(type3)
                    stack.append(type1)
                    # type2 type3 type1
            elif operation == OpSet.SWAP:
                if len(stack) < 2:
                    print_compiler_error("Not enough operands for operation",
                                         f"{location}: {operation} expected 2 arguments, found {len(stack)} instead: {stack}")
                else:
                    # type1 type2
                    type2 = stack.pop()
                    type1 = stack.pop()
                    stack.append(type2)
                    stack.append(type1)
                    # type2 type1
            elif operation in [OpSet.ADD_INT, OpSet.SUB_INT, OpSet.MUL_INT, OpSet.DIV_INT,
                               OpSet.MOD_INT] or operation in [OpSet.LT, OpSet.GT, OpSet.EQ, OpSet.NEQ]:
                if len(stack) < 2:
                    print_compiler_error("Not enough operands for operation",
                                         f"{location}: {operation} expected 2 arguments, found {len(stack)} instead: {stack}")
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
                                         f"{location}: {operation} expected 2 arguments, found {len(stack)} instead: {stack}")
                else:
                    type2 = stack.pop()
                    type1 = stack.pop()
                    if type1 in [Type.BYTE_PTR, Type.INT_PTR] and type2 == Type.INT:
                        stack.append(type1)
                    else:
                        print_compiler_error("Wrong types for operation",
                                             f"{location}: {operation} expected PTR, INT, found {type1} and {type2} instead.")
            elif operation == OpSet.SET_INT:
                # value variable !int
                if len(stack) < 2:
                    print_compiler_error("Not enough operands for operation",
                                         f"{location}: {operation} expected 2 arguments, found {len(stack)} instead: {stack}")
                else:
                    # type1 type2
                    type2: Type = stack.pop()
                    type1: Type = stack.pop()
                    if type1 == Type.INT and type2 == Type.INT_PTR:
                        pass
                    else:
                        print_compiler_error("Wrong types for operation",
                                             f"{location}: {operation} expected [INT, INT-PTR], found {type1} and {type2} instead.")
            elif operation == OpSet.SET_BYTE:
                # value variable !byte
                if len(stack) < 2:
                    print_compiler_error("Not enough operands for operation",
                                         f"{location}: {operation} expected 2 arguments, found {len(stack)} instead: {stack}")
                else:
                    # type1 type2
                    type2: Type = stack.pop()
                    type1: Type = stack.pop()
                    if type2 == Type.BYTE_PTR and type1 == Type.CHAR:
                        pass
                    else:
                        print_compiler_error("Wrong types for operation",
                                             f"{location}: {operation} expected [CHAR, CHAR-PTR], found {type1} and {type2} instead.")
            elif operation == OpSet.GET_INT:
                # variable ?64
                if len(stack) < 1:
                    print_compiler_error("Not enough operands for operation",
                                         f"{location}: {operation} expected 1 argument, found {len(stack)} instead: {stack}")
                else:
                    type1 = stack.pop()
                    if type1 == Type.INT_PTR:
                        stack.append(Type.INT)
                    else:
                        print_compiler_error("Wrong types for operation",
                                             f"{location}: {operation} expected INT-PTR, found {type1} instead.")
            elif operation == OpSet.GET_BYTE:
                # variable ?64
                if len(stack) < 1:
                    print_compiler_error("Not enough operands for operation",
                                         f"{location}: {operation} expected 1 argument, found {len(stack)} instead: {stack}")
                else:
                    type1 = stack.pop()
                    if type1 == Type.BYTE_PTR:
                        stack.append(Type.CHAR)
                    else:
                        print_compiler_error("Wrong types for operation",
                                             f"{location}: {operation} expected CHAR-PTR, found {type1} instead.")
            elif operation == OpSet.PRINT_INT:
                if len(stack) < 1:
                    print_compiler_error("Not enough operands for operation",
                                         f"{location}: {operation} expected 1 argument, found {len(stack)} instead: {stack}")
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
                                         f"{location}: {operation} expected 1 argument, found {len(stack)} instead: {stack}")
                else:
                    type1 = stack.pop()
                    if type1 == Type.BYTE_PTR:
                        pass
                    else:
                        print_compiler_error("Wrong types for operation",
                                             f"{location}: {operation} expected CHAR-PTR, found {type1} instead.")
            elif operation == OpSet.PREP_PROC:
                pass
                # assert False, f"{operation} type-checking not implemented yet"
            elif operation == OpSet.RET_PROC:
                pre_ret_tuple: tuple = stack_checkpoint.pop()
                proc_instruction: Instruction = pre_ret_tuple[1]
                proc_keyword: Keyword = proc_instruction.word.operation
                assert proc_keyword == Keyword.PROC, "This might be a bug in parsing"
                proc_name: str = proc_instruction.word.operand.value
                assert proc_name in procedures, "This might be a bug in parsing"
                procedure: Procedure = procedures[proc_name]
                proc_signature: Signature = procedure.signature
                proc_outputs: List[Type] = proc_signature.outputs
                pre_proc_stack: List[Type] = pre_ret_tuple[2]
                for out in proc_outputs:
                    pre_proc_stack.append(out)
                if len(pre_proc_stack) != len(stack):
                    print_compiler_error("Signature mismatch",
                                         f"{proc_instruction.loc}: Procedure does not match the provided signature.")
                else:
                    for i in range(len(stack)):
                        if stack[i] != pre_proc_stack[i]:
                            print_compiler_error("Signature mismatch",
                                                 f"{proc_instruction.loc}: Procedure does not match the provided signature.")
                    for _ in proc_outputs:
                        pre_proc_stack.pop()
                    stack = pre_proc_stack.copy()
                # pass
                # assert False, f"{operation} type-checking not implemented yet"
            elif operation == OpSet.CALL_PROC:
                proc_id: int = op.word.operand.value
                assert proc_id < len(instructions), "This might be a bug in parsing"
                proc_name: str = instructions[proc_id].word.operand.value
                assert proc_name in procedures, "This might be a bug in parsing"
                procedure: Procedure = procedures[proc_name]
                assert procedure.type_checked, "This might be a bug in parsing"
                expected_inputs: List[Type] = procedure.signature.inputs
                expected_inputs_as_str: str = str(expected_inputs)
                if len(stack) < len(expected_inputs):
                    print_compiler_error("Not enough operands for operation",
                                         f"{location}: `{proc_name}` expected {len(expected_inputs)} arguments, found {len(stack)} instead:\n"
                                         f"Expected stack: {expected_inputs_as_str}\n"
                                         f"Actual stack:   {stack}")
                else:
                    # at least arg_count types on stack
                    # int ptr int -> int
                    len1 = len(stack)
                    len2 = len(expected_inputs)
                    for i in range(len2):
                        type_on_stack = stack[len1 - i - 1]
                        expected_type = expected_inputs[len2 - i - 1]
                        if type_on_stack != expected_type:
                            print_compiler_error("Type mismatch in type checking",
                                                 f"{location}: Attempted to call procedure {proc_name} with the wrong argument types:\n"
                                                 f"Expected stack: {expected_inputs_as_str}\n"
                                                 f"Actual stack:   {stack}")
                    for _ in range(len2):
                        stack.pop()
                    proc_outputs: List[Type] = procedure.signature.outputs
                    for out in proc_outputs:
                        stack.append(out)
                # pass
                # assert False, f"{operation} type-checking not implemented yet"
            elif operation == OpSet.PUSH_GLOBAL_MEM or operation == OpSet.PUSH_LOCAL_MEM:
                stack.append(word.operand.typ)
            elif operation == OpSet.CAST_PTR:
                if len(stack) < 1:
                    print_compiler_error("Not enough operands for operation",
                                         f"{location}: {operation} expected 1 argument, found {len(stack)} instead: {stack}")
                type1: Type = stack.pop()
                if type1 == Type.INT:
                    stack.append(Type.INT_PTR)
                elif type1 == Type.CHAR:
                    stack.append(Type.BYTE_PTR)
                else:
                    print_compiler_error("Wrong types for operation",
                                         f"{location}: {operation} expected CHAR or INT, found {type1} instead.")
            elif operation == OpSet.CAST_CHAR:
                if len(stack) < 1:
                    print_compiler_error("Not enough operands for operation",
                                         f"{location}: {operation} expected 1 argument, found {len(stack)} instead: {stack}")
                type1: Type = stack.pop()
                stack.append(Type.CHAR)
            elif operation == OpSet.CAST_INT:
                if len(stack) < 1:
                    print_compiler_error("Not enough operands for operation",
                                         f"{location}: {operation} expected 1 argument, found {len(stack)} instead: {stack}")
                type1: Type = stack.pop()
                stack.append(Type.INT)
            else:
                assert False, f"Not implemented type checking for {operation} yet"
        elif operation in Keyword:
            # assert False, "Type checking Keyword not refactored yet"
            if operation == Keyword.MEMORY or operation == Keyword.CONSTANT:
                pass
            elif operation in [Keyword.WHILE, Keyword.IF]:
                stack_checkpoint.append((ip, op, stack.copy()))
                keyword_stack.append((ip, operation))
            elif operation == Keyword.DO:
                pre_do = keyword_stack.pop()
                pre_do_ip = pre_do[0]
                pre_do_keyword = pre_do[1]
                if pre_do_keyword in [Keyword.WHILE, Keyword.IF, Keyword.ELIF]:
                    if len(stack) < 1:
                        print_compiler_error("Not enough operands for operation",
                                             f"{location}: {operation} expected 1 argument, found {len(stack)} instead: {stack}")
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
                assert block_keyword == Keyword.IF or block_keyword == Keyword.ELIF, "This might be a bug in parsing"
                if block_keyword == Keyword.IF:
                    stack_checkpoint.append((ip, op, stack.copy(), block_instr, block_stack.copy()))
                    stack = block_stack.copy()
                elif block_keyword == Keyword.ELIF:
                    new_stack = block[4]
                    stack_checkpoint.append((ip, op, stack.copy(), block_instr, block_stack.copy()))
                    stack = new_stack.copy()
            elif operation == Keyword.ELIF:
                block: tuple = stack_checkpoint.pop()
                block_instr: Instruction = block[1]
                block_stack = block[2]
                block_keyword: Keyword = block_instr.word.operation
                if block_keyword == Keyword.IF:
                    stack_checkpoint.append((ip, op, stack.copy(), block_instr, block_stack.copy()))
                    stack = block_stack.copy()
                    keyword_stack.append((ip, operation))
                elif block_keyword == Keyword.ELIF:
                    if len(stack) != len(block_stack):
                        print_compiler_error("Stack modification error in type checking",
                                             f"{location}: The branch before this `elif` does not match the expected stack.\n"
                                             f"\tExpected stack: {block_stack}\n"
                                             f"\tActual stack: {stack}")
                    else:
                        for i in range(len(stack)):
                            if stack[i] != block_stack[i]:
                                print_compiler_error("Stack modification error in type checking",
                                                     f"{block_instr.loc}: The branch before this `elif` does not match the expected stack.\n"
                                                     f"\tExpected stack: {block_stack}\n"
                                                     f"\tActual stack:   {stack}")
                    new_stack = block[4]
                    stack_checkpoint.append((ip, op, stack.copy(), block_instr, new_stack.copy()))
                    stack = new_stack.copy()
                    keyword_stack.append((ip, operation))

                else:
                    assert False, f"This might be a bug in parsing, Unexpected keyword: {block_keyword} in type-checking ELIF"
            elif operation == Keyword.END:
                block: tuple = stack_checkpoint.pop()
                # block_ip = block[0]
                block_instr: Instruction = block[1]
                block_keyword: Operation = block_instr.word.operation
                block_stack = block[2]
                if block_keyword == Keyword.WHILE:
                    if len(stack) < len(block_stack):
                        print_compiler_error("Stack modification error in type checking",
                                             f"{block_instr.loc}: `{block_keyword.name}` is not allowed to decrease the size of the stack.")
                    else:
                        pre_stack_len = len(block_stack)
                        post_stack_len = len(stack)
                        stack_len = pre_stack_len if pre_stack_len < post_stack_len else post_stack_len
                        for i in range(stack_len):
                            if stack[i] != block_stack[i]:
                                print_compiler_error("Stack modification error in type checking",
                                                     f"{block_instr.loc}: `{block_keyword.name}` is not allowed to modify the types on the stack.\n"
                                                     f"Before: {block_stack}\n"
                                                     f"After: {stack}")
                elif block_keyword == Keyword.IF or block_keyword == Keyword.ELSE or block_keyword == Keyword.ELIF:
                    len1 = len(block_stack)
                    len2 = len(stack)
                    if len1 != len2:
                        print_compiler_error("Stack modification error in type checking",
                                             f"{location}: The branch before this `end` does not match the expected stack.\n"
                                             f"\tExpected stack: {block_stack}\n"
                                             f"\tActual stack:   {stack}")
                    else:
                        for i in range(len1):
                            if not (stack[i] == block_stack[i]):
                                print_compiler_error("Stack modification error in type checking",
                                                     f"{location}: The branch before this `end` does not match the expected stack.\n"
                                                     f"\tExpected stack: {block_stack}\n"
                                                     f"\tActual stack:   {stack}")
                else:
                    assert False, f"{block_keyword} not implemented yet in type-checking Keyword.END"
            elif operation == Keyword.PROC:
                proc_name: str = op.word.operand.value
                assert proc_name in procedures, "This might be a bug in parsing"
                stack_checkpoint.append((ip, op, stack.copy()))
                procedure: Procedure = procedures[proc_name]
                proc_inputs: List[Type] = procedure.signature.inputs
                for t in proc_inputs:
                    stack.append(t)
                procedure.type_checked = True
            else:
                # print(stack_checkpoint)
                assert False, f"{operation} type-checking not implemented yet"
        else:
            assert False, "Unreachable - This might be a bug in parsing"
        # print(operation, stack)

    if len(stack) > 0:
        print_compiler_error("Unhandled Data on Stack",
                             f"There are still {len(stack)} item(s) on the stack:\n"
                             f"{stack}\n"
                             "Please make sure that the stack is empty after program is finished executing.",
                             ErrorType.STACK)


# noinspection PyUnreachableCode
def evaluate_static_equations(instructions: List[Instruction]):
    # TODO: Optimizing anything inside here breaks jump-labels
    return instructions
    # optimizes instructions like "10 2 1 3 * 4 5 * + - 2 * 7 - + 5 * 15"
    # by evaluating them in the pre-compiler phase and only pushing the result
    assert len(OpSet) == 31, "Make sure that `stack_op` in" \
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
                        new_instr = Instruction(loc=last_instr.loc, word=Operation(operation=OpSet.PUSH_INT,
                                                                                   operand=DataTuple(typ=Type.INT,
                                                                                                     value=result)))
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
def compile_program(program: Program, opt_flags: dict):
    program_name: str = program.name
    instructions: List[Instruction] = program.instructions
    procedures: dict[str, Procedure] = program.procedures
    memory: List[Memory] = program.memory
    strings: List[tuple] = program.strings
    labels: List[DataTuple] = program.labels
    silenced = opt_flags['-m']
    optimized = opt_flags['-o']
    keep_asm = opt_flags['-a']
    run_program = opt_flags['-r']

    name = program_name.replace(".hpt", "")

    last_proc: str = None

    label_name = "instr"
    with open(name + ".tmp", "w") as output:
        output.write(f"  ; Generated code for {program_name}\n")
        output.write("default rel\n")
        output.write("\n")
        output.write("segment .text\n"
                     "  global main\n"
                     "  extern ExitProcess\n"
                     "  extern printf\n")
        output.write("\n")
        output.write("main:\n")
        output.write("  mov rax, ret_stack_end\n")
        output.write("  mov [ret_stack_rsp], rax\n")

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
                if operation == OpSet.NOP:
                    pass
                elif operation == OpSet.SWAP:
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
                elif operation == OpSet.PUSH_CHAR:
                    output.write(f"  mov rax, qword {operand.value}\n")
                    output.write(f"  push rax\n")
                elif operation == OpSet.PUSH_PTR:
                    assert False, "This might be a bug in parsing"
                elif operation == OpSet.PUSH_STR:
                    for string in strings:
                        if string[0] == operand.value:
                            output.write(f"  mov rax, qword {len(string[1])}\n")
                            output.write(f"  push rax\n")
                            break
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
                elif operation == OpSet.SET_BYTE:
                    # value var set_int
                    output.write("  pop rax\n")
                    # rax contains var
                    output.write("  pop rbx\n")
                    # rbx contains val
                    output.write("  mov [rax], bl\n")
                elif operation == OpSet.GET_BYTE:
                    # var get_int
                    output.write("    pop rax\n")
                    # rax contains ptr to var
                    output.write("    xor rbx, rbx\n")
                    output.write("    mov bl, [rax]\n")
                    # rbx contains value of var
                    output.write("    push rbx\n")
                elif operation == OpSet.CALL_PROC:
                    proc_start = op.word.operand.value
                    output.write("  mov rax, rsp\n")
                    output.write("  mov rsp, [ret_stack_rsp]\n")
                    output.write(f"  call {label_name}_{proc_start}\n")
                    output.write("  mov [ret_stack_rsp], rsp\n")
                    output.write("  mov rsp, rax\n")
                elif operation == OpSet.PREP_PROC:
                    last_proc = operand.value
                    n = procedures[last_proc].mem_size
                    padded = n + (8 - n % 8)
                    output.write(f"  sub rsp, {padded}\n")
                    output.write("  mov [ret_stack_rsp], rsp\n")
                    output.write("  mov rsp, rax\n")
                elif operation == OpSet.RET_PROC:
                    assert last_proc is not None, "This might be a bug in parsing"
                    n = procedures[last_proc].mem_size
                    last_proc = None
                    padded = n + (8 - n % 8)
                    output.write("  mov rax, rsp\n")
                    output.write("  mov rsp, [ret_stack_rsp]\n")
                    output.write(f"  add rsp, {padded}\n")
                    output.write("  ret\n")
                elif operation == OpSet.PUSH_LOCAL_MEM:
                    output.write("  mov rax, [ret_stack_rsp]\n")
                    output.write(f"  add rax, {operand.value}\n")
                    output.write("  push rax\n")
                elif operation == OpSet.PUSH_GLOBAL_MEM:
                    output.write(f"  push qword mem+{operand.value}\n")
                elif operation in [OpSet.CAST_PTR, OpSet.CAST_CHAR, OpSet.CAST_INT]:
                    pass
                else:
                    assert False, f"Unreachable - This means that an operation can't be compiled yet, namely: {operation}"
            elif operation in Keyword:
                if operation == Keyword.IF or operation == Keyword.WHILE or operation == Keyword.CONSTANT:
                    pass
                elif operation == Keyword.ELSE or operation == Keyword.ELIF:
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
                elif operation == Keyword.PROC:
                    proc = procedures[operand.value]
                    output.write(f"  jmp {label_name}_{proc.end + 1}\n")
                elif operation == Keyword.ELIF:
                    assert False, "Compiling elif not implemented"
                else:
                    assert False, f"Unreachable - This means that a keyword can't be compiled yet, namely: {operation}"
            else:
                print_compiler_error(f"Compilation failed",
                                     f"at {location}: {operation} can't be compiled yet.")
        output.write("\n")
        output.write(f"{label_name}_{len(instructions)}:\n")
        output.write("  xor rcx, rcx\n")
        output.write("  call ExitProcess\n")
        output.write("\n")
        output.write("segment .data\n")
        output.write("  format_string db \"%lld\", 0\n")
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
        output.write("  ret_stack_rsp resb 8\n")
        output.write("  ret_stack resb 4194304\n")
        output.write("  ret_stack_end resb 8\n")
        output.write(f"  mem resb {GLOBAL_MEM_CAP}\n")
        # for var in memory:
        #     output.write(f"  {var.name} resb {var.content.value}\n")
        output.write("\n")

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
    if run_program:
        call_cmd([f"{name}.exe"], silenced)
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
        case '-r':
            return "Runs the program after compilation."
        case '-unsafe':
            return "Disables type checking."
        case _:
            return "[No description]"


def get_usage(program_name):
    return f"Usage: {program_name} [-h] <input.hpt> " \
           f"[-c | -d] [-o, -m, -a, -r, -unsafe]\n" \
           f"       If you need more help, run `{program_name} -h`"


def main():
    flags = ['-h', '-c', '-d', '-o', '-m', '-a', '-r', '-unsafe']
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
        print("Supported flags:")
        for flag in flags:
            print(f"    {flag}: " + get_help(flag))
        exit(0)
    if len(sys.argv) < 2:
        print_compiler_error("Not enough parameters!",
                             f"{get_usage(program_name)}\n")
    input_file, sys.argv = shift(sys.argv)
    if not input_file.startswith("./"):
        input_file = "./" + input_file
    if not input_file.endswith(".hpt"):
        print_compiler_error(f"File {input_file} does not end with `.hpt`!",
                             get_usage(program_name))

    run_flag, sys.argv = shift(sys.argv)
    if run_flag not in exec_flags:
        print_compiler_error("Third Parameter has to be an execution flag!",
                             get_usage(program_name))

    if len(sys.argv) > 0:
        opt_args = sys.argv
        for opt in opt_args:
            if opt not in optional_flags:
                print_compiler_error("Unknown Flag",
                                     f"Found `{opt}`. For valid flags run `{program_name} -h`")
            else:
                opt_flags[opt] = True

    main_program: Program = parse_source_code(input_file, opt_flags, program_name)

    if run_flag == '-c':
        compile_program(main_program, opt_flags)
    elif run_flag == '-d':
        for i, mem in enumerate(main_program.memory):
            print(mem)
        print("*" * 50)
        for i, op in enumerate(main_program.instructions):
            print(i, op.word)
        exit(1)
    else:
        print(f"Unknown flag `{run_flag}`")
        print(get_usage(program_name))
        exit(1)


def parse_source_code(input_file, opt_flags, program_name, included_files=None):
    if included_files is None:
        included_files = []
    code: List[Token] = []
    try:
        code = load_from_file(input_file)
    except FileNotFoundError:
        print_compiler_error(f"File `{input_file} does not exist!",
                             get_usage(program_name))
    main_program: Program = parse_instructions(code, opt_flags, input_file, included_files)
    if not opt_flags['-unsafe']:
        type_check_program(instructions=main_program.instructions, procedures=main_program.procedures)
    main_program.instructions = evaluate_static_equations(instructions=main_program.instructions)

    return main_program


if __name__ == '__main__':
    main()

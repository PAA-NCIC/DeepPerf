from sets import Set
from inst import Inst
from dumper import dump
import sys
import logging

ops = dict()

def check_operand_types(inst):
    operand_types = ""
    operands = inst.operands().split();
    for operand in operands:
        key = operand[0]
        if key == 'R': # Register
            value = operand[1:]
            if value == 'Z' or value == 'N' or value == 'M' or \
                value == 'P' or float(value).is_integer():
                operand_types += 'R'
            else:
                return 'X'
        elif key == 'P': # Predicate
            value = operand[1:]
            if float(value).is_integer():
                operand_types += 'P'
            else:
                return 'X'
        elif key == 'c': # Constant memory
            operand_types += 'C'
        elif key == '[': # Memory
            operand_types += 'M'
        elif key == 'S': # Special register
            operand_types += 'S'
        else:
            if len(operand) >= 2 and (operand[0:2] == "0x" or operand[0:3] == "-0x"): # Hex immediate
                operand_types += 'I'
            elif float(operand).is_integer(): # Immediate value
                operand_types += 'I'
            else:
                return 'X'
    if inst.op() not in ops:
        ops[inst.op()] = set()
        ops[inst.op()].add(operand_types)
        return operand_types
    elif inst.op() in ops and operand_types not in ops[inst.op()]:
        ops[inst.op()].add(operand_types)
        return operand_types
    else:
        return 'X'

def change(inst, origin):
    if inst.op() != origin.op():
        return -1
    elif inst.modifier() != origin.modifier():
        return -2
    else:
        inst_operands = inst.operands().split()
        origin_operands = origin.operands().split()
        for i in range(len(origin_operands)):
            if (inst_operands[i] != origin_operands[i]):
                return i
        return -3
            
if __name__ == "__main__":
    logging.basicConfig(filename = sys.argv[3], level = logging.INFO)
    logging.debug("argv[1]: Disassemble file")
    logging.debug("argv[2]: Arch")
    logging.debug("argv[3]: Output file")
    logging.debug("argv[4]: Instruction limit (default 100)")
    sass = sys.argv[1]
    arch = sys.argv[2]
    if len(sys.argv) >= 5:
        limit = sys.argv[4]
    else:
        limit = 100
    count = 0;
    version = int(arch.split("_")[1])
    with open(sys.argv[1]) as f:
        for line in f:
            pos = []
            count += 1
            if count == limit:
                break
            line_split = line.split()
            # Construct instruction structure
            origin = Inst(line_split)
            # Find the 64-bit encodings
            base = int(origin.enc(), 16)
            origin_operand_types = check_operand_types(origin)
            if len(origin.operands()) and origin_operand_types.find('X') == -1:
                pp = [[] for i in range(len(origin_operand_types))]
                logging.info(origin.op() + " " + origin.modifier())
                logging.info("0b{:064b}".format(base) + ": " + origin.operands())
                for i in range(0, 64):
                    mask = 2**i
                    newcode = base ^ mask
                    # Disassemble the new code
                    dump_file = dump("0x{:016x}".format(newcode), arch)
                    if dump_file and dump_file.find("?") == -1 and dump_file.find("error") == -1:
                        line = dump_file.split("\n")
                        if version < 40:
                            line_inst = line[1].split();
                        else:
                            line_inst = line[5].split();
                        # [0]: header info, [1] instruction part
                        line_inst.pop(0)
                        inst = Inst(line_inst, raw = version > 40)
                        pos = change(inst, origin) 
                        if pos >= 0:
                            pp[pos].append(i)
                            logging.info("0b{:064b}".format(newcode) + ": " + inst.operands())
                logging.info("Operand combination types: %s", origin_operand_types)
                for i in range(0, len(pp)):
                    logging.info("Operand type: %s", origin_operand_types[i])
                    logging.info("Encoding: %s", pp[i])

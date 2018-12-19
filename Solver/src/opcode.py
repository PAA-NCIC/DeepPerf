from inst import Inst
from dumper import dump
import sys
import logging

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
    with open(sass) as f:
        for line in f:
            pos = []
            bits = 0x0
            count += 1
            if count == limit:
                break
            line_split = line.split()
            # Construct instruction structure
            origin = Inst(line_split)
            # Find the 64-bit encodings
            base = int(origin.enc(), 16)
            # Bit by bit xor, observe whether opcode changes and guess what this bit represent
            for i in range(0, 64):
                mask = 2**i
                newcode = base ^ mask
                # Disassemble the new code
                dump_file = dump("0x{:016x}".format(newcode), arch)
                # Compare the disassemble to check which field changes: opcode, operand or modifer
                if dump_file and dump_file.find("?") == -1 and dump_file.find("error") == -1:
                    line = dump_file.split("\n")
                    if version < 40:
                        line_inst = line[1].split();
                    else:
                        line_inst = line[5].split();
                    # [0]: header info, [1] instruction part
                    line_inst.pop(0)
                    # Parse the new generated disassembly
                    inst = Inst(line_inst, raw = version > 40)
                    # If opcode is changed, then this bit represent opcode, we find it!
                    # LDG and TEX are the same instructions in fact 
                    # RED and ATOM are the same instruction
                    if inst.op() != origin.op() and not i in pos and not \
                    (inst.op() == "LDG" and origin.op() == "TEX") and not \
                    (inst.op() == "TEX" and origin.op() =="LDG") and not \
                    (inst.op() == "RED" and origin.op() == "ATOM") and not \
                    (inst.op() == "ATOM" and origin.op() == "RED"):
                        logging.info("Opcode changes: %s => %s when bit [%d] is flipped from [%d]", \
                            origin.op(), inst.op(), i, (base >> i) & 0x1)
                        bits = bits | (((base >> i) & 0x1) << i)
                        pos.append(i)
            if len(pos) > 0:
                logging.info("0b{:064b}".format(bits) + ": %s opcode bits %s: ", origin.op(), pos);

from dumper import dump
import logging

def enumerate(base, pos, arch):
    version = int(arch.split("_")[1])
    for i in range(1 << len(pos)):
        bits = 0x0
        enc = base
        # Expresss i in binary
        for j in range(len(pos)):
            bits = (((i >> j) & 0x1) << pos[j]) | bits
            enc = enc & (~(1 << pos[j]))
        dump_file = dump("0x{:016x}".format(enc | bits), arch)
        if dump_file and dump_file.find("?") == -1 and dump_file.find("error") == -1 and dump_file.find("INVALID") == -1:
            line = dump_file.split("\n")
            if version < 40:
                line_inst = line[1].split();
            else:
                line_inst = line[5].split();
            line_inst.pop(0)
            logging.info("0b{:064b}".format(bits) + ": " + " ".join(line_inst))

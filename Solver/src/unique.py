from sets import Set
from inst import Inst
import subprocess 
import sys

if __name__ == "__main__":
    opset = Set([])
    with open(sys.argv[1]) as f:
        for line in f:
            field = line.split()
            inst = Inst(field, False)
            if not inst.op() in opset:
                opset.add(inst.op())
                sys.stdout.write(line)

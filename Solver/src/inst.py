from sets import Set

class Inst:
    def __init__(self, inst, raw = True):
        # Fetech binary encoding
        if raw == True: # From cuobjdump
            self.__enc = inst[-2]
            inst.pop(-1)
            inst.pop(-1)
            inst.pop(-1)
        else: # From nvdisasm
            self.__enc = ""

        if inst[0] == '{':  # Check dual issue
            inst.pop(0)
            self.__pred = ""
        if inst[0].find('@') != -1:  # Check predicate, such as @P0
            self.__pred = inst.pop(0)

        # Remove semicolon of zero operand field instruction such as "RRO;" 
        ops = inst.pop(0).replace(";", "")
        # Fetech opcode
        self.__op = ops.split(".")[0]
        # Split opcode
        self.__modifier = ops.split(".")[1:]
        # Fetech operands and remove ; and ,
        self.__operands = ' '.join(inst).replace(";", "").replace(",", "").replace("-","").replace("|","")

    def op(self):
        return str(self.__op)

    def modifier(self):
        return str(self.__modifier)

    def enc(self):
        return str(self.__enc)

    def operands(self):
        return str(self.__operands)

    def pred(self):
        return str(self.__pred)

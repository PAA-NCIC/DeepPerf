import os
import struct

def arch2mode(arch):
    return arch.replace("_", "").upper()

def dump(newcode, arch):
    version = arch.split("_")[1]
    if version < 40:
        tmp_bin = "/tmp/tmp_dumper.bin"
        fout = open(tmp_bin, "wb")
        fout.write(struct.pack("<Q", int(newcode, 16)))
        fout.close()
        cmd = "nvdisasm -b {0} {1} 2>&1".format(arch2mode(arch), tmp_bin)
        tmp_read = os.popen(cmd).read()
        rmfile = "rm {0}".format(tmp_bin)
        os.system(rmfile)
        return tmp_read
    else:
        tmp_cubin = "data/" + arch + "/" + arch + ".cubin"
        f = open(tmp_cubin,'rb+')  
        f.seek(904)
        f.write(struct.pack('Q', int(newcode, 16)))
        f.close()
        cmd = "cuobjdump -arch {0} -sass {1} 2>&1".format(arch, tmp_cubin)
        tmp_read = os.popen(cmd).read()
        return tmp_read

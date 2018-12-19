
# Cracking GPU ISA Encodings

## Output

* Bit positions of opcodes
* Bit positions of operands for different operand type
* Bit positions of modifiers for each instruction

## How to run the workflow?

The workflow is composed of four stages:

1. Generate PTX code->`./bin/generate_disassemble [arch]`
    * Generate PTX code (.ptx) in ptxgen directory and compile PTX to cubin; 
    * Disassemble cubins to sass files, which feed into the following three solvers;
    * Each line of sass files looks like this:
    
    `/∗0048∗/ IADD R0, R2, R0; /∗0x4800000000201c03∗/`
    
2. Opcode solver->`./bin/opcode [arch]`
    * Probe 64-bit binary code of sass files by flipping each bit and observe whether opcodes change;
    
3. Modifer solver->`./bin/modifier [arch]`
    * Probe 64-bit binary code of sass files by flipping each bit and observe whether modifiers change;
    * Enuermerate bits on all modifier positions to generate all the modifiers;
    
4. Operand solver->`./bin/operand [arch]`
    * Probe 64-bit binary code of sass files by flipping each bit and observe whether operands change;
    * Operand type: R: Register, S: Special Register, I: Immediate, C: constant[][], M: Memory, P: Predicate;

5. Allowed values for `[arch]` options:  'sm_30','sm_32','sm_35','sm_37','sm_50','sm_52','sm_53','sm_60','sm_61','sm_62'.

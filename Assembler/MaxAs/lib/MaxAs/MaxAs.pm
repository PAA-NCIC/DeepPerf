package MaxAs::MaxAs;

require 5.10.0;

use strict;
use Data::Dumper;
use MaxAs::MaxAsGrammar;
use File::Spec;
use Carp;
use POSIX;
use List::Util qw[min max];

our $VERSION = '1.06';

# these ops need to be converted from absolute addresses to relative in the sass output by cuobjdump
my %relOffset  = map { $_ => 1 } qw(BRA SSY CAL PBK PCNT);

# these ops use absolute addresses
my %absOffset  = map { $_ => 1 } qw(JCAL);

my %jumpOp     = (%relOffset, %absOffset);

# These instructions use r0 but do not write to r0
my %noDest     = map { $_ => 1 } qw(ST STG STS STL RED);

# Map register slots to reuse control codes
my %reuseSlots = (r8 => 1, r20 => 2, r39 => 4);

# break the registers down into source and destination categories for the scheduler
my %srcReg   = map { $_ => 1 } qw(r8 r20 r39 p12 p29 p39 X);
my %destReg  = map { $_ => 1 } qw(r0 p0 p3 p45 p48 CC);
my %regops   = (%srcReg, %destReg);
my @itypes   = qw(class lat rlat tput dual);

# init resource usage
my $activeWarp = 1;
my $scheduler = 2;
my $warpSize = 32;
my $bankWidth = 4;
my $maxThreads = 1024;
my $maxSharedMem = 49152;
my $maxReg = 65536;

my $AnalyzeRe = qr'^[\t ]*<ANALYZE_BLOCK>(.*?)^\s*</ANALYZE_BLOCK>\n?'ms;

sub Occupancy
{
    my ($fileName) = @_;

    print "Occupancy\n";

    local $/ = "\n";
    open my $fh, "<", $fileName or die "Cannot open: ", $fileName;
    my $usedThreads = <$fh>;
    chomp $usedThreads;
    $usedThreads =~ s/threads=//g;

    my $usedSharedMem = <$fh>;
    chomp $usedSharedMem;
    $usedSharedMem =~ s/shared=//g;

    my $usedReg = <$fh>;
    chomp $usedReg;
    $usedReg =~ s/regs=//g;

    my $activeBlock = min(ceil($maxThreads / $usedThreads),
      ceil($maxSharedMem / $usedSharedMem), ceil(ceil($maxReg / $usedReg) / $usedThreads));
    $activeWarp = $activeBlock * ceil($usedThreads / $warpSize);

    print "Active Blocks: ", $activeBlock, "\n";
    print "Active Warps: ", $activeWarp, "\n\n\n";
    close $fh;
}

sub LongestPath
{
    my ($instructs) = @_;

    # calculate longest path
    my @path;
    foreach my $i (0 .. $#$instructs)
    {
        push @path, 0;
    }

    foreach my $i (0 .. $#$instructs)
    {
        my $instruct = $instructs->[$i];
        foreach my $child (@{$instruct->{children}}) {
            my $ins = @$child[0];
            my $weight = @$child[1];
            $path[$ins] = $weight + $path[$i] if $weight + $path[$i] > $path[$ins];
        }
    }

    my $longestPath = 0;
    foreach my $i (0 .. $#$instructs)
    {
        $longestPath = $path[$i] if $path[$i] > $longestPath;
    }

    return $longestPath;
}

sub PreprocessBlock
{
    my ($analyzeBlock) = @_;
    my ($lineNum, @instructs, @branches, %labels);

    # push first dummy instruct
    push @instructs, {dualCnt=>0, nodual=>1};

    # Preprocess instructions
    foreach my $line (split "\n", $analyzeBlock)
    {
        # keep track of line nums in the physical file
        $lineNum++;

        next unless preProcessLine($line);

        # Match an instruction
        if (my $inst = processAsmLine($line, $lineNum))
        {
            # Save us from crashing the display driver
            die "It is illegal to set a Read-After-Write dependency on a memory store op (store ops don't write to a register)\n$inst->{inst}\n"
                if exists $noDest{$inst->{op}} && ($inst->{ctrl} & 0x000e0) != 0x000e0;

            # track branches/jumps/calls/etc for label remapping
            push @branches, @instructs+0 if exists $jumpOp{$inst->{op}};

            # add the op name and full instruction text
            push @instructs, $inst;
        }
        # Match a label
        elsif ($line =~ m'^([a-zA-Z]\w*):')
        {
            # map the label name to the index of the instruction about to be inserted
            $labels{$1} = @instructs+0;
        }
        else
        {
            die "badly formed line at $lineNum: $line\n";
        }
    }

    # remap labels
    foreach my $i (@branches)
    {
        if (exists $relOffset{$instructs[$i]{op}})
        {
            $instructs[$i]{inst} =~ s/(\w+);$/sprintf '0x%06x;', (($labels{$1} - $i - 1) * 8) & 0xffffff/e;
        }
        else
        {
            $instructs[$i]{inst} =~ s/(\w+);$/sprintf '0x%06x;', ($labels{$1} * 8) & 0xffffff/e;
        }
    }

    return @instructs;
}

sub CalculateEfficiency
{
    my ($instructs) = @_;
    print "Instructions\tDispatches\tEcompute\tEcmp\tEmem\n";

    # Analyze efficiency
    foreach my $i (0 .. $#$instructs) 
    {
        my $instruct = $instructs->[$i];
        $instruct->{dualCnt} = 0;
        $instruct->{nodual} = 1;

        next unless $i != 0;

        my ($op, $inst) = @{$instructs->[$i]}{qw(op inst)};
  
        foreach my $gram (@{$grammar{$op}})
        {
            my $capData = parseInstruct($inst, $gram) or next;
            @{$instruct}{@itypes} = @{$gram->{type}}{@itypes};
            $instruct->{dualCnt} = $instruct->{dual} ? 1 : 0;

            # Handle P2R and R2P specially
            if ($instruct->{op} =~ m'P2R|R2P' && $capData->{i20w7})
            {
                # These instructions can't be dual issued
                $instruct->{nodual} = 1;
            }

            # For pascal and maxwell
            my $dispatches = 1;
            my $instructType = $gram->{type}; 
            if ($instructType->{class} eq 'x32' || $instructType->{class} eq 's2r' ||
                $instructType->{class} eq 'qtr' || $instructType->{class} eq 'rro' ||
                $instructType->{class} eq 'vote')
            {
                my $units = $instructType->{units};
                $instruct->{efficiency} = 1 / ceil(($dispatches * $warpSize) / $units);
            }
            elsif ($instructType->{class} eq 'shift' || $instructType->{class} eq 'cmp')
            {
                my $units = $instructType->{units};
                my $tput = $instructType->{tput};
                $instruct->{efficiency} = 1 / (ceil(($dispatches * $warpSize) / $units) * $tput);
            }
            elsif ($instructType->{class} eq 'mem')
            {
                my $units = $instructType->{units};
                my $memType = $capData->{type};
                my $issue = 1;
                # vector instruction
                if ($memType =~ s/^\.//g)
                {
                    $issue *= $memType / $warpSize;
                }
                # TODO(keren): cache instruction ???
                if ($op eq 'LDG')
                {
                    $issue = 1;
                }
                $instruct->{efficiency} = 1 / ceil(($dispatches * $warpSize) / $units * $issue);
            }
            else
            {
                die "No such instruct type: ", Dumper($instruct);
            }
            if ($i > 1 and $instruct->{dual}) {
                my ($prevOp) = @{$instructs->[$i - 1]}{qw(op)};
                foreach my $prevGram (@{$grammar{$prevOp}})
                {
                    #TODO(keren): not noly same class, but also same units
                    if ($prevGram->{type}->{class} eq $instructType->{class}) 
                    {
                        #TODO(keren): ceil?
                        $instructs->[$i - 1]->{efficiency} = $instruct->{efficiency} =
                          1 / 2 * $instruct->{efficiency};
                    }
                }
            }
        }
    }
    foreach my $i (0 .. $#$instructs) 
    {
        next unless $i > 0;

        my $instruct = $instructs->[$i];
        my ($op, $inst) = @{$instructs->[$i]}{qw(op inst)};
  
        foreach my $gram (@{$grammar{$op}})
        {
            my $dispatches = 1;
            print "\t" if $instruct->{dualCnt};
            print $inst, "\t", $dispatches, "\t";
            my $instructType = $gram->{type}; 
            if ($instructType->{class} eq 'x32' || $instructType->{class} eq 's2r' ||
                $instructType->{class} eq 'qtr' || $instructType->{class} eq 'rro' ||
                $instructType->{class} eq 'vote')
            {
                print $instruct->{efficiency}, "\t0\t0";
            }
            elsif ($instructType->{class} eq 'shift' || $instructType->{class} eq 'cmp')
            {
                print "0\t", $instruct->{efficiency}, "\t0";
            }
            elsif ($instructType->{class} eq 'mem')
            {
                # TODO(keren): simulate
                print "0\t0\t", $instruct->{efficiency};
            }
            else
            {
                die "No such instruct type: ", Dumper($instruct);
            }
            print "\n";
        }
    }
}

sub AnalyzeDAG
{
    my ($instructs, $effInstructs, $regMap) = @_;
    my $vectors = $regMap->{__vectors};
    my %deps;

    # efficiency dependencies
    foreach my $i (0 .. $#$instructs)
    {
        next unless $i != 0;
        my $instruct = $instructs->[$i];
        foreach my $gram (@{$grammar{$instruct->{op}}})
        {
            my $parent = $instructs->[$i - 1];
            my $effParent = $effInstructs->[$i - 1];
            my $instructType = $gram->{type};
            if ($parent->{dualCnt} == 1) # parent dual
            {
                if ($instruct->{dualCnt} == 0) # links to parent and grandparent
                {
                    my $grandparent = $instructs->[$i - 2];
                    my $effGrandparent = $effInstructs->[$i - 2];
                    push @{$parent->{children}}, [$i, 1 / $instruct->{efficiency}];
                    push @{$grandparent->{children}}, [$i, 1 / $instruct->{efficiency}];
                    push @{$effParent->{children}}, [$i, 1 / $instruct->{efficiency}, $instructType];
                    push @{$effGrandparent->{children}}, [$i, 1 / $instruct->{efficiency}, $instructType->{class}];
                }
                else # not recommend issue pattern, TODO(keren): cannot dual in this way?
                {
                    my $grandparent = $instructs->[$i - 2];
                    my $effGrandparent = $effInstructs->[$i - 2];
                    if ($grandparent->{dualCnt} == 0)
                    { # links to grandparent and parent
                        push @{$parent->{children}}, [$i, 1 / $instruct->{efficiency}];
                        push @{$grandparent->{children}}, [$i, 1 / $instruct->{efficiency}];
                        push @{$effParent->{children}}, [$i, 1 / $instruct->{efficiency}, $instructType->{class}];
                        push @{$effGrandparent->{children}}, [$i, 1 / $instruct->{efficiency}, $instructType->{class}];
                    }
                    else
                    { # links to parent becuase it is illegal
                        push @{$parent->{children}}, [$i, 1 / $instruct->{efficiency}];
                        push @{$effParent->{children}}, [$i, 1 / $instruct->{efficiency}, $instructType->{class}];
                    }
                }
            }
            elsif ($parent->{dualCnt} == 0) # parent single
            {
                if ($instruct->{dualCnt} == 0) # links to parent
                {
                    push @{$parent->{children}}, [$i, 1 / $instruct->{efficiency}];
                    push @{$effParent->{children}}, [$i, 1 / $instruct->{efficiency}, $instructType->{class}];
                }
                else # links to grandparent
                {
                    my $grandparent = $instructs->[$i - 2];
                    my $effGrandparent = $effInstructs->[$i - 2];
                    push @{$grandparent->{children}}, [$i, 1 / $instruct->{efficiency}];
                    push @{$effGrandparent->{children}}, [$i, 1 / $instruct->{efficiency}, $instructType->{class}];
                }
            }
        }
    }

    foreach my $i (0 .. $#$instructs)
    {
        next unless $i != 0;

        #skip control instructions
        my $instruct = $instructs->[$i];
        my ($op, $inst) = @{$instructs->[$i]}{qw(op inst)};

        # write dependencies
        my $match = 0;
        foreach my $gram (@{$grammar{$instruct->{op}}})
        {
            my $capData = parseInstruct($instruct->{inst}, $gram) or next;
            my (@dest, @src);

            # copy over instruction types for easier access
            @{$instruct}{@itypes} = @{$gram->{type}}{@itypes};

            # A predicate prefix is treated as a source reg
            push @src, $instruct->{predReg} if $instruct->{pred};

            # Handle P2R and R2P specially
            if ($instruct->{op} =~ m'P2R|R2P' && $capData->{i20w7})
            {
                my $list = $instruct->{op} eq 'R2P' ? \@dest : \@src;
                my $mask = hex($capData->{i20w7});
                foreach my $p (0..6)
                {
                    if ($mask & (1 << $p))
                    {
                        push @$list, "P$p";
                    }
                    # make this instruction dependent on any predicates it's not setting
                    # this is to prevent a race condition for any predicate sets that are pending
                    elsif ($instruct->{op} eq 'R2P')
                    {
                        push @src, "P$p";
                    }
                }
                # These instructions can't be dual issued
                $instruct->{nodual} = 1;
            }
            # Populate our register source and destination lists, skipping any zero or true values
            foreach my $operand (grep { exists $regops{$_} } sort keys %$capData)
            {
                # figure out which list to populate
                my $list = exists($destReg{$operand}) && !exists($noDest{$instruct->{op}}) ? \@dest : \@src;

                # Filter out RZ and PT
                my $badVal = substr($operand,0,1) eq 'r' ? 'RZ' : 'PT';

                if ($capData->{$operand} ne $badVal)
                {
                    # add the value to list with the correct prefix
                    push @$list,
                        $operand eq 'r0' ? map(getRegNum($regMap, $_), getVecRegisters($vectors, $capData)) :
                        $operand eq 'r8' ? map(getRegNum($regMap, $_), getAddrVecRegisters($vectors, $capData)) :
                        $operand eq 'CC' ? 'CC' :
                        $operand eq 'X'  ? 'CC' :
                        getRegNum($regMap, $capData->{$operand});
                }
            }

            # Find Read-After-Write dependencies
            foreach my $src (grep { exists $deps{$_} } @src)
            {
                # the parent should be the most recently added dest op to the stack
                foreach my $parent (@{$deps{$src}})
                {
                    # add this instruction as a child of the parent
                    # set the edge to the total latency of reg source availability
                    #print "R $parent->{inst}\n\t\t$instruct->{inst}\n";
                    my $latency = $src =~ m'^P\d' ? 13 : $parent->{lat};
                    # update weights
                    my $find = 0;
                    foreach my $child (@{$parent->{children}})
                    {
                        my $ins = $instructs->[$child->[0]];
                        my $weight = $child->[1];
                        if ($ins eq $instruct)
                        {
                            $child->[1] = $weight > $latency ? $weight : $latency;
                            $find = 1;
                            last;
                        }
                    }
                    # parent and child does not has efficiency dependency
                    if ($find == 0)
                    {
                         push @{$parent->{children}}, [$i, $latency];
                    }
                    $instruct->{parents}++;

                    # if the destination was conditionally executed, we also need to keep going back till it wasn't
                    last unless $parent->{pred};
                }
            }

            # For a dest reg, push it onto the write stack
            unshift @{$deps{$_}}, $instruct foreach @dest;

            $match = 1;
            last;
        }

        die "Unable to recognize instruction: $instruct->{inst}\n" unless $match;
    }
}

sub ConstructEfficiencyDAG
{
    my ($effInstructs, $typeInstructs, $types) = @_;

    foreach my $i (0 .. $#$effInstructs)
    {
        my $instruct = $effInstructs->[$i];
        my $typeInstruct = $typeInstructs->[$i];

        foreach my $child (@{$instruct->{children}})
        {
            my $instructType = $child->[2];

            my $find = 0;
            my $weight = 0;
            foreach my $type (@$types) 
            {
                if ($instructType eq $type)
                {
                    $weight = $child->[1];
                }
            }
            push @{$typeInstruct->{children}}, [$child->[0], $weight];
        }
    }
}

sub CalculateBcomp
{
    my ($instructs) = @_;
    # Bcomp
    my $unitsSum = 0;
    my $unitsUse = 0;

    foreach my $i (0 .. $#$instructs)
    {
        next unless $i != 0;

        my $instruct = $instructs->[$i];
        my ($op, $inst) = @{$instructs->[$i]}{qw(op inst)};
  
        my $match = 0;
        foreach my $gram (@{$grammar{$op}})
        {
            my $dispatches = 1;
            my $instructType = $gram->{type}; 
            if ($instructType->{class} eq 'x32' || $instructType->{class} eq 's2r' || $instructType->{class} eq 'qtr' ||
                $instructType->{class} eq 'rro' || $instructType->{class} eq 'vote') {
                $unitsSum = $unitsSum +  $instructType->{units};
                $unitsUse = $unitsUse + $dispatches * $warpSize  
            }
        }
    }
    print "Bcomp: ", $unitsSum > 0 ? 1.0 - $unitsUse / $unitsSum : 0, "\n";
}

sub CalculateBmem
{
    my ($instructs) = @_;
    # Bmem
    my $sharedWidthSum = 0;
    my $sharedWidthUse = 0;
    my $globalWidthSum = 0;
    my $globalWidthUse = 0;
    foreach my $i (0 .. $#$instructs)
    {
        next unless $i != 0;

        my $instruct = $instructs->[$i];
        my ($op, $inst) = @{$instructs->[$i]}{qw(op inst)};
  
        my $match = 0;
        foreach my $gram (@{$grammar{$op}})
        {
            my $capData = parseInstruct($inst, $gram) or next;
            @{$instruct}{@itypes} = @{$gram->{type}}{@itypes};
            my $dispatches = 1;
            my $instructType = $gram->{type}; 
            if ($instructType->{class} eq 'mem') {
                my $memType = $capData->{type};
                # default 32 bit
                my $insWidth = 4;
                # vector instruction
                if ($memType =~ s/^\.//g) {
                    $insWidth = $memType / 8;
                }
                if ($instructType->{type} eq 'global') {
                    $globalWidthSum = $globalWidthSum + 16 * $warpSize; # LDG.128
                    #TODO cache 
                    if ($op eq 'LDG') {
                        $globalWidthUse = $globalWidthUse + $insWidth * $warpSize;
                    } else {
                        $globalWidthUse = $globalWidthUse + ($insWidth / ceil($insWidth / 4)) * $warpSize;
                    }
                } else { #shared 
                    $sharedWidthSum = $sharedWidthSum + $bankWidth * $warpSize;
                    $sharedWidthUse = $sharedWidthUse + ($insWidth / ceil($insWidth / $bankWidth)) * $warpSize;
                }
            }
        }
    }
    print "Bshared: ", $sharedWidthSum > 0 ? 1.0 - $sharedWidthUse / $sharedWidthSum : 0, "\n";
    print "Bglobal: ", $globalWidthSum > 0 ? 1.0 - $globalWidthUse / $globalWidthSum : 0, "\n";
}

sub CalculateBilp
{
    # efficiency dependencies for each unit
    # TODO(keren): analyze more units
    my ($effInstructs, $cweff) = @_;

    my @x32type = ('s2r', 'x32', 'shift', 'cmp', 'vote');
    my @x64type = ('x64');
    my @sptype = ('qtr', 'rro');
    my @memtype = ('mem');

    my @x32Instructs;
    my @x64Instructs;
    my @spInstructs;
    my @memInstructs;

    foreach my $i (0 .. $#$effInstructs)
    {
        push @x32Instructs, {};
        push @x64Instructs, {};
        push @spInstructs, {};
        push @memInstructs, {};
    }

    ConstructEfficiencyDAG($effInstructs, \@x32Instructs, \@x32type);
    ConstructEfficiencyDAG($effInstructs, \@x64Instructs, \@x64type);
    ConstructEfficiencyDAG($effInstructs, \@spInstructs, \@sptype);
    ConstructEfficiencyDAG($effInstructs, \@memInstructs, \@memtype);
    
    my $cx32eff = LongestPath(\@x32Instructs);
    my $cx64eff = LongestPath(\@x64Instructs);
    my $cspeff = LongestPath(\@spInstructs);
    my $cmemeff = LongestPath(\@memInstructs);
    my $maxeff = max($cx32eff, $cspeff, $cx64eff, $cmemeff);
    #print "cx32eff: ", $cx32eff, "\n";
    #print "cx64eff: ", $cx64eff, "\n";
    #print "csp32eff: ", $cspeff, "\n";
    #print "cmemeff: ", $cmemeff, "\n";

    print "Bilp: ", $cweff > 0 ? 1.0 - $maxeff / $cweff : 0, "\n";
}

# Bpipe
# push longest path
sub CalculateBpipe
{
    my ($instructs, $cweff) = @_;

    my @path;
    foreach my $i (0 .. $#$instructs)
    {
        $path[$i] = 0;
    }

    foreach my $i (0 .. $#$instructs)
    {
        my $instruct = $instructs->[$i];
        foreach my $child (@{$instruct->{children}})
        {
            my $iChild= $child->[0];
            my $weight = $child->[1];
            if ($weight + $path[$i] > $path[$iChild])
            {
                $path[$iChild] = $weight + $path[$i];
                my $ins = $instructs->[$iChild];
                $ins->{prev} = {prevInstruct=>$instruct, prevWeight=>$weight};
            }
        }
    }

    my $longestPath = 0;
    foreach my $i (0 .. $#$instructs)
    {
        $longestPath = $path[$i] if $path[$i] > $longestPath;
    }

    my $longestLatency = 0;
    foreach my $i (0 .. $#$instructs) 
    {
        my $instruct = $instructs->[$i];
        my $latencies = 0;
        if ($path[$i] == $longestPath)
        {
            while (defined($instruct->{prev}))
            {
                my $prevIns = $instruct->{prev}->{prevInstruct};
                my $prevWeight = $instruct->{prev}->{prevWeight};
                my $prevLat = $prevIns->{lat};
                if ($prevLat == $prevWeight) 
                {
                    $latencies = $latencies + $prevWeight;
                }
                $instruct = $prevIns;
            }
        }
        $longestLatency = $latencies if $latencies > $longestLatency;
    }
    my $eff = $cweff * $activeWarp / $scheduler;
    print "Bpipe: ", $eff > 0 ? $longestLatency / $eff : 0, "\n";
}

sub Analyze
{
    # 1. Read two files, architecture configurations and software resource usage
    # 2. Output each instruction, and its efficiency
    # 3. Identify the critical path
    # 4. Compute bottlenecks
    my ($file, $include) = @_;
  
    my $regMap = {};
    $file = Preprocess($file, $include, 0, $regMap, 1);

    # Extract analyze block
    my @analyzeBlocks = $file =~ /$AnalyzeRe/g;

    # Iterate over analyz blocks
    foreach my $i (0 .. $#analyzeBlocks)
    {
        print "Analyze block $i\n\n";

        # Preprocess instructs
        my @instructs = PreprocessBlock($analyzeBlocks[$i]);

        # Calculate each instruction's efficiency
        CalculateEfficiency(\@instructs);

        # Analyze DAG dependencies
        # Init eff instructs
        my @effInstructs;
        foreach my $ins (@instructs)
        {
            push @effInstructs, {};
        }
        AnalyzeDAG(\@instructs, \@effInstructs, $regMap);

        # calculate longest path
        my $predictedCycle = LongestPath(\@instructs);
        print "predict cycles $predictedCycle\n";

        ## bottleneck analyze
        CalculateBcomp(\@instructs);
        CalculateBmem(\@instructs);
        my $cweff = LongestPath(\@effInstructs);
        CalculateBilp(\@effInstructs, $cweff);
        CalculateBpipe(\@instructs, $cweff);

        print "\n\n";
    }

    print "End analyze\n";
}

# Preprocess and Assemble a source file
sub Assemble
{
    my ($file, $include, $doReuse, $nowarn) = @_;

    my $regMap = {};
    $file = Preprocess($file, $include, 0, $regMap, 0);
    my $vectors = delete $regMap->{__vectors};
    my $regBank = delete $regMap->{__regbank};

    # initialize cubin counts
    my $regCnt = 0;
    my $barCnt = 0;

    my ($lineNum, @instructs, %labels, $ctrl, @branches, %reuse);

    # initialize the first control instruction
    push @instructs, $ctrl = {};

    foreach my $line (split "\n", $file)
    {
        # keep track of line nums in the physical file
        $lineNum++;

        next unless preProcessLine($line);

        # match an instruction
        if (my $inst = processAsmLine($line, $lineNum))
        {
            # Save us from crashing the display driver
            die "It is illegal to set a Read-After-Write dependency on a memory store op (store ops don't write to a register)\n$inst->{inst}\n"
                if exists $noDest{$inst->{op}} && ($inst->{ctrl} & 0x000e0) != 0x000e0;

            # track branches/jumps/calls/etc for label remapping
            push @branches, @instructs+0 if exists $jumpOp{$inst->{op}};

            # push the control code onto the control instruction
            push @{$ctrl->{ctrl}}, $inst->{ctrl};

            # now point the instruction to its associated control instruction
            $inst->{ctrl} = $ctrl;

            # add the op name and full instruction text
            push @instructs, $inst;

            # add a 4th control instruction for every 3 instructions
            push @instructs, $ctrl = {} if ((@instructs & 3) == 0);
        }
        # match a label
        elsif ($line =~ m'^([a-zA-Z]\w*):')
        {
            # map the label name to the index of the instruction about to be inserted
            $labels{$1} = @instructs+0;
        }
        else
        {
            die "badly formed line at $lineNum: $line\n";
        }
    }
    # add the final BRA op and align the number of instructions to a multiple of 8
    push @{$ctrl->{ctrl}}, 0x007ff;
    push @instructs, { op => 'BRA', inst => 'BRA 0xfffff8;' };
    while (@instructs & 7)
    {
        push @instructs, $ctrl = {} if ((@instructs & 3) == 0);
        push @{$ctrl->{ctrl}}, 0x007e0;
        push @instructs, { op => 'NOP', inst => 'NOP;' };
    }

    # remap labels
    foreach my $i (@branches)
    {
        if ($instructs[$i]{inst} !~ m'(\w+);$' || !exists $labels{$1})
            { die "instruction has invalid label: $instructs[$i]{inst}"; }

        $instructs[$i]{jump} = $labels{$1};

        if (exists $relOffset{$instructs[$i]{op}})
            { $instructs[$i]{inst} =~ s/(\w+);$/sprintf '0x%06x;', (($labels{$1} - $i - 1) * 8) & 0xffffff/e; }
        else
            { $instructs[$i]{inst} =~ s/(\w+);$/sprintf '0x%06x;', ($labels{$1} * 8) & 0xffffff/e; }
    }

    # calculate optimal register reuse
    # This effects register bank decisions so do it before analyzing register use
    foreach my $i (0 .. $#instructs)
    {
        #skip control instructions
        next unless $i & 3;

        my ($op, $inst, $ctrl) = @{$instructs[$i]}{qw(op inst ctrl)};

        my $match = 0;
        foreach my $gram (@{$grammar{$op}})
        {
            # Apply the rule pattern
            my $capData = parseInstruct($inst, $gram) or next;

            if ($doReuse)
            {
                # get any vector registers for r0
                my @r0 = getVecRegisters($vectors, $capData);

                # There are 2 reuse slots per register slot
                # The reuse hash points to most recent instruction index where register was last used in this slot

                # For writes to a register, clear any reuse opportunity
                if (@r0 && !exists $noDest{$op})
                {
                    foreach my $slot (keys %reuseSlots)
                    {
                        if (my $reuse = $reuse{$slot})
                        {
                            # if writing with a vector op, clear all linked registers
                            delete $reuse->{$_} foreach @r0;
                        }
                    }
                }
                # clear cache if jumping elsewhere
                %reuse = () if exists $jumpOp{$op};

                # only track register reuse for instruction types this works with
                if ($gram->{type}{reuse})
                {
                    foreach my $slot (keys %reuseSlots)
                    {
                        next unless exists $capData->{$slot};

                        my $r = $capData->{$slot};
                        next if $r eq 'RZ';
                        next if $r eq $capData->{r0}; # dont reuse if we're writing this reg in the same instruction

                        my $reuse = $reuse{$slot} ||= {};

                        # if this register was previously marked for potential reuse
                        if (my $p = $reuse->{$r})
                        {
                            # flag the previous instruction's ctrl reuse array slot
                            $instructs[$p]{ctrl}{reuse}[($p & 3) - 1] |= $reuseSlots{$slot};

                            #print "reuse $slot $r $instructs[$p]{inst}\n";
                        }
                        # list full, delete the oldest
                        elsif (keys %$reuse > 2)
                        {
                            my $oldest = (sort {$reuse->{$a} <=> $reuse->{$b}} keys %$reuse)[0];
                            delete $reuse->{$oldest};
                        }
                        # mark the new instruction for potential reuse
                        $reuse->{$r} = $i;
                    }
                }
            }
            # if reuse is disabled then pull value from code.
            elsif ($gram->{type}{reuse})
            {
                $ctrl->{reuse}[($i & 3) - 1] = genReuseCode($capData);
            }
            $match = 1;
            last;
        }
        unless ($match)
        {
            print "$_->{rule}\n\n" foreach @{$grammar{$op}};
            die "Unable to encode instruction: $inst\n";
        }
    }

    # Assign registers to requested banks if possible
    foreach my $r (sort keys %$regBank)
    {
        my $bank  = $regBank->{$r};
        my $avail = $regMap->{$r};
        foreach my $pos (0 .. $#$avail)
        {
            if ($bank == ($avail->[$pos] & 3))
            {
                # assign it, while removing the assigned register from the pool
                $regMap->{$r} = 'R' . splice @$avail, $pos, 1;
                last;
            }
        }
    }

    # calculate register live times and preferred banks for non-fixed registers.
    # LiveTime only half implemented...
    my (%liveTime, %pairedBanks, %reuseHistory);
    foreach my $i (0 .. $#instructs)
    {
        #skip control instructions
        next unless $i & 3;

        my ($op, $inst, $ctrl) = @{$instructs[$i]}{qw(op inst ctrl)};

        my $match = 0;
        foreach my $gram (@{$grammar{$op}})
        {
            # Apply the rule pattern
            my $capData   = parseInstruct($inst, $gram) or next;
            my $reuseType = $gram->{type}{reuse};

            # liveTimes and bank conflicts with source operands
            my (%addReuse, %delReuse);
            foreach my $slot (qw(r8 r20 r39))
            {
                my $r = $capData->{$slot} or next;
                next if $r eq 'RZ';

                my $liveR = ref $regMap->{$r} ? $r : $regMap->{$r};

                # All registers should be written prior to being read..
                if (my $liveTime = $liveTime{$liveR})
                {
                    # for each read set the current instruction index as the high value
                    $liveTime->[$#$liveTime][1] = $i;
                    push @{$liveTime->[$#$liveTime]}, "$i $inst";
                }
                else
                {
                    warn "register used without initialization ($r): $inst\n" unless $nowarn;
                    push @{$liveTime{$liveR}}, [$i,$i];
                }

                # Is this register active in the reuse cache?
                my $slotHist  = $reuseHistory{$slot} ||= {};
                my $selfReuse = $reuseType ? exists $slotHist->{$r} : 0;

                #print "IADD3-1: $slot:$r (!$selfReuse && $regMap->{$r})\n" if $op eq 'IADD3';

                # If this is an auto reg, look at the open banks.
                # No need to look at banks if this register is in the reuse cache.
                if (!$selfReuse && ref $regMap->{$r})
                {
                    # Look at other source operands in this instruction and flag what banks are being used
                    foreach my $slot2 (grep {$_ ne $slot && exists $capData->{$_}} qw(r8 r20 r39))
                    {
                        my $r2 = $capData->{$slot2};
                        next if $r2 eq 'RZ' || $r2 eq $r;

                        my $slotHist2 = $reuseHistory{$slot2} ||= {};

                        #print "IADD3-2: $slot:$r $slot2:$r2 (!$reuseType && !$slotHist2->{$r2})\n" if $op eq 'IADD3';

                        # Dont be concerned with non-reuse type instructions or
                        # If this operand is in the reuse cache, we don't care what bank it's on.
                        if (!$reuseType || !exists $slotHist2->{$r2})
                        {
                            # if the operand is also an auto-allocated register then link them
                            # Once we choose the bank for one we want to update that choice for the other register.
                            if (ref $regMap->{$r2})
                            {
                                push @{$pairedBanks{$r}{pairs}}, $r2;
                                $pairedBanks{$r}{banks} ||= [];
                            }
                            # For a fixed register, calculate the bank, flag it, and update the count of banks to avoid.
                            else
                            {
                                my $bank = substr($regMap->{$r2},1) & 3;
                                #print "IADD3-3: $r2:$bank\n" if $op eq 'IADD3';

                                $pairedBanks{$r}{bnkCnt}++ unless $pairedBanks{$r}{banks}[$bank]++;
                                $pairedBanks{$r}{pairs} ||= [];
                            }
                            # Update the total use count for this register.
                            # This will be the number of times the register is pulled out of the bank.
                            $pairedBanks{$r}{useCnt}++;
                        }
                    }
                }
                # update the reuse history so we know which bank conflicts we can ignore.
                if ($reuseType)
                {
                    # flag these slots for addition or removal from reuseHistory
                    if ($ctrl->{reuse}[($i & 3) - 1] & $reuseSlots{$slot})
                        { $addReuse{$slot} = $r; }
                    else
                        { $delReuse{$slot} = $r; }
                }
            }
            # update reuse history after we're done with the instruction (when the flag is actually in effect).
            # we don't want to updated it in the middle since that can interfere with the checks,
            $reuseHistory{$_}{$addReuse{$_}} = 1    foreach keys %addReuse;
            delete $reuseHistory{$_}{$delReuse{$_}} foreach keys %delReuse;

            # liveTimes for destination operands and vector registers
            foreach my $r0 (getVecRegisters($vectors, $capData))
            {
                # fixed register mappings can have aliases so use the actual register value for those.
                my $liveR = ref $regMap->{$r0} ? $r0 : $regMap->{$r0};

                # If not writing treat just like a read
                if (exists $noDest{$op})
                {
                    if (my $liveTime = $liveTime{$liveR})
                    {
                        $liveTime->[$#$liveTime][1] = $i;
                        push @{$liveTime->[$#$liveTime]}, "$i $inst";
                    }
                    else
                    {
                        warn "register used without initialization ($r0): $inst\n" unless $nowarn;
                        push @{$liveTime{$liveR}}, [$i,$i];
                    }
                }
                # If writing, push a new bracket on this register's stack.
                elsif (my $liveTime = $liveTime{$liveR})
                {
                    if ($i > $liveTime->[$#$liveTime][1])
                    {
                        push @{$liveTime{$liveR}}, [$i,$i, "$i $inst"];
                    }
                }
                else
                {
                    # Initialize the liveTime stack for this register.
                    push @{$liveTime{$liveR}}, [$i,$i, "$i $inst"];
                }
            }

            $match = 1;
            last;
        }
        unless ($match)
        {
            print "$_->{rule}\n\n" foreach @{$grammar{$op}};
            die "Unable to encode instruction: $inst\n";
        }
    }
    #print Dumper(\%liveTime); exit(1);

    # assign unassigned registers
    # sort by most restricted, then most used, then name
    foreach my $r (sort {
                    $pairedBanks{$b}{bnkCnt} <=> $pairedBanks{$a}{bnkCnt} ||
                    $pairedBanks{$b}{useCnt} <=> $pairedBanks{$a}{useCnt} ||
                    $a cmp $b
                  } keys %pairedBanks)
    {
        my $banks = $pairedBanks{$r}{banks};
        my $avail = $regMap->{$r};

        #printf "%10s: (%d,%d) %d,%d,%d,%d, %s\n", $r, $pairedBanks{$r}{bnkCnt}, $pairedBanks{$r}{useCnt}, @{$banks}[0,1,2,3], join ',', @$avail;

        # Pick a bank with zero or the smallest number of conflicts
        BANK: foreach my $bank (sort {$banks->[$a] <=> $banks->[$b] || $a <=> $b } (0..3))
        {
            # pick an available register that matches the requested bank
            foreach my $pos (0 .. $#$avail)
            {
                if ($bank == ($avail->[$pos] & 3))
                {
                    # assign it, while removing the assigned register from the pool
                    $regMap->{$r} = 'R' . splice @$avail, $pos, 1;

                    # update bank info for any unassigned pair
                    $pairedBanks{$_}{banks}[$bank]++ foreach @{$pairedBanks{$r}{pairs}};
                    last BANK;
                }
            }
        }
    }
    # Now assign any remaining to first available
    foreach my $r (sort keys %$regMap)
    {
        if (ref($regMap->{$r}) eq 'ARRAY')
        {
            $regMap->{$r} = 'R' . shift @{$regMap->{$r}};
        }
    }
    #print map "$regMap->{$_}: $_\n", sort { substr($regMap->{$a},1) <=> substr($regMap->{$b},1) } keys %$regMap;

    # apply the register mapping and assemble the instructions to op codes
    foreach my $i (0 .. $#instructs)
    {
        #skip control instructions
        next unless $i & 3;

        # save the original and replace the register names with numbers
        $instructs[$i]{orig} = $instructs[$i]{inst};
        $instructs[$i]{inst} =~ s/(?<!\.)\b(\w+)\b(?!\[)/ exists($regMap->{$1}) ? $regMap->{$1} : $1 /ge;

        my ($op, $inst, $ctrl) = @{$instructs[$i]}{qw(op inst ctrl)};

        my $match = 0;
        foreach my $gram (@{$grammar{$op}})
        {
            # Apply the rule pattern
            my $capData = parseInstruct($inst, $gram) or next;

            # update the register count
            foreach my $r (qw(r0 r8 r20 r39))
            {
                next unless exists($capData->{$r}) && $capData->{$r} ne 'RZ';

                # get numeric portion of regname
                my $val = substr $capData->{$r}, 1;

                my @r0 = getVecRegisters($vectors, $capData);
                my @r8 = getAddrVecRegisters($vectors, $capData);

                # smart enough to count vector registers for memory instructions.
                my $regInc = $r eq 'r0' ? scalar(@r0) || 1 : 1;
                my $regInc = $r eq 'r8' ? scalar(@r8) || 1 : 1;

                if ($val + $regInc > $regCnt)
                {
                    $regCnt = $val + $regInc;
                    #print "$val $regCnt $regInc\n";
                }
            }
            # update the barrier resource count
            if ($op eq 'BAR')
            {
                if (exists $capData->{i8w4})
                {
                    $barCnt = $capData->{i8w4}+1 if $capData->{i8w4}+1 > $barCnt;
                }
                # if a barrier value is a register, assume the maximum
                elsif (exists $capData->{r8})
                {
                    $barCnt = 16;
                }
            }
            # Generate the op code.
            my ($code, $reuse) = genCode($op, $gram, $capData);
            $instructs[$i]{code} = $code;

            # cache this for final pass when we want to calculate reuse stats.
            if ($gram->{type}{reuse})
                { $instructs[$i]{caps} = $capData; }
            # use the parsed value of reuse for non-reuse type instructions
            else
                { $ctrl->{reuse}[($i & 3) - 1] = $reuse; }


            $match = 1;
            last;
        }
        unless ($match)
        {
            print "$_->{rule}\n\n" foreach @{$grammar{$op}};
            die "Unable to encode instruction: $inst\n";
        }
    }

    # final pass to piece together control codes
    my (@codes, %reuseHistory, @exitOffsets, @ctaidOffsets, $ctaidzUsed);
    foreach my $i (0 .. $#instructs)
    {
        # op code
        if ($i & 3)
        {
            push @codes, $instructs[$i]{code};

            if ($instructs[$i]{caps})
            {
                # calculate stats on registers
                registerHealth(\%reuseHistory, $instructs[$i]{ctrl}{reuse}[($i & 3) - 1], $instructs[$i]{caps}, $i * 8, "$instructs[$i]{inst} ($instructs[$i]{orig})", $nowarn);
            }
            if ($instructs[$i]{inst} =~ m'EXIT')
            {
                push @exitOffsets, (scalar(@codes)-1)*8;
            }
            elsif ($instructs[$i]{inst} =~ m'SR_CTAID\.(X|Y|Z)')
            {
                push @ctaidOffsets, (scalar(@codes)-1)*8;
                $ctaidzUsed = 1 if $1 eq 'Z';
            }
        }
        # control code
        else
        {
            my ($ctrl, $ruse) = @{$instructs[$i]}{qw(ctrl reuse)};
            push @codes,
                ($ctrl->[0] <<  0) | ($ctrl->[1] << 21) | ($ctrl->[2] << 42) | # ctrl codes
                ($ruse->[0] << 17) | ($ruse->[1] << 38) | ($ruse->[2] << 59);  # reuse codes
        }
    }

    # return the kernel data
    return {
        RegCnt       => $regCnt,
        BarCnt       => $barCnt,
        ExitOffsets  => \@exitOffsets,
        CTAIDOffsets => \@ctaidOffsets,
        CTAIDZUsed   => $ctaidzUsed,
        ConflictCnt  => $reuseHistory{conflicts},
        ReuseCnt     => $reuseHistory{reuse},
        ReuseTot     => $reuseHistory{total},
        ReusePct     => ($reuseHistory{total} ? 100 * $reuseHistory{reuse} / $reuseHistory{total} : 0),
        KernelData   => \@codes,
    };
}

# Useful for testing op code coverage of existing code, extracting new codes and flags
sub Test
{
    my ($fh, $printConflicts, $all) = @_;

    my @instructs;
    my %reuseHistory;
    my ($pass, $fail) = (0,0);

    while (my $line = <$fh>)
    {
        my (@ctrl, @reuse);

        next unless processSassCtrlLine($line, \@ctrl, \@reuse);

        foreach my $fileReuse (@reuse)
        {
            $line = <$fh>;

            my $inst = processSassLine($line) or next;

            $inst->{reuse} = $fileReuse;
            my $fileCode = $inst->{code};

            if (exists $relOffset{$inst->{op}})
            {
                # these ops need to be converted from absolute addresses to relative in the sass output by cuobjdump
                $inst->{inst} =~ s/(0x[0-9a-f]+)/sprintf '0x%06x', ((hex($1) - $inst->{num} - 8) & 0xffffff)/e;
            }

            my $match = 0;
            foreach my $gram (@{$grammar{$inst->{op}}})
            {
                my $capData = parseInstruct($inst->{inst}, $gram) or next;
                my @caps;

                # Run in test mode to list what capture groups were captured
                my ($code, $reuse) = genCode($inst->{op}, $gram, $capData, \@caps);

                # Detect register bank conflicts but only for reuse type instructions.
                # If a bank conflict is avoided by a reuse flag then ignore it.
                registerHealth(\%reuseHistory, $reuse, $capData, $inst->{num}, $printConflicts ? $inst->{inst} : '') if $gram->{type}{reuse};

                $inst->{caps}      = join ', ', sort @caps;
                $inst->{codeDiff}  = $fileCode  ^ $code;
                $inst->{reuseDiff} = $fileReuse ^ $reuse;

                # compare calculated and file values
                if ($code == $fileCode && $reuse == $fileReuse)
                {
                    $inst->{grade} = 'PASS';
                    push @instructs, $inst if $all;
                    $pass++;
                }
                else
                {
                    $inst->{grade} = 'FAIL';
                    push @instructs, $inst;
                    $fail++;
                }
                $match = 1;
                last;
            }
            unless ($match)
            {
                $inst->{grade}     = 'FAIL';
                $inst->{codeDiff}  = $fileCode;
                $inst->{reuseDiff} = $fileReuse;
                push @instructs, $inst;
                $fail++;
            }
        }
    }
    my %maxLen;
    foreach (@instructs)
    {
        $maxLen{$_->{op}} = length($_->{ins}) if length($_->{ins}) > $maxLen{$_->{op}};
    }
    my ($lastOp, $template);
    foreach my $inst (sort {
        $a->{op}        cmp $b->{op}        ||
        $a->{codeDiff}  <=> $b->{codeDiff}  ||
        $a->{reuseDiff} <=> $b->{reuseDiff} ||
        $a->{ins}       cmp $b->{ins}
        } @instructs)
    {
        if ($lastOp ne $inst->{op})
        {
            $lastOp   = $inst->{op};
            $template = "%s 0x%016x %x 0x%016x %x %5s%-$maxLen{$lastOp}s   %s\n";
            printf "\n%s %-18s %s %-18s %s %-5s%-$maxLen{$lastOp}s   %s\n", qw(Grad OpCode R opCodeDiff r Pred Instruction Captures);
        }
        printf $template, @{$inst}{qw(grade code reuse codeDiff reuseDiff pred ins caps)};
    }
    my $reusePct = $reuseHistory{total} ? 100 * $reuseHistory{reuse} / $reuseHistory{total} : 0;

    printf "\nRegister Bank Conflicts: %d, Reuse: %.1f% (%d/%d)\nOp Code Coverage Totals: Pass: $pass Fail: $fail\n",
        $reuseHistory{conflicts}, $reusePct, $reuseHistory{reuse}, $reuseHistory{total};

    return $fail;
}

# Convert cuobjdump sass to the working format
sub Extract
{
    my ($in, $out, $params) = @_;

    my %paramMap;
    my %constants =
    (
        blockDimX => 'c[0x0][0x8]',
        blockDimY => 'c[0x0][0xc]',
        blockDimZ => 'c[0x0][0x10]',
        gridDimX  => 'c[0x0][0x14]',
        gridDimY  => 'c[0x0][0x18]',
        gridDimZ  => 'c[0x0][0x1c]',
    );
    print $out "<CONSTANT_MAPPING>\n";

    foreach my $const (sort keys %constants)
    {
        print $out "    $const : $constants{$const}\n";
        $paramMap{$constants{$const}} = $const;
    }
    print $out "\n";

    foreach my $p (@$params)
    {
        my ($ord,$offset,$size,$align) = split ':', $p;

        if ($size > 4)
        {
            my $num = 0;
            $offset = hex $offset;
            while ($size > 0)
            {
                my $param = sprintf 'param_%d[%d]', $ord, $num;
                my $const = sprintf 'c[0x0][0x%x]', $offset;
                $paramMap{$const} = $param;
                print $out "    $param : $const\n";
                $size   -= 4;
                $offset += 4;
                $num    += 1;
            }
        }
        else
        {
            my $param = sprintf 'param_%d', $ord;
            my $const = sprintf 'c[0x0][%s]', $offset;
            $paramMap{$const} = $param;
            print $out "    $param : $const\n";
        }
    }
    print $out "</CONSTANT_MAPPING>\n\n";

    my %labels;
    my $labelnum = 1;

    my @data;
    FILE: while (my $line = <$in>)
    {
        my (@ctrl, @ruse);
        next unless processSassCtrlLine($line, \@ctrl, \@ruse);

        CTRL: foreach my $ctrl (@ctrl)
        {
            $line = <$in>;

            my $inst = processSassLine($line) or next CTRL;

            # Convert branch/jump/call addresses to labels
            if (exists($jumpOp{$inst->{op}}) && $inst->{ins} =~ m'(0x[0-9a-f]+)')
            {
                my $target = hex($1);

                # skip the final BRA and stop processing the file
                last FILE if $inst->{op} eq 'BRA' && ($target == $inst->{num} || $target == $inst->{num}-8);

                # check to see if we've already generated a label for this target address
                my $label = $labels{$target};
                unless ($label)
                {
                    # generate a label name and cache it
                    $label = $labels{$target} = "TARGET$labelnum";
                    $labelnum++;
                }
                # replace address with name
                $inst->{ins} =~ s/(0x[0-9a-f]+)/$label/;
            }
            $inst->{ins} =~ s/(c\[0x0\])\s*(\[0x[0-9a-f]+\])/ $paramMap{$1 . $2} || $1 . $2 /eg;

            $inst->{ctrl} = printCtrl($ctrl);

            push @data, $inst;
        }
    }
    # make a second pass now that we have the complete instruction address to label mapping
    foreach my $inst (@data)
    {
        print $out "$labels{$inst->{num}}:\n" if exists $labels{$inst->{num}};
        printf $out "%s %5s%s\n", @{$inst}{qw(ctrl pred ins)};
    }
}

my $CommentRe  = qr'^[\t ]*<COMMENT>.*?^\s*</COMMENT>\n?'ms;
my $IncludeRe  = qr'^[\t ]*<INCLUDE\s+file="([^"]+)"\s*/?>\n?'ms;
my $CodeRe     = qr'^[\t ]*<CODE(\d*)>(.*?)^\s*<\/CODE\1>\n?'ms;
my $ConstMapRe = qr'^[\t ]*<CONSTANT_MAPPING>(.*?)^\s*</CONSTANT_MAPPING>\n?'ms;
my $RegMapRe   = qr'^[\t ]*<REGISTER_MAPPING>(.*?)^\s*</REGISTER_MAPPING>\n?'ms;
my $ScheduleRe = qr'^[\t ]*<SCHEDULE_BLOCK>(.*?)^\s*</SCHEDULE_BLOCK>\n?'ms;
my $InlineRe   = qr'\[(\+|\-)(.+?)\1\]'ms;

sub IncludeFile
{
    my ($file, $include) = @_;
    my ($vol,$dir,$name) = File::Spec->splitpath($file);
    local $/;
    my $fh;
    if (!open $fh, $file)
    {
        open $fh, File::Spec->catpath(@$include, $name) or die "Could not open file for INCLUDE: $file ($!)\n";
    }
    my $content = <$fh>;
    close $fh;
    return $content;
}

sub Preprocess
{
    my ($file, $include, $debug, $regMap, $doAnalyze) = @_;

    my $constMap = {};
    my $removeRegMap;
    if ($regMap)
        { $removeRegMap = 1; }
    else
        { $regMap = {}; }

    # include nested files
    1 while $file =~ s|$IncludeRe| IncludeFile($1, $include) |eg;

    # Strip out comments
    $file =~ s|$CommentRe||g;

    # Execute the CODE sections (old way to run code, to be deprecated)
    1 while $file =~ s|$CodeRe|
        my $out = eval "package MaxAs::MaxAs::CODE; $2";
        $@ ? die("CODE:\n$2\n\nError: $@\n") : $out |eg;

    # Execute the inline code (new way)
    $file =~ s|$InlineRe|
        my ($type, $code) = ($1, $2);
        my $out = eval "package MaxAs::MaxAs::CODE; $code";
        $@ ? die("CODE:\n$code\n\nError: $@\n") : $type eq "+" ? $out : "" |eg;

    #Pull in the constMap
    $file =~ s/$ConstMapRe/ setConstMap($constMap, $1) /eg;

    my @newFile;
    foreach my $line (split "\n", $file)
    {
        # skip comments
        if ($line !~ m'^\s*(?:#|//).*')
        {
            $line =~ s|(\w+(?:\[\d+\])?)| exists $constMap->{$1} ? $constMap->{$1} : $1 |eg;
        }
        push @newFile, $line;
    }
    $file = join "\n", @newFile;

    # Pull in the reg map first as the Scheduler will need it to handle vector instructions
    # Remove the regmap if we're going on to assemble
    $file =~ s/$RegMapRe/ setRegisterMap($regMap, $1); $removeRegMap ? '' : $& /eg;

    # Pick out the SCHEDULE_BLOCK sections
    my @schedBlocks = $file =~ /$ScheduleRe/g;

    # Schedule them
    foreach my $i (0 .. $#schedBlocks)
    {
        # XMAD macros should only appear in SCHEDULE_BLOCKs
        $schedBlocks[$i] = replaceXMADs($schedBlocks[$i]);

        $schedBlocks[$i] = Scheduler($schedBlocks[$i], $i+1, $regMap, $debug);
    }

    # Replace the results
    $file =~ s|$ScheduleRe| shift @schedBlocks |eg;

    # Strip out analyzeBlocks
    $file =~ s|$AnalyzeRe||eg if not $doAnalyze;

    return $file;
}

sub Scheduler
{
    my ($block, $blockNum, $regMap, $debug) = @_;

    my $vectors = $regMap->{__vectors};
    my $lineNum = 0;

    my (@instructs, @comments, $ordered, $first);
    foreach my $line (split "\n", $block)
    {
        # keep track of line nums in the physical file
        $lineNum++;

        unless (preProcessLine($line))
        {
            push @comments, $line if $line =~ m'\S';
            next;
        }

        # match an instruction
        if (my $inst = processAsmLine($line, $lineNum))
        {
            # if the first instruction in the block is waiting on a dep, it should go first.
            $inst->{first}   = !$first++ && ($inst->{ctrl} & 0x1f800) ? 0 : 1;

            # if the instruction has a stall of zero set, it's meant to be last (to mesh with next block)
            #$inst->{first}   = $inst->{ctrl} & 0x0000f ? 1 : 2;
            $inst->{exeTime} = 0;
            $inst->{order}   = $ordered++ if $ordered;
            $inst->{force_stall} = $inst->{ctrl} & 0xf if $inst->{comment} =~ m'FORCE';

            push @instructs, $inst;
        }
        # match a label
        elsif ($line =~ m'^([a-zA-Z]\w*):')
        {
            die "SCHEDULE_BLOCK's cannot contain labels. block: $blockNum line: $lineNum\n";
        }
        # open an ORDERED block
        elsif ($line =~ m'^<ORDERED>')
        {
            die "you cannot use nested <ORDERED> tags" if $ordered;
            $ordered = 1;
        }
        # close an ORDERED block
        elsif ($line =~ m'^</ORDERED>')
        {
            die "missing opening <ORDERED> for closing </ORDERED> tag" if !$ordered;
            $ordered = 0;
        }
        else
        {
            die "badly formed line at block: $blockNum line: $lineNum: $line\n";
        }
    }
    my (%writes, %reads, @ready, @schedule, $orderedParent);
    # assemble the instructions to op codes
    foreach my $instruct (@instructs)
    {
        my $match = 0;
        foreach my $gram (@{$grammar{$instruct->{op}}})
        {
            my $capData = parseInstruct($instruct->{inst}, $gram) or next;
            my (@dest, @src);

            # copy over instruction types for easier access
            @{$instruct}{@itypes} = @{$gram->{type}}{@itypes};

            $instruct->{dualCnt} = $instruct->{dual} ? 1 : 0;

            # A predicate prefix is treated as a source reg
            push @src, $instruct->{predReg} if $instruct->{pred};

            # Handle P2R and R2P specially
            if ($instruct->{op} =~ m'P2R|R2P' && $capData->{i20w7})
            {
                my $list = $instruct->{op} eq 'R2P' ? \@dest : \@src;
                my $mask = hex($capData->{i20w7});
                foreach my $p (0..6)
                {
                    if ($mask & (1 << $p))
                    {
                        push @$list, "P$p";
                    }
                    # make this instruction dependent on any predicates it's not setting
                    # this is to prevent a race condition for any predicate sets that are pending
                    elsif ($instruct->{op} eq 'R2P')
                    {
                        push @src, "P$p";
                    }
                }
                # These instructions can't be dual issued
                $instruct->{nodual} = 1;
            }

            # Populate our register source and destination lists, skipping any zero or true values
            foreach my $operand (grep { exists $regops{$_} } sort keys %$capData)
            {
                # figure out which list to populate
                my $list = exists($destReg{$operand}) && !exists($noDest{$instruct->{op}}) ? \@dest : \@src;

                # Filter out RZ and PT
                my $badVal = substr($operand,0,1) eq 'r' ? 'RZ' : 'PT';

                if ($capData->{$operand} ne $badVal)
                {
                    # add the value to list with the correct prefix
                    push @$list,
                        $operand eq 'r0' ? map(getRegNum($regMap, $_), getVecRegisters($vectors, $capData)) :
                        $operand eq 'r8' ? map(getRegNum($regMap, $_), getAddrVecRegisters($vectors, $capData)) :
                        $operand eq 'CC' ? 'CC' :
                        $operand eq 'X'  ? 'CC' :
                        getRegNum($regMap, $capData->{$operand});
                }
            }
            $instruct->{const} = 1 if exists($capData->{c20}) || exists($capData->{c39});

            # Find Read-After-Write dependencies
            foreach my $src (grep { exists $writes{$_} } @src)
            {
                # Memory operations get delayed access to registers but not to the predicate
                my $regLatency = $src eq $instruct->{predReg} ? 0 : $instruct->{rlat};

                # the parent should be the most recently added dest op to the stack
                foreach my $parent (@{$writes{$src}})
                {
                    # add this instruction as a child of the parent
                    # set the edge to the total latency of reg source availability
                    #print "R $parent->{inst}\n\t\t$instruct->{inst}\n";
                    my $latency = $src =~ m'^P\d' ? 13 : $parent->{lat};
                    push @{$parent->{children}}, [$instruct, $latency - $regLatency];
                    $instruct->{parents}++;

                    # if the destination was conditionally executed, we also need to keep going back till it wasn't
                    last unless $parent->{pred};
                }
            }

            # Find Write-After-Read dependencies
            foreach my $dest (grep { exists $reads{$_} } @dest)
            {
                # Flag this instruction as dependent to any previous read
                foreach my $reader (@{$reads{$dest}})
                {
                    # no need to stall for these types of dependencies
                    #print "W $reader->{inst} \t\t\t $instruct->{inst}\n";
                    push @{$reader->{children}}, [$instruct, 0];
                    $instruct->{parents}++;
                }
                # Once dependence is marked we can clear out the read list (unless this write was conditional).
                # The assumption here is that you would never want to write out a register without
                # subsequently reading it in some way prior to writing it again.
                delete $reads{$dest} unless $instruct->{pred};
            }

            # Enforce instruction ordering where requested
            if ($instruct->{order})
            {
                if ($orderedParent && $instruct->{order} > $orderedParent->{order})
                {
                    push @{$orderedParent->{children}}, [$instruct, 0];
                    $instruct->{parents}++;
                }
                $orderedParent = $instruct;
            }
            elsif ($orderedParent)
                {  $orderedParent = 0; }

            # For a dest reg, push it onto the write stack
            unshift @{$writes{$_}}, $instruct foreach @dest;

            # For a src reg, push it into the read list
            push @{$reads{$_}}, $instruct foreach @src;

            # if this instruction has no dependencies it's ready to go
            push @ready, $instruct if !exists $instruct->{parents};

            $match = 1;
            last;
        }
        die "Unable to recognize instruction at block: $blockNum line: $lineNum: $instruct->{inst}\n" unless $match;
    }
    %writes = ();
    %reads  = ();

    if (@ready)
    {
        # update dependent counts for sorting hueristic
        my $readyParent = { children => [ map { [ $_, 1 ] } @ready ], inst => "root" };

        countUniqueDescendants($readyParent, {});
        updateDepCounts($readyParent, {});

        # sort the initial ready list
        @ready = sort {
            $a->{first}   <=> $b->{first}  ||
            $b->{deps}    <=> $a->{deps}   ||
            $a->{dualCnt} <=> $b->{dualCnt}  ||
            $a->{lineNum} <=> $b->{lineNum}
            } @ready;

        if ($debug)
        {
            print  "0: Initial Ready List State:\n\tf,ext,stl,mix,dep,lin, inst\n";
            printf "\t%d,%3s,%3s,%3s,%3s,%3s,%3s, %s\n", @{$_}{qw(first exeTime stall dualCnt mix deps lineNum inst)} foreach @ready;
        }
    }

    # Process the ready list, adding new instructions to the list as we go.
    my $clock = 0;
    while (my $instruct = shift @ready)
    {
        my $stall = $instruct->{stall};

        # apply the stall to the previous instruction
        if (@schedule && $stall < 16)
        {
            my $prev = $schedule[$#schedule];

            $stall = $prev->{force_stall} if $prev->{force_stall} > $stall;

            # if stall is greater than 4 then also yield
            # the yield flag is required to get stall counts 12-15 working correctly.
            $prev->{ctrl} &= $stall > 4 ? 0x1ffe0 : 0x1fff0;
            $prev->{ctrl} |= $stall;
            $clock += $stall;
        }
        # For stalls bigger than 15 we assume the user is managing it with a barrier
        else
        {
            $instruct->{ctrl} &= 0x1fff0;
            $instruct->{ctrl} |= 1;
            $clock += 1;
        }
        print "$clock: $instruct->{inst}\n" if $debug;

        # add a new instruction to the schedule
        push @schedule, $instruct;

        # update each child with a new earliest execution time
        if (my $children = $instruct->{children})
        {
            foreach (@$children)
            {
                my ($child, $latency) = @$_;

                # update the earliest clock value this child can safely execute
                my $earliest = $clock + $latency;
                $child->{exeTime} = $earliest if $child->{exeTime} < $earliest;

                print "\t\t$child->{exeTime},$child->{parents} $child->{inst}\n" if $debug;

                # decrement parent count and add to ready queue if none remaining.
                push @ready, $child if --$child->{parents} < 1;
            }
            delete $instruct->{children};
        }

        # update stall and mix values in the ready queue on each iteration
        foreach my $ready (@ready)
        {
            # calculate how many instructions this would cause the just added instruction to stall.
            $stall = $ready->{exeTime} - $clock;
            $stall = 1 if $stall < 1;

            # if using the same compute resource as the prior instruction then limit the throughput
            if ($ready->{class} eq $instruct->{class})
            {
                $stall = $ready->{tput} if $stall < $ready->{tput};
            }
            # dual issue with a simple instruction (tput <= 2)
            # can't dual issue two instructions that both load a constant
            elsif ($ready->{dual} && !$instruct->{dual} && $instruct->{tput} <= 2 && !$instruct->{nodual} &&
                   $stall == 1 && $ready->{exeTime} <= $clock && !($ready->{const} && $instruct->{const}))
            {
                $stall = 0;
            }
            $ready->{stall} = $stall;

            # add an instruction class mixing huristic that catches anything not handled by the stall
            $ready->{mix} = $ready->{class} ne $instruct->{class} || 0;
            $ready->{mix} = 2 if $ready->{mix} && $ready->{op} eq 'R2P';
        }

        # sort the ready list by stall time, mixing huristic, dependencies and line number
        @ready = sort {
            $a->{first}   <=> $b->{first}  ||
            $a->{stall}   <=> $b->{stall}  ||
            $a->{dualCnt} <=> $b->{dualCnt}  ||
            $b->{mix}     <=> $a->{mix}    ||
            $b->{deps}    <=> $a->{deps}   ||
            $a->{lineNum} <=> $b->{lineNum}
            } @ready;

        if ($debug)
        {
            print  "\tf,ext,stl,duc,mix,dep,lin, inst\n";
            printf "\t%d,%3s,%3s,%3s,%3s,%3s,%3s, %s\n", @{$_}{qw(f exeTime stall dualCnt mix deps lineNum inst)} foreach @ready;
        }

        foreach my $ready (@ready)
        {
            $ready->{dualCnt} = 0 if $ready->{dualCnt} && $ready->{stall} == 1;
        }
    }

    my $out;
    #$out .= "$_\n" foreach @comments;
    $out .= join('', printCtrl($_->{ctrl}), @{$_}{qw(space inst comment)}, "\n") foreach @schedule;
    return $out;
}

sub setConstMap
{
    my ($constMap, $constMapText) = @_;

    foreach my $line (split "\n", $constMapText)
    {
        # strip leading space
        $line =~ s|^\s+||;
        # strip comments
        $line =~ s{(?:#|//).*}{};
        # strip trailing space
        $line =~ s|\s+$||;
        # skip blank lines
        next unless $line =~ m'\S';

        my ($name, $value) = split '\s*:\s*', $line;

        $constMap->{$name} = $value;
    }
    return;
}

sub setRegisterMap
{
    my ($regMap, $regmapText) = @_;

    my $vectors = $regMap->{__vectors} ||= {};
    my $regBank = $regMap->{__regbank} ||= {};
    my %aliases;

    foreach my $line (split "\n", $regmapText)
    {
        # strip leading space
        $line =~ s|^\s+||;
        # strip comments
        $line =~ s{(?:#|//).*}{};
        # strip trailing space
        $line =~ s|\s+$||;
        # skip blank lines
        next unless $line =~ m'\S';

        my $auto  = $line =~ /~/;
        my $share = $line =~ /=/;

        my ($regNums, $regNames) = split '\s*[:~=]\s*', $line;

        my (@numList, @nameList, %vecAliases);
        foreach my $num (split '\s*,\s*', $regNums)
        {
            my ($start, $stop) = split '\s*\-\s*', $num;
            die "REGISTER_MAPPING Error: Bad register number or range: $num\nLine: $line\nFull Context:\n$regmapText\n" if grep m'\D', $start, $stop;
            push @numList, ($start .. $stop||$start);
        }
        foreach my $fullName (split '\s*,\s*', $regNames)
        {
            if ($fullName =~ m'^(\w+)<((?:\d+(?:\s*\-\s*\d+)?\s*\|?\s*)+)>(\w*)(?:\[([0-3])\])?$')
            {
                my ($name1, $name2, $bank) = ($1, $3, $4);
                foreach (split '\s*\|\s*', $2)
                {
                    my ($start, $stop) = split '\s*\-\s*';
                    foreach my $r (map "$name1$_$name2", $start .. $stop||$start)
                    {
                        # define an alias for use in vector instructions that omits the number portion
                        $aliases{$r} = "$name1$name2" unless exists $aliases{$r};
                        push @nameList, $r;
                        $regBank->{$r} = $bank if $auto && defined $bank;
                        warn "Cannot request a bank for a fixed register range: $fullName\n" if !$auto && defined $bank;
                    }
                }
            }
            elsif ($fullName =~ m'^(\w+)(?:\[([0-3])\])?$')
            {
                push @nameList, $1;
                $regBank->{$1} = $2 if $auto && defined $2;
                warn "Cannot request a bank for a fixed register range: $fullName\n" if !$auto && defined $2;
            }
            else
            {
                die "Bad register name: '$fullName' at: $line\n";
            }
        }
        die "Missmatched register mapping at: $line\n" if !$share && @numList < @nameList;
        die "Missmatched register mapping at: $line\n" if $share && @numList > 1;

        # detect if this list is monotonically ascending with no gaps
        my $i = 0;
        while ($i < $#numList-1)
        {
            last if $numList[$i] + 1 != $numList[$i+1];
            $i++;
        }
        my $ascending = $i+1 == $#numList;

        foreach my $n (0..$#nameList)
        {
            die "register defined twice: $nameList[$n]" if exists $regMap->{$nameList[$n]};

            if ($auto)
            {
                # assign possible values to be assigned on assembly
                $regMap->{$nameList[$n]} = \@numList;
            }
            elsif ($share)
            {
                # each name shares the same single register
                $regMap->{$nameList[$n]} = 'R' . $numList[0];
            }
            else
            {
                $regMap->{$nameList[$n]} = 'R' . $numList[$n];
                # flag any even register as a potential vector
                if ($ascending && ($numList[$n] & 1) == 0)
                {
                    # constrain potential range to vector alignment
                    my $end = $n + ($numList[$n] & 2 || $n + 3 > $#nameList ? 1 : 3);
                    if ($end <= $#nameList)
                    {
                        $vectors->{$nameList[$n]} = [ @nameList[$n .. $end] ];
                        #setup an alias for the base name without the number
                        if (exists $aliases{$nameList[$n]} && !exists $regMap->{$aliases{$nameList[$n]}})
                        {
                            $regMap->{$aliases{$nameList[$n]}}  = $regMap->{$nameList[$n]};
                            $vectors->{$aliases{$nameList[$n]}} = $vectors->{$nameList[$n]};
                            delete $aliases{$nameList[$n]};
                        }
                    }
                }
            }
        }
    }
    #print Dumper($regMap); exit(1);
}

sub preProcessLine
{
    # strip leading space
    $_[0] =~ s|^\s+||;

    # preserve comment but check for emptiness
    my $val = shift;

    # strip comments
    $val =~ s{(?:#|//).*}{};

    # skip blank lines
    return $val =~ m'\S';
}

# traverse the graph and count total descendants per node.
# only count unique nodes (by lineNum)
sub countUniqueDescendants
{
    my ($node, $edges) = @_;

    #print "P:$node->{inst}\n";

    if (my $children = $node->{children})
    {
        foreach my $child (grep $_->[1], @$children) # skip WaR deps and traversed edges
        {
            next if $edges->{"$node->{lineNum}^$child->[0]{lineNum}"}++;

            $node->{deps}{$_}++ foreach countUniqueDescendants($child->[0], $edges);
        }
        foreach my $child (grep !$_->[1], @$children) # WaR deps
        {
            next if $edges->{"$node->{lineNum}^$child->[0]{lineNum}"}++;

            1 foreach countUniqueDescendants($child->[0], $edges);
        }
    }
    else
    {
        return $node->{lineNum};
    }
    return ($node->{lineNum}, keys %{$node->{deps}});
}
# convert hash to count for easier sorting.
sub updateDepCounts
{
    my ($node, $edges) = @_;

    #warn "$node->{inst}\n";

    if (my $children = $node->{children})
    {
        foreach my $child (@$children)
        {
            next if $edges->{"$node->{lineNum}^$child->[0]{lineNum}"}++;
            updateDepCounts($child->[0], $edges);
        }
    }
    $node->{deps} = ref $node->{deps} ? keys %{$node->{deps}} : $node->{deps}+0;
}

# Detect register bank conflicts and calculate reuse stats
sub registerHealth
{
    my ($reuseHistory, $reuseFlags, $capData, $instAddr, $inst, $nowarn) = @_;

    my (@banks, @conflicts);

    foreach my $slot (qw(r8 r20 r39))
    {
        my $r = $capData->{$slot} or next;
        next if $r eq 'RZ';

        my $slotHist = $reuseHistory->{$slot} ||= {};

        $reuseHistory->{total}++;

        # if this register is in active reuse then ignore for bank conflict checking.
        if (exists $slotHist->{$r})
        {
            $reuseHistory->{reuse}++;
        }
        else
        {
            # extract number from reg and take the modulo-4 value.  This is the bank id.
            my $bank = substr($r,1) & 3;

            # check for conflict
            if ($banks[$bank] && $banks[$bank] ne $r)
            {
                push @conflicts, $banks[$bank] if !@conflicts;
                push @conflicts, $r;

                $reuseHistory->{conflicts}++;
            }
            $banks[$bank] = $r;
        }

        # update the history
        if ($reuseFlags & $reuseSlots{$slot})
            { $slotHist->{$r} = 1; }
        else
            { delete $slotHist->{$r};  }
    }
    if ($inst && @conflicts && !$nowarn)
    {
        printf "CONFLICT at 0x%04x (%s): $inst\n", $instAddr, join(',', @conflicts);
    }
    return scalar @conflicts;
}

1;

__END__

=head1 NAME

MaxAs::MaxAs - Assembler for NVIDIA Maxwell architecture

=head1 SYNOPSIS

    maxas.pl [opts]

=head1 DESCRIPTION

See the documentation at: https://github.com/NervanaSystems/maxas

=head1 SEE ALSO

See the documentation at: https://github.com/NervanaSystems/maxas


=head1 AUTHOR

Scott Gray, E<lt>sgray@nervanasys.com<gt>

=head1 COPYRIGHT AND LICENSE

The MIT License (MIT)

Copyright (c) 2014 Scott Gray

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

=cut

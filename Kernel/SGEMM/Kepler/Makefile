BINS := sgemm_nn_128x128 sgemm_nt_128x128 sgemm_tn_128x128 \
  sgemm_nn_128x128_vec sgemm_tn_128x128_vec sgemm_nt_128x128_vec
TARGETS := $(addsuffix .cubin, $(BINS))
TEMPLATES := $(addsuffix _template.cubin, $(BINS))

all: $(BINS)

$(BINS):
	nvcc -arch sm_35 -m 64 $@.cu -cubin -O3 -o $@_template.cubin
	KeplerAs.pl -i $@.sass $@_template.cubin $@.cubin

clean:
	rm $(TARGETS) $(TEMPLATES)

.PHONY:
	all clean

#utils
print-% : ; $(info $* is $(flavor $*) variable set to [$($*)]) @true           

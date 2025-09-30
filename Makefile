.PHONY: setup test format

setup:
	julia --project=. -e 'using Pkg; Pkg.instantiate()'

test:
	julia --project=. -e 'using Pkg; Pkg.test()'

format:
	julia -e 'using Pkg; Pkg.add("JuliaFormatter"); using JuliaFormatter; format(".")'


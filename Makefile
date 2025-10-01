.PHONY: setup test format ci

setup:
	julia --project=. -e 'using Pkg; Pkg.instantiate()'

test:
	julia --project=. -e 'using Pkg; Pkg.test()'

format:
	julia -e 'using Pkg; Pkg.add(PackageSpec(name="JuliaFormatter", version="1.0.59")); using JuliaFormatter; format(".")'

ci:
	# Mirror CI locally: pinned format check + tests
	julia -e 'using Pkg; Pkg.add(PackageSpec(name="JuliaFormatter", version="1.0.59")); using JuliaFormatter; format(".")'
	git diff --exit-code
	julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'

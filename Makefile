.PHONY: setup test format ci

# Install/instantiate all deps for the single project environment
setup:
	julia --project=. -e 'using Pkg; Pkg.instantiate()'

test:
	julia --project=. -e 'using Pkg; Pkg.test()'

# KISS: pin JuliaFormatter directly instead of a separate .format env.
# Julia lacks generic dev groups; we keep one env and a reproducible formatter.
format:
	julia -e 'using Pkg; Pkg.add(PackageSpec(name="JuliaFormatter", version="1.0.59")); using JuliaFormatter; format(".")'

ci:
	# Mirror CI locally: pinned format diff + tests (no separate env)
	julia -e 'using Pkg; Pkg.add(PackageSpec(name="JuliaFormatter", version="1.0.59")); using JuliaFormatter; format(".")'
	git diff --exit-code
	julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'

import pstats

p = pstats.Stats('main.profile')
p.strip_dirs().sort_stats("cumtime")
p.print_stats(20)

print "=====================  Callers ===================="
p.print_callers(10)

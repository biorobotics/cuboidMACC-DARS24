import timeit
def record_duration(phase,t_last,store,skip=False):
  now=timeit.default_timer()
  if not skip:
    store[phase]=now-t_last
  return now
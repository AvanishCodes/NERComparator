import multiprocessing as mp


def f(name: str="Denave"):
  print("Happy Independence Day from " + name)
  return

if __name__ == "__main__":
  p = mp.Process(target=f, args=("Denave",))
  p.start()
  p.join()
  exit(0)
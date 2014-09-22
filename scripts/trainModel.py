import subprocess
import sys
import os
import getopt
import copy

if __name__ == "__main__":
  opts, extraparams = getopt.gnu_getopt(sys.argv[1:], "", ["train_batch=", "test_batch=", "max_test_iter", "model=", "freq_snaps", "train_net=", "test_net=", "test_iter=", "test_interval=", "test_compute_loss=", "base_lr=", "display=", "max_iter=", "lr_policy=", "gamma=", "power=", "momentum=", "weight_decay=", "stepsize=", "snapshot=", "snapshot_prefix=", "snapshot_diff=", "solver_mode=", "device_id=", "random_seed="])
  nonSolverOpts = set(["max_test_iter", "model", "freq_snaps", "train_batch", "test_batch", "freeze_to", "reinit_from"])
  optDict = dict([(k[2:],v) for (k,v) in opts])
  print optDict

  if not "model" in optDict:
    raise Exception("Need to specify --model flag")
  task = optDict["model"]
  baseDir = os.path.abspath("../models/" + task) + "/"
  solverFile = baseDir + task + "_solver.prototxt"
  curOpts = [] 


  test_batch = 96
  if "test_iter" in optDict and "max_test_iter" in optDict:
    print "WARNING: \"test_iter\" was specified, ignoring \"max_test_iter\" flag"
    del optDict["max_test_iter"]
  if "max_test_iter" in optDict:
    with open("../data_info/" + task + "/val.txt", 'r') as f:
      noVal = f.read().strip().split('\n')
    test_iter = (noVal / test_batch) + 1
    #print "Changing \"test_iter\" from", optDict["test_iter"], "to", test_iter
    optDict["test_iter"] = test_iter
      
  if "snapshot" in optDict and "freq_snaps" in optDict:
    print "WARNING: \"snapshot\" was specified, ignoring \"freq_snaps\" flag"
    del optDict["freq_snaps"]
  if "freq_snaps" in optDict and "test_interval" in optDict:
    optDict["snapshot"] = optDict["test_interval"] 
    del optDict["freq_snaps"]

  test_interval = None
  with open(solverFile, 'r') as f:
    for line in f:
      opt, val = tuple(map(lambda x: x.strip(), line.strip().split(":")))
      if opt == "test_interval":
        test_interval = val
      if opt in optDict:
        print "Changing", opt, "from", val, "to", optDict[opt]
        val = optDict[opt]
        del optDict[opt]
      curOpts.append((opt,val))

  if "freq_snaps" in optDict:
    for i in range(len(curOpts)):
     if curOpts[i][0] == "snapshot":
       print "Changing \"snapshot\" from", curOpts[i][1], "to", test_interval
       curOpts[i] = (curOpts[i][0], test_interval)

  for k,v in optDict.items():
    print "Adding " + k + ": " + v + " to curOpts"
    curOpts.append((k,v))

  with open (solverFile, 'w') as f:
    for k,v in curOpts:
      if k not in nonSolverOpts:
        f.write(k + ": " + v + "\n")
  
  cmd = "cd " + baseDir + "; nohup ./fine_" + task + ".sh 2>&1 | tee train_output.log" 
  p = subprocess.Popen(cmd, shell=True, stdout = subprocess.PIPE, stderr=subprocess.STDOUT)

  lastLine = None 
  newAcc = 0 
  bestSnap = None
  bestAcc = 0
  delete = []
  while True:
    out = p.stdout.readline()
    if out == '' and p.poll() != None:
      break
    if out != '':
      if "Test score #0" in out:
        newAcc = float(out.strip().split()[-1])
      if "Snapshotting to" in out:
        newSnap = out.strip().split()[-1]
        if newAcc > bestAcc:
          if bestSnap:
            delete += [baseDir + bestSnap, baseDir + bestSnap+".solverstate"]
          bestAcc = newAcc
          bestSnap = newSnap
        else:
          delete += [baseDir + newSnap, baseDir + newSnap+".solverstate"]
      if len(delete) > 0 and (not "Snapshotting" in out):
        for f in delete:
          os.remove(f)
        delete = []
      sys.stdout.write(out)
      sys.stdout.flush()
      lastLine = out
 
# val_batch_size = 96
# test_iter = (val_set_size / val_batch_size) + 1









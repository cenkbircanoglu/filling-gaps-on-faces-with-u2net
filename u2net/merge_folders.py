import subprocess as sbp
import os

path = input('Please enter a path\n')
fol = os.listdir(path)
p2 = input('Please enter a path\n')

for i in fol:
    p1 = os.path.join(path, i)
    p3 = 'cp -r ' + p1 + ' ' + p2 + '/.'
    sbp.Popen(p3, shell=True)

import os
for i in os.listdir('./log'):
    if os.path.isdir(os.path.join('./log',i)):
        f=open(os.path.join('./log',i,'log.txt'),'r').read()
        if len(f)<1000:
            os.system('rm -r '+os.path.join('./log',i))
            print('del ',i)

        

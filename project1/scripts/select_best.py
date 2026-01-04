import json 

lrs = json.loads(r'''{{inputs.parameters.lr_list}}''')
accs = json.loads(r'''{{inputs.parameters.acc_list}}''')

best_i = max(range(len(accs)), key=lambda i: float(accs[i]))
print("all:", list(zip(lrs,accs)))
print("BEST:", "lr=",lrs[best_i],"acc=",accs[best_i])

params = {
    'lr': lrs[best_i],
    'acc': accs[best_i]
}

with open("/tmp/params.json","w") as f:
    json.dump(params,f)

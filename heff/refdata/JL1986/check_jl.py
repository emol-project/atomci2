import csv, sys, sympy as sp
LW={'S':0,'P':1,'D':2,'F':3,'G':4,'H':5,'I':6}
def val(x):
    x=x.strip(); 
    if 's' in x: c,r=x.split('*s'); return sp.Integer(c)*sp.sqrt(int(r))
    return sp.Integer(x)
def load(fn):
    rows=[r for r in open(fn) if not r.startswith('#')]
    rd=list(csv.reader(rows)); hdr=rd[0][2:]; data=[]
    for r in rd[1:]:
        S=int(r[0].split(':')[1][0]); L=LW[r[0][-1]]
        w=(2*L+1)*S*(1 if r[0]==r[1] else 2)
        data.append((w,[val(x) for x in r[2:]]))
    return hdr,data
norm={r[0]:r for r in csv.reader(l for l in open('JL1986_table9_norm.csv') if not l.startswith('#'))}
col={'d4':5,'d5':6}
ok=True
for fn,d in [('JL1986_table7_d4.csv','d4'),('JL1986_table8_d5.csv','d5')]:
    hdr,data=load(fn); n=len(hdr)
    for i in range(n):
        for j in range(i,n):
            s=sp.nsimplify(sum(w*v[i]*v[j] for w,v in data))
            if i==j:
                A=int(norm[hdr[i]][col[d]])
                if s!=A: ok=False; print(d,hdr[i],'norm',s,'table9',A)
            elif s!=0: ok=False; print(d,hdr[i],hdr[j],'overlap',s)
print('all checks passed' if ok else 'MISMATCH')

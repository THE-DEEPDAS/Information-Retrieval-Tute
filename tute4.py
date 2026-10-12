"""Assignment 4 mini search engine."""
import csv, math, re
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).parent
DATA=ROOT/'data/assignment4_documents.csv'
OUT=ROOT/'results'
QUERY='machine learning applications in healthcare'
DOCS=[('D1','Machine learning diagnosis','Machine learning helps doctors detect disease from medical images and patient data.',1),('D2','Healthcare AI analysis','Artificial intelligence analyzes healthcare records and improves treatment.',1),('D3','Computer networks history','History of network protocols packet switching and the Internet.',0),('D4','Deep learning disease prediction','Deep neural networks predict disease risk from clinical data.',1),('D5','Database management','Relational databases provide SQL queries and structured storage.',0),('D6','Machine learning applications','Supervised learning includes classification prediction and recommendation.',1),('D7','Healthcare information systems','Systems store patient records and support hospitals.',1),('D8','Computer graphics','Rendering geometric transformations and image synthesis.',0),('D9','Statistics in medicine','Statistical analysis supports clinical trials diagnosis and healthcare.',1),('D10','Operating systems','Processes memory files and scheduling are operating system topics.',0)]
def tok(s): return re.findall(r'[a-z]+',s.lower())
def cosine(a,b):
 d=sum(x*y for x,y in zip(a,b)); n=math.sqrt(sum(x*x for x in a)*sum(y*y for y in b)); return d/n if n else 0
def metrics(rank,rel):
 z=[int(x in rel) for x in rank]; p=sum(z)/len(z); r=sum(z)/len(rel); f=2*p*r/(p+r) if p+r else 0; rr=next((1/(i+1) for i,x in enumerate(z) if x),0); ap=sum(sum(z[:i+1])/(i+1) for i,x in enumerate(z) if x)/len(rel); dcg=sum(x/math.log2(i+2) for i,x in enumerate(z)); ideal=sum(1/math.log2(i+2) for i in range(min(len(rel),len(z)))); return p,r,f,rr,ap,dcg/ideal if ideal else 0
def main():
 DATA.parent.mkdir(exist_ok=True)
 if not DATA.exists():
  with DATA.open('w',newline='',encoding='utf8') as f: csv.writer(f).writerows([['id','title','content','relevant']]+DOCS)
 rows=list(csv.DictReader(DATA.open(encoding='utf8'))); texts=[r['title']+' '+r['content'] for r in rows]; ts=[tok(x) for x in texts]; q=tok(QUERY); df=Counter(t for x in ts for t in set(x)); terms=sorted(df)
 def v(x):
  c=Counter(x); return [(c[t]/len(x))*math.log((len(rows)+1)/(df[t]+1))+1 for t in terms]
 X=[v(x) for x in ts]; qv=v(q); tf=sorted(zip(rows,[cosine(x,qv) for x in X]),key=lambda x:x[1],reverse=True); qt=set(q)
 # Probabilistic/BIM score: log-odds weight for each query term, as guided by PRP.
 prob=[]
 for r,x in zip(rows,ts):
  present=set(x); score=sum(math.log((len(rows)-df[t]+.5)/(df[t]+.5)) for t in qt if t in present)
  prob.append((r,score))
 probabilistic=sorted(prob,key=lambda x:x[1],reverse=True)
 rel={r['id'] for r in rows if r['relevant']=='1'}; OUT.mkdir(exist_ok=True)
 with (OUT/'assignment4_results.txt').open('w',encoding='utf8') as f:
  f.write('Query: '+QUERY+'\n\nTF-IDF/VSM top five:\n'); f.write('\n'.join(f'{i+1}. {r["id"]} {r["title"]} score={s:.4f}' for i,(r,s) in enumerate(tf[:5]))); f.write('\n\nProbabilistic/BIM top five:\n'); f.write('\n'.join(f'{i+1}. {r["id"]} {r["title"]} score={s:.4f}' for i,(r,s) in enumerate(probabilistic[:5]))); f.write('\n\nEvaluation (P R F MRR MAP NDCG):\n'); f.write('TF-IDF '+' '.join(f'{x:.4f}' for x in metrics([r['id'] for r,s in tf],rel))+'\nProbabilistic '+' '.join(f'{x:.4f}' for x in metrics([r['id'] for r,s in probabilistic],rel)))
 print((OUT/'assignment4_results.txt').read_text())
if __name__=='__main__': main()

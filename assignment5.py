"""Assignment 5: TF-IDF clustering, topic summaries, anomaly detection and retrieval."""
import csv,math,re,random
from collections import Counter
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
ROOT=Path(__file__).parent; DATA=ROOT/'data/assignment5_documents.csv'; OUT=ROOT/'results'; QUERY='documents about protecting computer networks'
GROUPS={'AI':['neural networks deep learning images','natural language processing text models','machine learning prediction data','computer vision image recognition','speech recognition neural language'],'Cybersecurity':['network security firewalls protect computer networks','cryptography encryption secure communication','cyberattacks malware intrusion detection','authentication access control security','privacy security threat incident response'],'Databases':['relational databases SQL tables queries','database indexing improves performance','transactions concurrency management','data warehouse analytics information','NoSQL document databases distributed storage'],'Networking':['computer networks routing TCP IP protocols','wireless networks communication connectivity','network topology packet switching routers','Internet architecture distributed services','cloud networking virtual machines']}
def tok(s): return re.findall(r'[a-z]+',s.lower())
def cos(a,b):
 d=sum(x*y for x,y in zip(a,b)); n=math.sqrt(sum(x*x for x in a)*sum(y*y for y in b)); return d/n if n else 0
def main():
 DATA.parent.mkdir(exist_ok=True)
 if not DATA.exists():
  with DATA.open('w',newline='',encoding='utf8') as f:
   w=csv.writer(f); w.writerow(['id','topic','text']); n=1
   for topic,docs in GROUPS.items():
    for text in docs: w.writerow([f'D{n}',topic,text]); n+=1
 rows=list(csv.DictReader(DATA.open(encoding='utf8'))); ts=[tok(r['text']) for r in rows]; df=Counter(t for x in ts for t in set(x)); terms=sorted(df)
 def vec(x):
  c=Counter(x); return [(c[t]/len(x))*math.log((len(rows)+1)/(df[t]+1)) for t in terms]
 X=[vec(x) for x in ts]; random.seed(7); cent=[X[i][:] for i in [0,5,10,15]]
 for _ in range(20):
  labels=[max(range(4),key=lambda j:cos(x,cent[j])) for x in X]
  for j in range(4):
   a=[x for x,l in zip(X,labels) if l==j]
   if a: cent[j]=[sum(x[k] for x in a)/len(a) for k in range(len(terms))]
 labels=[max(range(4),key=lambda j:cos(x,cent[j])) for x in X]; sims=[cos(x,cent[l]) for x,l in zip(X,labels)]; q=vec(tok(QUERY)); ranking=sorted(zip(rows,[cos(x,q) for x in X]),key=lambda z:z[1],reverse=True); relevant={r['id'] for r in rows if r['topic'] in ('Cybersecurity','Networking')}; top=[r['id'] for r,s in ranking[:8]]; hit=len(set(top)&relevant); OUT.mkdir(exist_ok=True)
 with (OUT/'assignment5_results.txt').open('w',encoding='utf8') as f:
  f.write(f'Documents: {len(rows)} | Clusters: 4 | Query: {QUERY}\n\nClusters and topic keywords:\n')
  for j in range(4):
   group=[r for r,l in zip(rows,labels) if l==j]; words=Counter(w for r in group for w in tok(r['text'])); f.write(f'Cluster {j+1}: {[r["id"] for r in group]} | {", ".join(w for w,_ in words.most_common(5))}\n')
  f.write('\nAnomaly candidates (similarity < 0.55): '+str([rows[i]['id'] for i,s in enumerate(sims) if s<.55])+'\n\nTop retrieval:\n'); f.write('\n'.join(f'{i+1}. {r["id"]} ({r["topic"]}) score={s:.4f}' for i,(r,s) in enumerate(ranking[:8]))); f.write(f'\n\nEvaluation: Precision@8={hit/8:.4f}, Recall@8={hit/len(relevant):.4f}\n')
 # Proper PCA: center the complete TF-IDF matrix, then project onto its
 # first two eigenvectors. This prevents documents from collapsing on one
 # or two arbitrary vocabulary columns.
 matrix=np.array(X, dtype=float)
 centered=matrix-matrix.mean(axis=0)
 covariance=np.cov(centered, rowvar=False)
 eigenvalues,eigenvectors=np.linalg.eigh(covariance)
 components=eigenvectors[:,np.argsort(eigenvalues)[-2:]]
 projection=centered@components
 x=projection[:,0]; y=projection[:,1]
 fig, axes_plot=plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
 axes_plot=axes_plot.ravel()
 for cluster in range(4):
  ids=[i for i,label in enumerate(labels) if label==cluster]
  ax=axes_plot[cluster]
  ax.scatter([x[i] for i in ids],[y[i] for i in ids],s=80,color=f'C{cluster}')
  for i in ids: ax.annotate(rows[i]['id'],(x[i],y[i]),xytext=(5,5),textcoords='offset points')
  ax.set_title(f'Cluster {cluster+1}')
  ax.grid(True)
 fig.suptitle('Assignment 5: 2D PCA Projection by Cluster',fontsize=16)
 fig.supxlabel('Principal component 1'); fig.supylabel('Principal component 2')
 fig.tight_layout()
 projection_path=OUT/'assignment5_2d_projection.png'
 plt.savefig(projection_path,dpi=150,bbox_inches='tight'); plt.close()
 print(f'2D projection saved to: {projection_path.resolve()}')
 print((OUT/'assignment5_results.txt').read_text())
if __name__=='__main__': main()

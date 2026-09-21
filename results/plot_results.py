"""Aggregate independent builds and export recall/QPS and construction plots."""
from pathlib import Path
from collections import defaultdict
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
ROOT=Path(__file__).resolve().parent
manifest=json.loads((ROOT/'manifest.json').read_text())
build_count=manifest['independent_builds_per_variant']
selected_repeats=range(manifest.get('first_repeat',1),manifest.get('first_repeat',1)+build_count)
COLORS={'current':'#087f8c','upstream':'#cb6139'}
LABELS={'current':'Current HVS','upstream':'Original HVS'}
query=[];build=defaultdict(dict)
for path in sorted((ROOT/'raw').glob('*.log')):
 meta_path=path.with_suffix('.json')
 if not meta_path.exists():continue
 meta=json.loads(meta_path.read_text())
 if meta['returncode']!=0:continue
 repeat=meta['repeat'];variant=meta['variant']
 if repeat not in selected_repeats:continue
 for line in path.read_text().splitlines():
  if not line.startswith('PERF '):continue
  row=json.loads(line[5:]);row['repeat']=repeat
  if row['kind']=='query':query.append(row)
  else:build[(variant,repeat)].update(row)
if not query:raise RuntimeError('No completed query runs')
expected={(variant,repeat,threads,k,ef)
 for variant in ['upstream','current'] for repeat in selected_repeats
 for threads in [1,16] for k in [1,10,100]
 for ef in [1,2,4,8,12,16,24,32,48,64,96,128,192,256,384,512] if ef>=k}
actual={(r['variant'],r['repeat'],r['threads'],r['k'],r['ef']) for r in query}
assert actual==expected, f'Incomplete or unexpected configurations: missing {expected-actual}; extra {actual-expected}'
assert all(r['nq']==10000 and r['seconds']>0 and r['batches']>=1 for r in query)
# Concurrent queries on the same index must preserve recall for every setting.
recalls=defaultdict(set)
for r in query:recalls[(r['variant'],r['repeat'],r['k'],r['ef'])].add(r['recall'])
assert all(len(v)==1 for v in recalls.values()), 'Recall differs between 1 and 16 query threads'
def save_csv(name,rows):
 columns=list(dict.fromkeys(k for row in rows for k in row))
 with (ROOT/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(rows)
save_csv('query_samples.csv',query)
build_rows=[]
for (variant,repeat),row in sorted(build.items()):
 if 'build_api_s' not in row:continue
 if variant=='upstream':
  if 'index_load_s' not in row:continue
  row['ready_s']=row['build_api_s']+row['index_load_s']
 build_rows.append({k:row[k] for k in ['variant','repeat','build_threads','input_load_s','build_api_s','native_build_s','index_load_s','ready_s'] if k in row})
save_csv('build_times.csv',build_rows)
assert len(build_rows)==2*build_count
per_build=defaultdict(list)
for row in query:per_build[(row['variant'],row['repeat'],row['threads'],row['k'],row['ef'])].append(row)
points=[]
for (variant,repeat,threads,k,ef),rows in per_build.items():
 assert len(rows)==1
 assert len({r['recall'] for r in rows})==1
 points.append({'variant':variant,'repeat':repeat,'threads':threads,'k':k,'ef':ef,'recall':rows[0]['recall'],'qps':float(np.median([r['qps'] for r in rows]))})
save_csv('query_by_build.csv',points)
groups=defaultdict(list)
for row in points:groups[(row['variant'],row['threads'],row['k'],row['ef'])].append(row)
summary=[]
for (variant,threads,k,ef),rows in sorted(groups.items()):
 r=np.array([v['recall'] for v in rows]);q=np.array([v['qps'] for v in rows])
 summary.append({'variant':variant,'threads':threads,'k':k,'ef':ef,'builds':len(rows),
 'recall':float(np.median(r)),'recall_min':float(r.min()),'recall_max':float(r.max()),
 'qps':float(np.median(q)),'qps_min':float(q.min()),'qps_max':float(q.max())})
save_csv('recall_vs_qps.csv',summary)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','pdf.fonttype':42,'svg.fonttype':'none'})
fig,axes=plt.subplots(2,3,figsize=(16,9.3))
for row,threads in enumerate([1,16]):
 for col,k in enumerate([1,10,100]):
  ax=axes[row,col]
  for variant in ['upstream','current']:
   rows=sorted((p for p in summary if p['variant']==variant and p['threads']==threads and p['k']==k),key=lambda p:p['ef'])
   if not rows:continue
   x=np.array([p['recall'] for p in rows]);y=np.array([p['qps'] for p in rows])
   low=np.array([p['qps_min'] for p in rows]);high=np.array([p['qps_max'] for p in rows])
   ax.plot(x,y,label=LABELS[variant],color=COLORS[variant],marker='o' if variant=='current' else 's',markersize=4.5,lw=1.8)
   if build_count>1:
    ax.fill_between(x,low,high,color=COLORS[variant],alpha=.12)
    ax.errorbar(x,y,xerr=[x-np.array([p['recall_min'] for p in rows]),np.array([p['recall_max'] for p in rows])-x],fmt='none',ecolor=COLORS[variant],alpha=.4,capsize=2)
  ax.set_title(f'Recall@{k} | {threads} query thread'+('s' if threads>1 else ''))
  ax.set_xlabel(f'Recall@{k}');ax.set_ylabel('QPS (queries / second)')
  ax.set_yscale('log');ax.yaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x:,.0f}'))
  ax.set_xlim(left=max(0,min([p['recall_min'] for p in summary if p['k']==k])-.025),right=1.005)
  ax.grid(True,which='major',alpha=.22);ax.grid(True,which='minor',axis='y',alpha=.07)
  ax.legend(frameon=False,loc='upper right')
fig.suptitle('SIFT1M: current HVS vs original HVS',fontsize=21,fontweight='bold',y=.995)
fig.text(.5,.952,'1,000,000 base vectors · 10,000 queries · 128 dimensions · T=1, M=16, efConstruction=500',ha='center',fontsize=11)
sampling=('One build per variant; one timed sample of all 10,000 queries per configuration.' if build_count==1
 else f'Median across {build_count} builds; shading = build-to-build range.')
fig.text(.5,.018,f'Full query path includes rotation and distance tables. {sampling}\nHistorical run: 16 physical build cores, one NUMA socket, no interleave; same GCC 14.3 -O3 and OpenCV/Eigen/zlib, OpenCV 1 thread.',ha='center',fontsize=9,color='#444444')
fig.tight_layout(rect=[0,.065,1,.925])
for ext in ['png','pdf','svg']:fig.savefig(ROOT/f'recall_vs_qps.{ext}',dpi=200,bbox_inches='tight')
plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(11.5,4.8))
for ax,key,title in zip(axes,['build_api_s','ready_s'],['Construction API wall time','Time until index is query-ready']):
 for i,variant in enumerate(['upstream','current']):
  values=np.array([b[key] for b in build_rows if b['variant']==variant]);median=float(np.median(values))
  ax.bar(i,median,color=COLORS[variant],width=.55,alpha=.9)
  if build_count>1:ax.errorbar(i,median,yerr=[[median-values.min()],[values.max()-median]],fmt='none',ecolor='#333333',capsize=6)
  ax.text(i,values.max()+max(values.max()*.025,1),f'{median:.2f} s',ha='center',fontweight='bold')
 original=float(np.median([b[key] for b in build_rows if b['variant']=='upstream']))
 current=float(np.median([b[key] for b in build_rows if b['variant']=='current']))
 ax.set_xticks([0,1],[LABELS['upstream'],LABELS['current']]);ax.set_ylabel('Seconds');ax.set_title(f'{title}\nCurrent: {(current/original-1)*100:+.2f}% time');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.margins(y=.2)
fig.suptitle('SIFT1M construction | 16 physical cores',fontsize=17,fontweight='bold')
construction_sampling='One build per variant.' if build_count==1 else 'Median and min–max of independent builds.'
fig.text(.5,.012,f'{construction_sampling} Both include quantizer training and encoding. Original API includes native I/O; current API is in memory.\nQuery-ready time adds current input loading or original index/metadata reload. Warm file cache.\nHistorical run: one NUMA socket, no interleave; these are not 64-thread/interleave measurements.',ha='center',fontsize=9)
fig.tight_layout(rect=[0,.14,1,.93])
for ext in ['png','pdf','svg']:fig.savefig(ROOT/f'build_time.{ext}',dpi=200,bbox_inches='tight')
plt.close(fig)
# Matched recall: interpolate log(QPS) between measured points.
matched=[]
for threads in [1,16]:
 for k in [1,10,100]:
  for target in [.90,.95,.99]:
   values={}
   for variant in ['upstream','current']:
    rows=sorted((p for p in summary if p['variant']==variant and p['threads']==threads and p['k']==k),key=lambda p:p['recall'])
    best={}
    for p in rows:best[p['recall']]=max(best.get(p['recall'],0),p['qps'])
    x=np.array(sorted(best));y=np.array([best[t] for t in x])
    if len(x) and x[0]<=target<=x[-1]:values[variant]=float(np.exp(np.interp(target,x,np.log(y))))
   if len(values)==2:matched.append({'threads':threads,'k':k,'target_recall':target,'upstream_qps':values['upstream'],'current_qps':values['current'],'speedup':values['current']/values['upstream'],'method':'log-QPS interpolation between measured recall points'})
save_csv('matched_recall.csv',matched)
(ROOT/'summary.json').write_text(json.dumps({'builds':build_rows,'curves':summary,'matched_recall':matched},indent=2)+'\n')
print('Saved recall_vs_qps and build_time PNG/PDF/SVG, CSV files, and summary.json')

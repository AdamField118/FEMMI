"""Regenerate the frozen pre-acceleration BEM fixture using a clean baseline."""
import argparse
import subprocess
import sys
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--source-root',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
args=p.parse_args()
root=args.source_root.resolve()
commit=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
if commit!='fa079a9468a19cb1713f814304270723f2025e44':
    raise SystemExit('expected baseline fa079a9')
if subprocess.check_output(['git','-C',str(root),'status','--porcelain','--untracked-files=no'],text=True):
    raise SystemExit('baseline has tracked changes')
sys.path.insert(0,str(root))
import numpy as np
from femmi.bem_hp import build_boundary_mesh,assemble_single_layer_hp,assemble_double_layer_hp
from femmi.bem import assemble_single_layer,assemble_double_layer
from pathlib import Path
out={}
points=np.array([[.1,-.2],[1.7,-.1],[2.,.8],[.4,1.3],[-.3,.7]])
out['points']=points
for degree in [3,5]:
 for quad in [7,25]:
  for scale in [.3,2.]:
   b=build_boundary_mesh(points*scale,degree)
   prefix=f'd{degree}_q{quad}_s{scale}'
   out[prefix+'_V']=assemble_single_layer_hp(b,degree,quad)
   out[prefix+'_K']=assemble_double_layer_hp(b,degree,quad)
   if degree==3:
    out[prefix+'_P3V']=assemble_single_layer(b,quad)
    out[prefix+'_P3K']=assemble_double_layer(b,quad)
p=args.output
p.parent.mkdir(exist_ok=True)
np.savez_compressed(p,**out)

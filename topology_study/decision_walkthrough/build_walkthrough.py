"""Render twelve checks individually and as three four-panel sheets."""
from drawing import *
from checks_01_06 import DRAWERS as FIRST
from checks_07_10 import DRAWERS as SECOND
from checks_11_12 import DRAWERS as THIRD

DRAWERS=FIRST|SECOND|THIRD
assert sorted(DRAWERS)==list(range(1,13))
style()
out=ROOT/'figures';out.mkdir(exist_ok=True)
for number,draw in DRAWERS.items():save_single(number,draw)
names=['Define the winding choices','Plan the jumps and returns','Validate, search and select']
for page in range(3):
    fig,axes=plt.subplots(2,2,figsize=(12,12))
    for number,ax in zip(range(4*page+1,4*page+5),axes.flat):DRAWERS[number](ax)
    fig.suptitle(names[page],fontsize=23,x=.045,ha='left',y=.975,color=INK)
    fig.text(.05,.017,'Schematics • jumps first, B0 error second • no optimized field results claimed',fontsize=14,color=MUTED)
    fig.subplots_adjust(left=.025,right=.98,top=.935,bottom=.052,wspace=.07,hspace=.04)
    for suffix in ('png','svg'):fig.savefig(out/f'checks_{page+1}.{suffix}',dpi=160)
    plt.close(fig)
print(f'Rendered {len(DRAWERS)} checks and 3 overview sheets in {out}')

"""Shared drawing primitives for exact, reproducible explanatory schematics.

World coordinates are arbitrary schematic lengths. Orthographic projection is
used consistently; figures are not manufacturing drawings or optimized routes.
"""
from pathlib import Path
import textwrap
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle, Circle

ROOT = Path(__file__).resolve().parent
INK = '#193745'
MUTED = '#4c626d'
TEAL = '#087f8c'
ORANGE = '#d58018'
PURPLE = '#8655b2'
RED = '#ba413d'
GREEN = '#24764e'
GRID = '#dbe5e8'
PAPER = '#ffffff'


def style():
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':15,
                         'text.color':INK,'figure.facecolor':PAPER,
                         'savefig.facecolor':PAPER,'svg.fonttype':'none'})


def setup(ax, number, title, caption):
    ax.set_xlim(0,100); ax.set_ylim(0,100); ax.set_aspect('equal'); ax.axis('off')
    ax.text(2,98,f'{number:02d}  {title}',fontsize=19,weight='normal',va='top')
    wrapped='\n'.join(textwrap.fill(row,width=42) for row in caption.split('\n'))
    ax.text(2,4,wrapped,fontsize=14.5,va='bottom',linespacing=1.35,color=MUTED)


def text(ax,x,y,s,size=15,color=INK,ha='center',va='center',weight='normal'):
    if '$' not in s:
        s='\n'.join(textwrap.fill(row,width=43) for row in s.split('\n'))
    return ax.text(x,y,s,fontsize=size,color=color,ha=ha,va=va,weight=weight,
                   bbox={'facecolor':PAPER,'edgecolor':'none','pad':1.2})


def arrow(ax,a,b,color=INK,lw=1.8,scale=13,style='-|>'):
    patch=FancyArrowPatch(a,b,arrowstyle=style,mutation_scale=scale,
                          linewidth=lw,color=color,zorder=8)
    ax.add_patch(patch)
    return patch


def line(ax,p,color=TEAL,lw=2.4,ls='-',alpha=1):
    p=np.asarray(p)
    ax.plot(p[:,0],p[:,1],color=color,lw=lw,ls=ls,alpha=alpha,solid_capstyle='round')


def mark(ax,p,label=None,color=INK,marker='o',offset=(2,2)):
    ax.plot(*p,marker=marker,color=color,ms=5,zorder=9)
    if label:text(ax,p[0]+offset[0],p[1]+offset[1],label,size=14,ha='left')


def curve(control,n=80):
    p=np.asarray(control,dtype=float); t=np.linspace(0,1,n)[:,None]
    assert len(p)==4
    return (1-t)**3*p[0]+3*(1-t)**2*t*p[1]+3*(1-t)*t*t*p[2]+t**3*p[3]


def arc(r,center=(0,0,0),direction=1,seam=0,gap=.18,n=170):
    a=seam+direction*np.linspace(gap,2*np.pi-gap,n)
    c=np.asarray(center)
    return c+np.column_stack((r*np.cos(a),r*np.sin(a),np.zeros(n)))


def circle(r,center=(0,0,0),n=170):
    a=np.linspace(0,2*np.pi,n); c=np.asarray(center)
    return c+np.column_stack((r*np.cos(a),r*np.sin(a),np.zeros(n)))


def spiral(radii=(28,21,14),center=(0,0,0),direction=1,seam=0):
    result=[]
    for r in radii:
        q=arc(r,center,direction,seam)
        if result:
            end=result[-1][1][-1]; start=q[0]
            ta=(result[-1][1][-1]-result[-1][1][-2]); ta/=np.linalg.norm(ta)
            tb=q[1]-q[0]; tb/=np.linalg.norm(tb)
            h=.3*np.linalg.norm(start-end)
            result.append(('join',curve([end,end+h*ta,start-h*tb,start])))
        result.append(('turn',q))
    return result


class View:
    def __init__(self,ax,box=(8,23,84,61),bounds=(-42,42,-36,36,0,18)):
        self.ax=ax; self.box=box
        az=np.deg2rad(-20); el=np.deg2rad(48)
        self.matrix=np.array([[np.cos(az),-np.sin(az),0],
                              [np.sin(el)*np.sin(az),np.sin(el)*np.cos(az),np.cos(el)]])
        xmin,xmax,ymin,ymax,zmin,zmax=bounds
        self.bounds=bounds
        corners=np.array([[x,y,z] for x in (xmin,xmax) for y in (ymin,ymax) for z in (zmin,zmax)])
        pr=corners@self.matrix.T
        self.center=(pr.min(0)+pr.max(0))/2
        self.scale=min(box[2]/np.ptp(pr[:,0]),box[3]/np.ptp(pr[:,1]))

    def p(self,points):
        xy=np.asarray(points)@self.matrix.T
        return (xy-self.center)*self.scale+np.array([self.box[0]+self.box[2]/2,self.box[1]+self.box[3]/2])

    def path(self,points,color=TEAL,lw=2.6,ls='-',alpha=1):
        line(self.ax,self.p(points),color,lw,ls,alpha)

    def point(self,point,label=None,**kwargs):
        mark(self.ax,self.p(point),label,**kwargs)

    def grid(self):
        xmin,xmax,ymin,ymax,*_=self.bounds
        for x in np.linspace(xmin,xmax,9):self.path([[x,ymin,0],[x,ymax,0]],GRID,.6)
        for y in np.linspace(ymin,ymax,9):self.path([[xmin,y,0],[xmax,y,0]],GRID,.6)

    def winding(self,pieces,arrows=True,terminals=False):
        for kind,p in pieces:
            self.path(p,TEAL if kind=='turn' else ORANGE,2.7)
            if arrows and kind=='turn':
                k=len(p)//3
                self.direction(p[k],p[k+6]-p[k],length=5)
        if terminals:
            self.point(pieces[0][1][0],'A',offset=(2,-3))
            self.point(pieces[-1][1][-1],'B',marker='s',offset=(2,3))

    def direction(self,p,tangent,length=6,color=INK):
        p=np.asarray(p); tangent=np.asarray(tangent); tangent=tangent/np.linalg.norm(tangent)
        a,b=self.p(np.array([p-.5*length*tangent,p+.5*length*tangent]))
        arrow(self.ax,a,b,color,1.6,12)

    def b0(self,p=(-32,-30,0),length=18):
        a,b=self.p(np.array([p,np.asarray(p)+[length,0,0]]))
        arrow(self.ax,a,b,INK,2.0,14)
        text(self.ax,b[0]+3,b[1],r'$B_0$',size=16,ha='left')


def elevated_transfer(x0=-33,x1=33,y=0,height=12,humps=1,n=300):
    t=np.linspace(0,1,n)
    if humps==1:
        # Constant-height middle with zero-slope rise and landing.
        z=np.full_like(t,height)
        left=t<.15; right=t>.85
        z[left]=height*(1-np.cos(np.pi*t[left]/.15))/2
        z[right]=height*(1-np.cos(np.pi*(1-t[right])/.15))/2
    else:
        z=height*np.sin(np.pi*humps*t)**2
    return np.column_stack((x0+(x1-x0)*t,np.full(n,y),z))


def box(ax,x,y,w,h,label,color=INK,size=15):
    ax.add_patch(Rectangle((x,y),w,h,facecolor=PAPER,edgecolor=color,lw=1.4))
    text(ax,x+w/2,y+h/2,label,size,color)


def save_single(number,draw):
    style(); fig,ax=plt.subplots(figsize=(6,6))
    fig.subplots_adjust(left=.03,right=.97,top=.97,bottom=.03)
    draw(ax)
    out=ROOT/'figures';out.mkdir(exist_ok=True)
    for suffix in ('png','svg'):fig.savefig(out/f'check_{number:02d}.{suffix}',dpi=160)
    plt.close(fig)

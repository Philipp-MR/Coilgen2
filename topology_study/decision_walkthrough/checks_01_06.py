"""The first six routing checks, drawn as scientific schematics."""
from drawing import *


def draw1(ax):
    setup(ax,1,'Fix the requirements',
          'Fix terminals, conductor and field limits.\nKeep them identical across candidates.')
    v=View(ax,box=(6,30,88,56),bounds=(-40,40,-34,34,-16,12));v.grid()
    roi=np.array([[-12,-12,-14],[12,-12,-14],[12,12,-14],[-12,12,-14],[-12,-12,-14]])
    v.path(roi,MUTED,1.4,'--')
    for r in (28,21,14):v.path(circle(r),TEAL,2.4)
    for p,name in (((-34,-29,0),'A'),((-24,-29,0),'B')):v.point(p,name,offset=(-2,-5))
    v.b0((-31,30,0),24)
    q=v.p([12,12,-14]);text(ax,80,35,'ROI',size=15)
    arrow(ax,(74,36),q,MUTED,1.0,8)
    text(ax,50,23,r'wire width   clearance   $R_{min}$',size=15)


def draw2(ax):
    setup(ax,2,'Read topology and direction',
          'Separate nesting relationships\nfrom the required current direction.')
    v=View(ax,box=(5,35,90,46),bounds=(-57,55,-27,27,0,5));v.grid()
    for center,radii,direction in (((-29,0,0),(22,15),1),((29,0,0),(19,12),-1)):
        for r in radii:
            p=circle(r,center);v.path(p)
            th=2.1;pt=np.array(center)+[r*np.cos(th),r*np.sin(th),0]
            v.direction(pt,direction*np.array([-np.sin(th),np.cos(th),0]),7)
    text(ax,26,81,'Family A',size=16)
    text(ax,76,81,'Family B',size=16)
    text(ax,26,29,'outer  →  inner',size=14.5)
    text(ax,76,29,'outer  →  inner',size=14.5)
    text(ax,50,21,'Containment only — no electrical branch',size=14.5)


def draw3(ax):
    setup(ax,3,'Wind adjacent turns',
          'Keep each turn’s circulation.\nThe inner terminal still needs a return.')
    for i,x in enumerate((1,55)):
        v=View(ax,(x,33,43,45),(-34,34,-34,34,0,2));v.grid()
        if i==0:
            for r in (28,21,14):
                v.path(circle(r));v.direction([0,r,0],[-1,0,0],6)
        else:v.winding(spiral(),terminals=True)
    arrow(ax,(47,56),(53,56),INK,1.8,14)
    text(ax,23,82,'Separate loops',size=16)
    text(ax,77,82,'One spiral',size=16)


def draw4(ax):
    setup(ax,4,'Keep alternative ports',
          'Change cut position and visit order.\nE = entry; X = exit. Circulation stays fixed.')
    for x,radii,seam,title in ((1,(28,21,14),.9,'Inward'),(55,(14,21,28),-.8,'Outward')):
        v=View(ax,(x,31,43,45),(-35,35,-35,35,0,2));v.grid()
        for r in radii:v.path(circle(r),MUTED,.8,'--',.45)
        route=spiral(radii,seam=seam);v.winding(route)
        v.point(route[0][1][0],'E',offset=(1,5))
        v.point(route[-1][1][-1],'X',marker='s',offset=(1,-5))
        text(ax,x+21,82,title,size=16)


def draw5(ax):
    setup(ax,5,'Set the jump budget',
          'Start with J = 0; count raised spans.\nSeveral turn crossings can be one jump.')
    for x,j in ((8,0),(39,1),(70,2)):
        box(ax,x,71,22,12,f'J ≤ {j}',INK,17)
    arrow(ax,(31,77),(38,77),INK,1.5,10)
    arrow(ax,(62,77),(69,77),INK,1.5,10)
    v=View(ax,(6,27,88,37),(-38,38,-13,13,0,17));v.grid()
    for x in (-18,0,18):v.path([[x,-12,0],[x,12,0]],TEAL,3)
    p=elevated_transfer(height=13);v.path(p,PURPLE,3)
    v.point(p[0]);v.point(p[-1])
    text(ax,50,61,'This span: 1 jump / 3 turns crossed',size=15,color=PURPLE)


def draw6(ax):
    setup(ax,6,'Connect the complete route',
          'Choose cuts, order and group states.\nInclude both leads and every return.')
    v=View(ax,(4,26,92,56),(-64,54,-40,28,0,18));v.grid()
    left=spiral((19,13,7),(-29,0,0),1,0)
    right=spiral((7,13,19),(29,0,0),1,np.pi)
    v.winding(left);v.winding(right)
    # Same fixed terminal positions, full A-to-B path with two group windings.
    A=np.array([-57,-29,0]);B=np.array([-57,-36,0])
    ent=left[0][1][0];end=right[-1][1][-1]
    lead_a=np.array([A,[-5,-29,0],[-5,ent[1],0],ent])
    lead_b=np.array([end,[4,end[1],0],[4,-36,0],B])
    v.path(lead_a,ORANGE,2.5);v.path(lead_b,ORANGE,2.5)
    p=left[-1][1][-1];q=right[0][1][0]
    t=np.linspace(0,1,160);transfer=p[None,:]+t[:,None]*(q-p)[None,:]
    # Smooth end transitions around a genuinely straight raised central span.
    z=elevated_transfer(height=13,n=160)[:,2];transfer[:,2]=z
    v.path(transfer,PURPLE,3)
    v.point(A,'A',offset=(-5,3));v.point(B,'B',offset=(-5,-3))
    v.path([A,[-63,-29,0],[-63,-36,0],B],MUTED,1.2,'--')
    text(ax,24,83,'Group 1: in',size=15)
    text(ax,76,83,'Group 2: out',size=15)
    text(ax,51,23,'Illustrative A → B candidate; not certified',size=14.5)


DRAWERS={1:draw1,2:draw2,3:draw3,4:draw4,5:draw5,6:draw6}

if __name__=='__main__':
    for number,draw in DRAWERS.items():save_single(number,draw)

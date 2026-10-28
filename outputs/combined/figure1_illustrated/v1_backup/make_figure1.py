#!/usr/bin/env python3
"""Manuscript flow: original generated icons with editable text and connectors.
Geometry is in millimetres from the lower-left. Original PNG pixels are unchanged.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.image import imread
from matplotlib.patches import FancyArrowPatch
from matplotlib.path import Path as CurvePath
from matplotlib.transforms import Bbox
import numpy as np

HERE=Path(__file__).resolve().parent
ASSETS=HERE/'assets'
FIGURES=HERE.parent/'figures'
WIDTH,HEIGHT,DPI=183.0,96.0,600
INK,MUTED,ARROW='#2A3830','#59665D','#7B8980'
RC={'font.family':'Arial','font.size':8,'svg.fonttype':'none',
    'svg.hashsalt':'illustrated-clinical-benchmark','pdf.fonttype':42,
    'ps.fonttype':42,'image.composite_image':False,
    'figure.facecolor':'white','savefig.facecolor':'white','savefig.bbox':None}


def put_icon(ax,name,center,max_width=24.0,max_height=26.0):
    """Scale the visible footprint without modifying the source PNG."""
    source=ASSETS/f'{name}.png'
    im=imread(source)
    assert im.ndim==3 and im.shape[2]==4,source
    ys,xs=np.where(im[...,3]>0.12)
    left,right,top,bottom=int(xs.min()),int(xs.max())+1,int(ys.min()),int(ys.max())+1
    h,w=im.shape[:2]
    scale=min(max_width/(right-left),max_height/(bottom-top))
    cx,cy=center
    mid_x,mid_y=(left+right)/2,(top+bottom)/2
    extent=(cx-mid_x*scale,cx+(w-mid_x)*scale,cy-(h-mid_y)*scale,cy+mid_y*scale)
    artist=ax.imshow(im,extent=extent,interpolation='none',resample=False,zorder=3)
    artist.set_gid(name+'-illustration')
    half_w,half_h=(right-left)*scale/2,(bottom-top)*scale/2
    return {'source':str(source.relative_to(HERE)),'center_mm':center,
            'visible_bbox_mm':[cx-half_w,cy-half_h,cx+half_w,cy+half_h],
            'alpha_bounds_px':[left,top,right,bottom],'full_extent_mm':extent,
            'effective_dpi':25.4/scale,
            'sha256':hashlib.sha256(source.read_bytes()).hexdigest()}


def connector(ax,name,points,head=True):
    codes=[CurvePath.MOVETO,CurvePath.CURVE4,CurvePath.CURVE4,CurvePath.CURVE4] if len(points)==4 else [CurvePath.MOVETO]+[CurvePath.LINETO]*(len(points)-1)
    patch=FancyArrowPatch(path=CurvePath(points,codes),arrowstyle='-|>' if head else '-',
                          mutation_scale=6.4,linewidth=0.7,color=ARROW,
                          capstyle='round',joinstyle='round',zorder=1)
    patch.set_gid(name)
    ax.add_patch(patch)


def label(ax,name,x,y,lines):
    artists=[]
    for index,(content,size,color) in enumerate(lines):
        artist=ax.text(x,y-index*4,content,ha='center',va='top',fontsize=size,color=color,zorder=5)
        artist.set_gid(f'{name}-label-{index}')
        artists.append(artist)
    return artists


def build():
    fig=plt.figure(figsize=(WIDTH/25.4,HEIGHT/25.4),dpi=150)
    ax=fig.add_axes([0,0,1,1])
    ax.set(xlim=(0,WIDTH),ylim=(0,HEIGHT),aspect='equal')
    ax.axis('off')
    icons={
        'clinical_tasks':put_icon(ax,'clinical_tasks',(18,51),25,25),
        'xgboost':put_icon(ax,'xgboost',(65,74),25,23),
        'system1':put_icon(ax,'system1',(65,29),24,24),
        'comparison':put_icon(ax,'comparison',(113,51),22.5,26),
        'complexity':put_icon(ax,'complexity',(160,74),23,25),
        'codebook':put_icon(ax,'codebook',(160,29),24,24),
    }
    connector(ax,'input-stem',[(32.2,51),(36,51)],False)
    connector(ax,'input-to-xgboost',[(36,51),(44,51),(42,74),(50.8,74)])
    connector(ax,'input-to-system1',[(36,51),(44,51),(42,29),(51.5,29)])
    connector(ax,'xgboost-to-comparison',[(79,74),(91,74),(91,51),(100,51)],False)
    connector(ax,'system1-to-comparison',[(78.5,29),(91,29),(91,51),(100,51)],False)
    connector(ax,'comparison-arrow',[(100,51),(105.6,51)])
    connector(ax,'association-stem',[(120.5,51),(132,51)],False)
    connector(ax,'to-complexity',[(132,51),(140,51),(138,74),(148,74)])
    connector(ax,'to-codebook',[(132,51),(140,51),(138,29),(146.7,29)])
    labels=[]
    labels+=label(ax,'tasks',18,35,[('45 clinical tasks',9,INK),('Binary + multiclass',8,MUTED)])
    labels+=label(ax,'xgboost',65,59,[('XGBoost',9,INK),('Supervised · tabular predictors',7.8,MUTED)])
    labels+=label(ax,'system1',65,14,[('System 1',9,INK),('Frozen Qwen3-4B',8,INK),('Zero-shot · no training labels',7.8,MUTED)])
    labels+=label(ax,'comparison',113,34,[('Balanced accuracy',9,INK),('Difference vs XGBoost',8,MUTED)])
    labels+=label(ax,'complexity',160,59,[('Task complexity',9,INK)])
    labels+=label(ax,'codebook',160,14,[('Codebook explanation',9,INK)])
    labels+=label(ax,'evaluation',113,74,[('Per-task',8,MUTED),('comparison',8,MUTED)])
    return fig,ax,icons,labels


def validate(fig,ax,icons,labels):
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    inverse=ax.transData.inverted()
    records=[]
    collisions=[]
    for artist in labels:
        b=artist.get_window_extent(renderer)
        lower,upper=inverse.transform([[b.x0,b.y0],[b.x1,b.y1]])
        records.append({'text':artist.get_text(),'font_pt':artist.get_fontsize(),
                        'bbox_mm':[*lower,*upper],
                        'inside_canvas':bool(lower[0]>1 and lower[1]>1 and upper[0]<WIDTH-1 and upper[1]<HEIGHT-1)})
    for i,first in enumerate(records):
        b=Bbox.from_extents(*first['bbox_mm'])
        for second in records[:i]:
            if b.overlaps(Bbox.from_extents(*second['bbox_mm'])):
                collisions.append([first['text'],second['text']])
        for name,icon in icons.items():
            if b.overlaps(Bbox.from_extents(*icon['visible_bbox_mm'])):
                collisions.append([first['text'],'illustration: '+name])
    report={'width_mm':WIDTH,'height_mm':HEIGHT,'minimum_font_pt':min(t.get_fontsize() for t in labels),
            'text_and_image_collisions':collisions,'labels':records,'icons':icons,
            'proof_widths_mm':[183,160],'minimum_font_at_160mm_pt':7.8*160/183}
    (HERE/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    if collisions or not all(r['inside_canvas'] for r in records):
        raise RuntimeError('Layout issue: '+repr(collisions))
    return report


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(RC):
        fig,ax,icons,labels=build()
        validate(fig,ax,icons,labels)
        fig.savefig(FIGURES/'Figure1.pdf',dpi=DPI,metadata={'Title':'Clinical prediction benchmark: study flow'})
        fig.savefig(FIGURES/'Figure1.svg',dpi=DPI)
        # Render the final figure smoothly; original embedded assets are unchanged.
        for artist in ax.images:
            artist.set_interpolation('lanczos')
            artist.set_resample(True)
        fig.savefig(FIGURES/'Figure1.png',dpi=DPI)
        for width in (183,160):
            fig.savefig(HERE/f'proof_{width}mm.png',dpi=220*width/WIDTH)
        plt.close(fig)
    tree=ET.parse(FIGURES/'Figure1.svg')
    ns={'s':'http://www.w3.org/2000/svg'}
    images=len(tree.findall('.//s:image',ns))
    texts=len(tree.findall('.//s:text',ns))
    assert images==6 and texts==len(labels)
    (HERE/'validation/export_checks.json').write_text(json.dumps({
        'svg_independently_editable_images':images,'svg_live_text_elements':texts,
        'canvas_mm':[WIDTH,HEIGHT],'png_dpi':DPI,'asset_pixels_changed':False,
        'minimum_native_asset_dpi':min(i['effective_dpi'] for i in icons.values()),
        'outputs_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in FIGURES.glob('Figure1.*')}
    },indent=2)+'\n')
    print(f'Created 183 × 96 mm figure: {images} generated icons, {texts} live labels; no overlaps.')


if __name__=='__main__':
    main()

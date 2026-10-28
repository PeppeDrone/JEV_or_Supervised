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
from matplotlib.patches import FancyArrowPatch,FancyBboxPatch
from matplotlib.path import Path as CurvePath
from matplotlib.transforms import Bbox
import numpy as np

HERE=Path(__file__).resolve().parent
ASSETS=HERE/'assets'
FIGURES=HERE.parent/'figures'
WIDTH,HEIGHT,DPI=183.0,110.0,600
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


def pill(ax,name,cx,cy,w,h,lines,face='#EEF2EE'):
    ax.add_patch(FancyBboxPatch((cx-w/2,cy-h/2),w,h,boxstyle='round,pad=0,rounding_size=1.6',
                                linewidth=0.7,edgecolor=ARROW,facecolor=face,zorder=2))
    artists=[]
    n=len(lines)
    for i,(content,size,color) in enumerate(lines):
        y=cy+(n-1)*1.9-i*3.8
        t=ax.text(cx,y,content,ha='center',va='center',fontsize=size,color=color,zorder=5)
        t.set_gid(f'{name}-label-{i}')
        artists.append(t)
    return artists


def build():
    fig=plt.figure(figsize=(WIDTH/25.4,HEIGHT/25.4),dpi=150)
    ax=fig.add_axes([0,0,1,1])
    ax.set(xlim=(0,WIDTH),ylim=(0,HEIGHT),aspect='equal')
    ax.axis('off')
    ax.add_patch(FancyBboxPatch((28,16),107,66.5,boxstyle='round,pad=0,rounding_size=3',
                                linewidth=0,facecolor='#F3F6F3',zorder=0))
    icons={
        'clinical_tasks':put_icon(ax,'clinical_tasks',(13,62),21,21),
        'xgboost':put_icon(ax,'xgboost',(58,100),20,17),
        'codebook':put_icon(ax,'codebook',(90,66),20,15),
        'system1':put_icon(ax,'system1',(119,38),19,19),
        'comparison':put_icon(ax,'comparison',(160,70),18,20),
        'complexity':put_icon(ax,'complexity',(142,30),15,16),
        'codebook_detail':put_icon(ax,'codebook',(169,30),17,12),
    }
    # flow: table -> XGBoost and table -> System 1 lanes
    connector(ax,'table-to-xgboost',[(13,74),(13,92),(17,100),(48,100)])
    connector(ax,'table-to-names',[(24,62),(27,62),(27,66),(30,66)])
    connector(ax,'table-to-values',[(24,62),(27,62),(27,38),(30,38)])
    # codebook production (row A) and rendering (row B)
    connector(ax,'names-to-grok',[(50,66),(53.5,66)])
    connector(ax,'grok-to-codebook',[(73,66),(79,66)])
    connector(ax,'codebook-to-renderer',[(90,46.2),(90,43)])
    connector(ax,'values-to-renderer',[(49.5,38),(76.5,38)])
    connector(ax,'renderer-to-model',[(103.5,38),(108.5,38)])
    # outputs to the comparison
    connector(ax,'xgboost-to-comparison',[(74,100),(150,100),(160,96),(160,82.5)])
    connector(ax,'model-to-comparison',[(129,38),(138,38),(140,68),(150,68)])
    connector(ax,'comparison-to-complexity',[(160,49),(160,45),(142,45),(142,40)])
    connector(ax,'comparison-to-codebook',[(160,49),(160,45),(169,45),(169,38)])
    labels=[]
    labels+=label(ax,'tasks',13,47,[('45 clinical',8,INK),('tables',8,INK)])
    labels+=label(ax,'xgboost',101,108,[('Nested XGBoost',9,INK),('Outcome labels · every patient given',7.6,MUTED)])
    ax.text(31,79.2,'System 1: zero-shot, no training labels',ha='left',va='top',fontsize=8,color=MUTED,zorder=5,gid='band-title')
    labels+=[ax.texts[-1]]
    labels+=pill(ax,'names',40,66,19,9,[('Variable names',7.4,INK),('and codes',7.4,INK)])
    labels+=pill(ax,'grok',63,66,19,9,[('Grok 4.7',8,INK),('writes glosses',7.2,MUTED)])
    labels+=label(ax,'codebook',90,56.5,[('Codebook',8.4,INK),('plain English · units · ranges',7.4,MUTED)])
    labels+=pill(ax,'values',40,38,19,9,[('Predictor',7.4,INK),('values only',7.4,INK)])
    labels+=pill(ax,'renderer',90,38,26,9,[('Renderer',8,INK),('per patient, as text',7.2,MUTED)])
    labels+=label(ax,'system1',119,27,[('Frozen Qwen3-4B',8.4,INK),('AnyJev L0',7.8,MUTED)])
    labels+=label(ax,'comparison',160,57,[('Balanced accuracy',8.4,INK),('gap vs XGBoost',7.8,MUTED)])
    labels+=label(ax,'complexity',142,20.5,[('Task',8,INK),('complexity',8,INK)])
    labels+=label(ax,'codebook-detail',169,20.5,[('Codebook',8,INK),('detail',8,INK)])
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
    assert images==7 and texts==len(labels),(images,texts,len(labels))
    (HERE/'validation/export_checks.json').write_text(json.dumps({
        'svg_independently_editable_images':images,'svg_live_text_elements':texts,
        'canvas_mm':[WIDTH,HEIGHT],'png_dpi':DPI,'asset_pixels_changed':False,
        'minimum_native_asset_dpi':min(i['effective_dpi'] for i in icons.values()),
        'outputs_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in FIGURES.glob('Figure1.*')}
    },indent=2)+'\n')
    print(f'Created 183 × 110 mm figure: {images} generated icons, {texts} live labels; no overlaps.')


if __name__=='__main__':
    main()

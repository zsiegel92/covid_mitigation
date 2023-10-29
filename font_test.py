import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

fonts = mpl.font_manager.findSystemFonts(fontpaths=None,fontext='ttf')
fontnames = sorted([thing.rsplit("/",1)[-1].rsplit(".",1)[0] for thing in fonts[:200]])

content = [[fontname] for fontname in fontnames]
fig,ax = plt.subplots(figsize=(10,len(fontnames)/4))

table = ax.table(cellText = content,
				  # rowColours=colors,
				  # rowLabels=row_labels,
				  # colLabels=cols,
				  loc='center',
				  cellLoc='left')
ax.axis('off')
ax.set_aspect('auto')
plt.tight_layout()
plt.savefig("figures/font_test.pdf")
table_d = table.get_celld()





for (tup,cell),font in zip(table_d.items(),fontnames):
	try:
		cell.set_text_props(fontfamily=font)
	except:
		print(f"Error on {font}")
	# cell.set_text_props(fontweight="bold")

# plt.show()

# Help on method set_text_props in module matplotlib.table:

# set_text_props(**kwargs) method of matplotlib.table.Cell instance
#     Update the text properties.

#     Valid keyword arguments are:

#     Properties:
#         agg_filter: a filter function, which takes a (m, n, 3) float array and a dpi value, and returns a (m, n, 3) array
#         alpha: scalar or None
#         animated: bool
#         backgroundcolor: color
#         bbox: dict with properties for `.patches.FancyBboxPatch`
#         clip_box: `.Bbox`
#         clip_on: bool
#         clip_path: Patch or (Path, Transform) or None
#         color or c: color
#         contains: unknown
#         figure: `.Figure`
#         fontfamily or family: {FONTNAME, 'serif', 'sans-serif', 'cursive', 'fantasy', 'monospace'}
#         fontproperties or font or font_properties: `.font_manager.FontProperties` or `str` or `pathlib.Path`
#         fontsize or size: float or {'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'}
#         fontstretch or stretch: {a numeric value in range 0-1000, 'ultra-condensed', 'extra-condensed', 'condensed', 'semi-condensed', 'normal', 'semi-expanded', 'expanded', 'extra-expanded', 'ultra-expanded'}
#         fontstyle or style: {'normal', 'italic', 'oblique'}
#         fontvariant or variant: {'normal', 'small-caps'}
#         fontweight or weight: {a numeric value in range 0-1000, 'ultralight', 'light', 'normal', 'regular', 'book', 'medium', 'roman', 'semibold', 'demibold', 'demi', 'bold', 'heavy', 'extra bold', 'black'}
#         gid: str
#         horizontalalignment or ha: {'center', 'right', 'left'}
#         in_layout: bool
#         label: object
#         linespacing: float (multiple of font size)
#         math_fontfamily: str
#         multialignment or ma: {'left', 'right', 'center'}
#         path_effects: `.AbstractPathEffect`
#         picker: None or bool or float or callable
#         position: (float, float)
#         rasterized: bool
#         rotation: float or {'vertical', 'horizontal'}
#         rotation_mode: {None, 'default', 'anchor'}
#         sketch_params: (scale: float, length: float, randomness: float)
#         snap: bool or None
#         text: object
#         transform: `.Transform`
#         transform_rotates_text: bool
#         url: str
#         usetex: bool or None
#         verticalalignment or va: {'center', 'top', 'bottom', 'baseline', 'center_baseline'}
#         visible: bool
#         wrap: bool
#         x: float
#         y: float
#         zorder: float

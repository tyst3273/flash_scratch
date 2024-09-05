import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = r"\usepackage{bm}"


x = 1000

ny = 1000
y = np.geomspace(0.1, 1000, num=ny) #np.linspace(1e-3,1e3)
 
fig, ax = plt.subplots(figsize=(4,4))

Ea = 1/(1+y/x)
Eb = 1/(1/y+1/x)

ax.plot(y,Ea,c='r',ms=0,ls='-',lw=1,label=r'E$_a$')
ax.plot(y,Eb,c='b',ms=0,ls='--',lw=1,label=r'E$_b$')

#ax.plot(y,Eb/Ea,c='k',ls=(0,(4,2,2,2)),label=r'E$_b$/E$_a$')

#ratio =  (x*y + y**2)/(x+y)  #1/(1/y + 1/x) + (y/x)/(1/y+1/x)  #(x+y)/(x/y+1)
#ax.plot(y,ratio,c='m',ms=0,ls=(0,(4,1,2,1)))

ax.plot([100,100],[0,10000],lw=0.5,ls=':',c='k')

for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(1.1)

ax.minorticks_on()
ax.tick_params(which='both', width=1, labelsize='x-large')
ax.tick_params(which='major', length=5)
ax.tick_params(which='minor', length=3, color='k')
ax.set_rasterization_zorder(10000)

ax.set_ylabel(r'E/V',labelpad=0.0,fontweight='normal',fontsize='large')
ax.set_xlabel(r'$\varepsilon_a/\varepsilon_b$',labelpad=3.0,fontweight='normal',fontsize='large')

ax.set_xscale('log')
ax.set_yscale('log')

ax.annotate(rf'd$_a$/(2d$_b$)={x:d}',xy=(0.05,0.65),xycoords='axes fraction',fontsize='large')

ax.legend(frameon=False)

#ax.autoscale(tight=True)
#ax.axis('tight')

ax.axis([0.1,1000,0.1,1000])

plt.savefig('e_fields.pdf',dpi=300,bbox_inches='tight')

plt.show()



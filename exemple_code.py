from phasor2 import phasor as ph
x=ph('C:/blala/blabla/yourimage',lamdbasmin=410)
x.calcul_phasor()
img=x.hsv()
x.plot_phasor()
x.display_img(img=x)

import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns 
import skimage.io as sk
import matplotlib as matp
from .utils import display_img


def preprocess_image(img, channel_first=False):
            #change in np.float32 and rescale the image
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    elif img.dtype == np.uint16:
        img = img.astype(np.float32) / 65535.0
    if channel_first :
        img = np.transpose(img, (2, 1, 0))
    return img

def calcul_lamdbasbis(min,nb_val=32):
    l_longeur_onde=np.zeros(nb_val)
    l_longeur_onde[0]=min
    i=1
    while i<nb_val:
        l_longeur_onde[i]=round(l_longeur_onde[i-1]+8.9)

        i+=1
    return(l_longeur_onde)


class Phasor:
    def __init__ (self,name_img,lamdbasmin = 410):
 
        #formatage de l'image 
        self.img=preprocess_image(sk.imread(name_img), channel_first=True)
        #calcule des lamdba pour l'utilisation du phasor 
        self.lamdbas=calcul_lamdbasbis(lamdbasmin)
        self._calcul_phasor()


    def _calcul_phasor(self):
        deltalambda = self.lamdbas[-1] - self.lamdbas[0]
        #application de la premiere partie du phasor sur l"image sur l'axe 2 qui correspond a l'axe des channel
        numerateursin=np.sum(self.img*np.sin((2*np.pi*(self.lamdbas-self.lamdbas[0]))/deltalambda),axis = 2)
        numerateursin = np.nan_to_num(numerateursin, nan=0, posinf=1)
        numerateursin[np.where(numerateursin==0)]=0.00001
        #--
        numerateurcos=np.sum(self.img*np.cos((2*np.pi*(self.lamdbas-self.lamdbas[0]))/deltalambda), axis = 2)
        numerateurcos = np.nan_to_num(numerateurcos, nan=0, posinf=1)
        numerateurcos[np.where(numerateurcos==0)]=0.00001
        #--
        denominateur=np.sum(self.img, axis = 2)
        denominateur=np.nan_to_num(denominateur, nan=0.0001, posinf=1)
        denominateur[np.where(denominateur==0)]=0.00001
        #--
        g =numerateurcos/denominateur
        s =numerateursin/denominateur
        #création de la liste et insertion des valeur a l'interieur 
        phasors = np.zeros((self.img.shape[0], self.img.shape[1],2))
        phasors[:,:,0] = np.sqrt(g**2+s**2)
        phasors[:,:,1] = np.arctan2(g,s)
        self.phasors=phasors
    
    #affichage du phasor sous la forme d'une heatmaps
    def plot_phasor(self, nb = 2000, coeff = 1,save_name = None, cmap = 'turbo'):
        img_phasor=self.phasors
        np.random.seed(19680801)
        r = img_phasor[:,:,0].ravel()
        theta =  coeff*img_phasor[:,:,1].ravel()
        if nb!=-1 and nb<len(theta):
            indices = np.random.choice(len(theta), nb, replace=False)
            r = r[indices]
            theta = theta[indices]
        x = r * np.cos(theta)
        y = r * np.sin(theta)

        fig = plt.figure(figsize=(10, 10))
        grid_ratio = 5
        gs = plt.GridSpec(grid_ratio + 1, grid_ratio + 1)

        ax_joint = fig.add_subplot(gs[1:, :-1])
        ax_marg_x = fig.add_subplot(gs[0, :-1])
        ax_marg_y = fig.add_subplot(gs[1:, -1])

        sns.kdeplot(x=x, y=y, bw_adjust=0.7, linewidths=1, fill=True, ax=ax_joint,levels = 100, cmap=cmap)

        ax_joint.set_aspect('equal', adjustable='box')  # equal aspect ratio is needed for a polar plot
        ax_joint.axis('off')
        xmin, xmax = ax_joint.get_xlim()
        xrange = 1
        ax_joint.set_xlim(-xrange, xrange)  # force 0 at center
        ymin, ymax = ax_joint.get_ylim()
        yrange = 1
        ax_joint.set_ylim(-yrange, yrange)  # force 0 at center

        ax_polar = fig.add_subplot(projection='polar')
        ax_polar.set_facecolor('none')  # make transparent
        ax_polar.set_position(pos=ax_joint.get_position())
        ax_polar.set_rlim(0, max(xrange, yrange))

        # add kdeplot of power as marginal y
        sns.kdeplot(y=r, ax=ax_marg_y)
        ax_marg_y.set_ylim(0, r.max() * 1.1)
        ax_marg_y.set_xlabel('')
        ax_marg_y.set_ylabel('')
        ax_marg_y.text(1, 0.5, 'power', transform=ax_marg_y.transAxes, ha='center', va='center')
        sns.despine(ax=ax_marg_y, bottom=True)

        # add kdeplot of angles as marginal x, repeat the angles shifted -360 and 360 degrees to enable wrap-around
        angles = np.degrees(theta)
        angles_trippled = np.concatenate([angles - 360, angles, angles + 360])
        sns.kdeplot(x=angles_trippled, ax=ax_marg_x)
        ax_marg_x.set_xlim(0, 360)
        ax_marg_x.set_xticks(np.arange(0, 361, 45))
        ax_marg_x.set_xlabel('')
        ax_marg_x.set_ylabel('')
        ax_marg_x.text(0.5, 1, 'angle', transform=ax_marg_x.transAxes, ha='center', va='center')
        sns.despine(ax=ax_marg_x, left=True)
        if save_name is not None:
            plt.savefig(save_name)
        else :
            plt.show()

    def display_histogram(self, channel = 0):
        img=self.phasors
        #fig large size
        plt.figure(figsize=(10, 5))
        chann=int(channel)
        if channel is not None:
            
            hist = cv2.calcHist([(img*255).astype(np.uint8)],[chann],None,[256],[0,256]) 
            plt.plot(hist, color='r') 
            plt.savefig("hist.png")
        else:
            for i in range (img.shape[2]):
                hist = cv2.calcHist([(img*255).astype(np.uint8)],[i],None,[256],[0,256]) 
            plt.plot(hist)
            plt.savefig("hist.png")
    #transforme le phasor en image hsv 

    def hsv(self,rgb=True,alpha_s = 0.3, alpha_v = 0.05):
        img_intensity = np.mean(self.img, axis=2)
        img_intensity = img_intensity / img_intensity.max()
        hsv_img = np.zeros((self.img.shape[0], self.img.shape[1], 3))
        hsv_img[:,:,0] = (np.pi/2+self.phasors[:,:,1])/np.pi 
        hsv_img[:,:,1] = alpha_s+ (1-alpha_s)* self.phasors[:,:,0]
        hsv_img[:,:,2] = alpha_v + (1- alpha_v)*img_intensity
        if rgb:
            rgb_img = cv2.cvtColor((hsv_img*255).astype(np.uint8), cv2.COLOR_HSV2RGB)
            return(rgb_img)
        return(hsv_img)
    #affiche l'image

    def display_img(self, hsv = False, alpha_s = 0.3, alpha_v = 0.2):
        if hsv:
            img = self.hsv(rgb=True, alpha_s = alpha_s, alpha_v = alpha_v)
        else:
            img = self.img.max(axis = 2)
        display_img(img)
 
    #calcule la phasor et plot l'image
    def globale(self):
        self.calcul_phasor()
        self.plot_phasor()
        plt.show()

import numpy as np
from PIL import Image
import os
import cv2



def rescale(img,short_side_size):
    (w,h) = img.size
    short_side = min(w,h)
    scale = short_side_size/short_side
    dim = (int(w*scale),int(h*scale))
    return img.resize(dim)


def genmask(foreground,background,cor,name):
    mask = np.zeros((background.size[0],background.size[1],1))
    fg = np.array(foreground)
    alpha = fg[:,:,3]
    alpha = np.expand_dims(alpha,axis=2)

    mask[cor[0]:cor[0]+foreground.size[0],cor[1]:cor[1]+foreground.size[1],:] = alpha
    cv2.imwrite('./pasted_img/mask/pm_'+name+'.png',mask)


def main():
    f_short_side_size = 600
    b_short_side_size = 800

    fg_list = os.listdir('./fgimgs')
    bg_list = os.listdir('./bgimgs')

    for fg in fg_list:
        try:
            randx = np.random.randint(-50,50)
            randy = np.random.randint(-50,50)
            foreground = Image.open('./fgimgs/'+fg)
            fg_name = fg.split('.')[0]

            # for bg in bg_list:
            #     bg_name = bg.split('.')[0]
            #     background = Image.open('./bgimgs/'+bg)
            #
            #     background = rescale(background,b_short_side_size)
            #     foreground = rescale(foreground,f_short_side_size)
            #
            #     b_center = (background.size[0]//2,background.size[1]//2)
            #     f_center = (foreground.size[0]//2,foreground.size[1]//2)
            #     f_left = (b_center[0]-f_center[0]+randx,b_center[1]-f_center[1]+randy)
            #
            #     background.paste(foreground,f_left,foreground)
            #     background.save('./pasted_img/img/p_'+fg_name+'_'+bg_name+'.jpg')
            #
            #     genmask(foreground,background,f_left,fg_name+'_'+bg_name)
            #
            bg = bg_list[np.random.randint(0,len(bg_list))]

            bg_name = bg.split('.')[0]
            background = Image.open('./bgimgs/' + bg)

            background = rescale(background, b_short_side_size)
            foreground = rescale(foreground, f_short_side_size)

            b_center = (background.size[0] // 2, background.size[1] // 2)
            f_center = (foreground.size[0] // 2, foreground.size[1] // 2)
            f_left = (b_center[0] - f_center[0] + randx, b_center[1] - f_center[1] + randy)

            background.paste(foreground, f_left, foreground)
            background.save('./pasted_img/img/p_' + fg_name + '_' + bg_name + '.jpg')

            print(foreground.size)
            print(background.size)
            genmask(foreground, background, f_left, fg_name + '_' + bg_name)
        except:
            pass


if __name__ =='__main__':
    main()
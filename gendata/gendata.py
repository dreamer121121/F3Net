import numpy as np
from PIL import Image
import os
import cv2


def rescale(img, short_side_size):
    (w, h) = img.size
    short_side = min(w,h)
    scale = short_side_size/short_side
    dim = (int(w*scale),int(h*scale))
    return img.resize(dim)


def genmask(foreground, background, cor, name):
    mask = Image.new('L', background.size)
    fg = np.array(foreground)
    alpha = Image.fromarray(fg[:,:,3])
    # print(foreground.size)
    # print(fg.shape)
    # cv2.imwrite('tmp.png', fg)
    # fg = foreground.convert('L')
    mask.paste(alpha, cor, alpha)
    mask.save('./pasted_img/mask/pm_'+name+'.png')
    # cv2.imwrite('./pasted_img/mask/pm_'+name+'.png', mask)
    # #

def main():
    f_short_side_size = 600
    b_short_side_size = 900

    fg_list = os.listdir('./fgimgs')
    bg_list = os.listdir('./bgimgs')

    for fg in fg_list:
        if fg.endswith('.png'):
            randx = np.random.randint(-50,50)
            randy = np.random.randint(-50,50)
            foreground = Image.open('./fgimgs/'+fg)
            fg_name = fg.split('.')[0]

            bg = bg_list[np.random.randint(len(bg_list))]

            if bg.endswith('.jpg'):
                #angle = [0, 90, 180, 270][np.random.randint(0, 4)]
                #foreground = foreground.rotate(angle)
                bg_name = bg.split('.')[0]
                background = Image.open('./bgimgs/'+bg)

                background = rescale(background,b_short_side_size)
                foreground = rescale(foreground,f_short_side_size)
                print('----background---', background.size)
                print('----foreground---', foreground.size)

                b_center = (background.size[0]//2,background.size[1]//2)
                f_center = (foreground.size[0]//2,foreground.size[1]//2)
                f_left = (b_center[0]-f_center[0]+randx,b_center[1]-f_center[1]+randy)

                background.paste(foreground,f_left,foreground)
                background.save('./pasted_img/img/p_'+fg_name+'_'+bg_name+'.jpg')

                genmask(foreground,background,f_left,fg_name+'_'+bg_name)


if __name__ =='__main__':
    main()

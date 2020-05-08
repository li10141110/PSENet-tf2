# -*- coding: utf-8 -*-
# !/usr/bin/python

import os
import sys
import random
import cv2
from PIL import Image
import cv2 as cv,  numpy as np
from PIL import ImageFilter

type_list = [16, 19]
number_list = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]
ftp = open("./bank_txt")  #不同银行卡号的开头
b_list = []
for line in ftp:
    b_list.append(line.strip())

vfp_map = open("./6855map.txt", 'r') #字符字典
map_dic = dict()

for iline in vfp_map:
    iline = iline.split(" ")

    word_zn = unicode(iline[1].strip(), 'utf-8')
    map_dic[word_zn] = iline[0].strip()


def getRandomId(num=20, length=10):
    chars = [u'A', u'B', u'C', u'D', u'E', u'F', u'G', u'H', u'I', u'I', u'I', u'I', u'I', u'J', u'J', u'J', u'J', u'J',
             u'K', u'L', u'M', u'N',
             u'O', u'P', u'Q', u'R', u'S', u'T', u'U', u'V', u'W', u'W', u'W', u'W', u'W', u'X', u'Y', u'Z', u'1', u'1',
             u'1', u'2', u'3',
             u'4', u'5', u'6', u'7', u'8', u'9', u'0', u'a', u'b', u'c', u'd', u'e', u'f', u'g', u'h', u'i', u'j', u'k',
             u'l', u'l', u'l', u'l', u'm', u'n',
             u'o', u'p', u'q', u'r', u's', u't', u'u', u'v', u'w', u'x', u'y', u'z']
    ret = []
    for i in range(num):
        id = u''
        for j in range(length):
            id += random.choice(chars)
        ret.append(id)  #list of 20 random ID
    return ret


def get_id():
    nid = getRandomId(1, 17)
    return nid[0]


def random_sequence(len=5):
    s = ""
    for i in range(len):
        s += (str(random.randint(0, 9)))
    # print(str(s))
    return str(s)


# random gen the bank number
def gen_bank_number():  # generate a number randomly
    seed = random.randint(0, 1)
    s_or_l = [16, 19]
    with open('./bank_txt', 'r') as f:
        f = f.readlines()
    index = random.randint(0, len(f) - 1)  #randomly choose a head number which is called index in this line

    front = str(f[index]).split('\n')[0].strip().split('\xc2\xa0')[0]

    print(front, len(front), s_or_l[seed] - len(front))
    back = random_sequence(s_or_l[seed] - len(front))
    sequence = front + back

    seed = 0  # 16 or 19
    if seed == 0:
        new_s = sequence[0:4] + '#' + sequence[4:8] + '#' + sequence[8:12] + '#' + sequence[12:16]
    else:
        new_s = sequence[0:4] + '#' + sequence[4:8] + '#' + sequence[8:12] + '#' + sequence[12:16] + '#' + sequence[
                                                                                                           16:18] + ''

    print("sequence", new_s)
    return new_s



def gen_bank_date_number():  # generate a number randomly
    seed = random.randint(0, 3)
    #s_or_l = [16, 19]
    #with open('./bank_txt', 'r') as f:
        #f = f.readlines()
    #index = random.randint(0, len(f) - 1)  #randomly choose a head number which is called index in this line

    #front = str(f[index]).split('\n')[0].strip().split('\xc2\xa0')[0]

    #print(front, len(front), s_or_l[seed] - len(front))
    #back = random_sequence(s_or_l[seed] - len(front))
    #sequence = front + back

    #seed = 0  # 16 or 19
    if seed == 0:
        new_s = '0' + str(random.randint(1, 9)) + '/' + str(random.randint(1, 4)) + str(random.randint(0, 9))  #gen year of date before 2049
    elif seed == 1:
        new_s = '1' + str(random.randint(0, 2)) + '/' + str(random.randint(1, 4)) + str(random.randint(0, 9))
    elif seed == 2:
        new_s = '20' + str(random_sequence(2)) + '/' + str(random_sequence(2))
    else:
        new_s = str(random_sequence(2)) + '/' + '20' + str(random_sequence(2))

    print("sequence", new_s)
    return new_s

def get_bank_number(num_file='./16_bank_num.txt', index=0):  # get a single number from generated_txt 16_bank_num.txt
    with open(num_file, 'r') as f:
        f = f.readlines()
    bank_num = f[index].split('\n')[0]
    return bank_num


def get_bank_date_number(num_file='./0000_bank_date_num.txt', index=0):
    with open(num_file, 'r') as f:
        f = f.readlines()
    bank_num = f[index].split('\n')[0]
    return bank_num


# choose the bg background pic by random

def get_random_bg():
    seed = random.randint(1, 15)
    if seed == 2:
        seed = 3
    bg_name = 'bg' + str(seed) + '.jpg'
    imgroot = './bank_pic/'
    # img=Image.open(imgroot+bg_name)
    # img.show()
    bg_path = imgroot + bg_name
    return bg_path
    # return "./b2.jpg"


def get_random_bg_bg():
    return "./b3.jpg"


def get_random_crop_bg():
    return "./bank_pic/bg3.jpg"


def auto_gen_num_list(num_file='./16_bank_num.txt', total=2000):  # auto generate a list of 16/19 number
    f_list = []
    with open(num_file, 'w') as f:
        for i in range(total):
            f_list.append(gen_bank_number() + '\n')
        f.writelines(f_list)


def auto_gen_date_num_list(date_num_file='./0000_bank_date.txt', total=20000):  # auto generate a list of date number
    f_list = []
    with open(date_num_file, 'w') as f:
        for i in range(total):
            f_list.append(gen_bank_date_number() + '\n')
        f.writelines(f_list)

def genbankpic_crop(num_file, index, des_folder='./16_bank_date_num_pic/', img_no='date_0000'):
    bank_numbget_random_bger = get_bank_number(num_file, index).strip()
    bg_path = get_random_bg()
    # Load two images
    img_f = []
    number_root = "./bank_number/"  # number image
    len_number = len(bank_numbget_random_bger) #length of numbers of date
    print(len_number)
    img = Image.open(bg_path)  #randomly open a bg image

    box_b = (0, 0, 5, 32)
    img_b = img.crop(box_b)
    # img_crop.show()
    img_b = cv2.cvtColor(np.asarray(img_b), cv2.COLOR_RGB2BGR)
    img_b = np.array(img_b)
    img_f = img_b

    for i in range(len_number):

        nu = 20 * i
        be_num = int(nu)
        box = (be_num, 0, 20 + be_num, 32)
        img_crop = img.crop(box)
        # img_crop.show()
        img_crop = cv2.cvtColor(np.asarray(img_crop), cv2.COLOR_RGB2BGR) # crop BGR patch
        img_blank = np.array(img_crop) # blank
        img1 = np.array(img_crop)

        line = bank_numbget_random_bger[i] # get random bank bg

        if line in ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "#", "/"):
            if line == "0":
                number_root_pic = number_root + "0b.jpg"
            elif line == "1":
                number_root_pic = number_root + "1b.jpg"
            elif line == "2":
                number_root_pic = number_root + "2b.jpg"
            elif line == "3":
                number_root_pic = number_root + "3b.jpg"
            elif line == "4":
                number_root_pic = number_root + "4b.jpg"
            elif line == "5":
                number_root_pic = number_root + "5b.jpg"
            elif line == "6":
                number_root_pic = number_root + "6b.jpg"
            elif line == "7":
                number_root_pic = number_root + "7b.jpg"
            elif line == "8":
                number_root_pic = number_root + "8b.jpg"
            elif line == "9":
                number_root_pic = number_root + "9b.jpg"
            elif line == "/":
                if random.randint(0,1)==1:
                    number_root_pic = number_root + "slashb.jpg"
                else:
                    number_root_pic = number_root + "slash.jpg"
            # read the number pic
            img2 = cv.imread(number_root_pic)
            # I want to put logo on top-left corner, So I create a ROI
            rows, cols, channels = img2.shape

            roi = img1[0:rows, 0:cols]
            # Now create a mask of logo and create its inverse mask also
            img2gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY) # BGR to GRAY
            ret, mask = cv.threshold(img2gray, 250, 255, cv.THRESH_BINARY)  # 这个254很重要 #

            mask_inv = cv.bitwise_not(mask)
            # Now black-out the area of logo in ROI
            img1_bg = cv.bitwise_and(roi, roi, mask=mask)  # 这里是mask,我参考的博文写反了,我改正了,费了不小劲

            # Take only region of logo from logo image.
            img2_fg = cv.bitwise_and(img2, img2, mask=mask_inv)  # 这里才是mask_inv

            # Put logo in ROI and modify the main image
            dst = cv.add(img1_bg, img2_fg)
            img1[0:rows, 0:cols] = dst

            if line == "#":
                img_f = np.concatenate([img_f, img_blank], axis=1)
            else:
                img_f = np.concatenate([img_f, img1], axis=1)

        # bank_pic_name="./bank_pic/"+str(get_id)+".jpg"
    bank_pic_name = des_folder + str(img_no) + '.jpg'
    print(bank_pic_name)
    # if you wanna debug, uncomment it
    # bank_pic_name='./bank_gen/3.jpg'
    img_f = np.concatenate([img_f, img_b], axis=1)

    # 比较锐化的图片
    # cv.imwrite(bank_pic_name, img_f)


    # blend the img array
    img_f_image = Image.fromarray(cv2.cvtColor(img_f, cv2.COLOR_BGR2RGB))
    img_f_image_gaosi = img_f_image.filter(ImageFilter.GaussianBlur(radius=1.5))
    img_f_image_gaosi = img_f_image_gaosi.convert('RGBA')
    width, height = img_f_image_gaosi.size

    img2 = Image.open(bg_path)
    img2 = img2.convert('RGBA')
    img2 = img2.crop([0, 0, width, height])
    img = Image.blend(img_f_image_gaosi, img2, 0.2)

    img_f_image_gaosi=img_f_image_gaosi.convert("RGB")

    # img = img.convert('RGB')
    img_f_image_gaosi.save(bank_pic_name)
    # img.show()


    bank_pic_txt = des_folder + str(img_no) + '.txt'
    ftpw = open(bank_pic_txt, 'w')

    bank_numbget_random_bger = bank_numbget_random_bger.replace("#", "").strip()
    for inum in range(len(bank_numbget_random_bger)):
        if inum < len(bank_numbget_random_bger) - 1:
            ftpw.write(map_dic[bank_numbget_random_bger[inum]] + "\n")
        else:
            ftpw.write(map_dic[bank_numbget_random_bger[inum]])

    ftpw.close()


# random gen the bank number
def gen_bank_number_19():  # generate a number randomly
    ban_num = []
    ftp = open('./bank_txt', 'r')

    for line in ftp:

        line = line.strip()
        if len(line) == 6:
            new_s = ""
            for i in range(13):
                new_s = new_s + (str(random.randint(0, 9)))

            new_s = line + "#" + new_s
            print("sequence", new_s)
            ban_num.append(new_s)
    print (len(ban_num))
    return ban_num


def auto_gen_num_list_19(num_file='./19_bank_num.txt', total=312):  # auto generate a list of 16/19 number
    f_list = []
    with open(num_file, 'w') as f:
        for i in range(total):

            n_list = gen_bank_number_19()
            for line in n_list:
                f.write(line.strip() + "\n")


def auto_gen_num_pic(num_file='./19_bank_num.txt', des_folder='./19_bank_num_pic/'):
    with open(num_file, 'r') as f:
        f = f.readlines()
    total = len(f)
    id = 0
    for i in range(total):
        id += 1
        img_no = '19_3_' + str(id).zfill(5)
        genbankpic_crop(num_file, i, des_folder, img_no)


def auto_gen_date_num_pic(num_file='./0000_bank_date.txt', des_folder='./bank_date_num_pic/'):
    with open(num_file, 'r') as f:
        f = f.readlines()
    total = len(f)
    id = 0
    for i in range(total):
    #for i in range(100):
        id += 1
        img_no = 'date_num_' + str(id).zfill(5)
        genbankpic_crop(num_file, i, des_folder, img_no)


if __name__ == "__main__":

    # auto_gen_num_list(num_file='./16_bank_num.txt',total=50000)#自动生成bank num list，设置的2000

    #genbankpic_crop('./16_bank_num.txt',0) #单步调试把注释掉的文件名解开
    # auto_gen_num_list_19(num_file='./19_bank_num.txt',total=312)#自动生成bank num list，设置的2000
    #
    # auto_gen_num_pic(num_file='./19_bank_num.txt', des_folder='./19_bank_num_pic/')  # 自动生成bank num pic
    #auto_gen_date_num_list()   #generate date data finished 12/25 jason
    auto_gen_date_num_pic(num_file='./0000_bank_date.txt', des_folder='./bank_date_num_pic/') #自动生成bank date num pic
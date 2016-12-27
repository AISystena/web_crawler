import os
import re
import argparse
import urllib.request as url_req
import configparser
from base_scraper import BaseScraper
#from pymongo import MongoClient


class ImageScraper(BaseScraper):
    def __init__(self,url):
        BaseScraper.__init__(self)
        self.url =  url
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument('-f', action='store')

        if parser.parse_args().f:
            cfgFile = parser.parse_args().f
            cfg = configparser.RawConfigParser()
            cfg.optionxform = str
            if cfg.read(cfgFile):
                for key,value in cfg.items('CUSTOM'):
                    self.config[key] = value.strip()

    def scrap_rule(self):
        bf=self.bf
        image_paths=bf.find_all('img')
        links=bf.find_all('a')
        return image_paths,links

    def __dl_images(self, image_paths, url, tmp_dir):

        success_num = 0
        for img in image_paths:
            src = img.get("src")
            img_url = self.getImgURL(src, url)
            fname = src.split("/")[-1]

            if "?" in fname:
                fname = self.rmQ(fname)
            try:
                if fname in os.listdir(tmp_dir):
                    fname = fname + str(success_num)
                url_req.urlretrieve(img_url, tmp_dir+"/"+fname)
                im_path=tmp_dir+"/"+fname
                if os.path.getsize(im_path) < 2000:
                    print("[ Passed ] " + img_url) 
                    os.remove(im_path)
                    continue
                a_class = self.classification(im_path)
                self.sort_out(im_path,fname,a_class)
                print("[ Success ] " + img_url)
                success_num += 1
            except Exception as e:
                print(e)
                print("[ Failed ] " + img_url)

        return success_num

    def run(self):
        self.call()
        image_paths,links = self.scrap_rule()
        img_num = len(image_paths)
        success_num = self.__dl_images(image_paths, self.url, self.config['TmpDir'])
        print(success_num, "images could be downloaded (in", img_num, "images).")
        return links


if __name__ == '__main__':
    # base_url = 'http://www.honda.co.jp/'
    base_url = 'http://www.goo-net.com/usedcar/'
    i_scr=ImageScraper(base_url)
    links=i_scr.run()
    # 周回および深度探索ロジック検討中。。。
    for _a in links:
        link=_a.get('href')
        if not link:
            continue
        instance_scr=ImageScraper(link)
        instance_scr.start()


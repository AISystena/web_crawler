from lib.image_cnn import ImagePredictor
from bs4 import BeautifulSoup
import requests
import shutil
import traceback


class BaseScraper:

    def __init__(self):
        self.config = {
            'TargetDB':'scraper',
            'Host':'localhost',
            'Port':'27017',
        }

    def call(self, url):
        proxies = None
        if proxies:
            html = requests.get(url, proxies=proxies)
        else:
            html = requests.get(url)
        bf = BeautifulSoup(html.text.encode(html.encoding))
        return bf

    def classification(self,image_path):
        ip = ImagePredictor.ImagePredictor()
        answer = ip.predict(image_path)
        return answer

    def scrap_rule(self,bf):
        return None

    def getImgURL(self,src, url):
        if "http" not in src:
            return url+src
        else:
            return src
    
    def mkdir(self, dirName):
        try:
            os.mkdir(dirName)
        except Exception as e:
            print(e)

    def rmQ(self,fname):
        return fname.split("?")[0]

    def sort_out(self,image_path,file_name,a_class):
        print(self.config[str(a_class)])
        try:
            shutil.move(image_path,self.config[str(a_class)]+file_name)
        except:
            print(traceback.format_exc())

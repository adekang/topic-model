import csv
import time
from cProfile import label

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from parsel import Selector
from tqdm import tqdm

"""
精准自动化爬数据
1. 根据标题爬取
一共有44页
"""

class Clarivate:
    def __init__(self, page=1, error_links=None):
        self.enter_url = "https://lib.swust.edu.cn/link/27/3"
        self.list_url = None
        self.page = page
        self.links = [] if error_links is None else error_links
        self.driver = webdriver.Chrome()
        # 自动全屏
        self.driver.maximize_window()
        self.driver.get(self.enter_url)
        ele =  WebDriverWait(self.driver, 20).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, '#onetrust-accept-btn-handler'))
        )
        time.sleep(2)
        ele.click()

    def search(self, keyword="energetic materials"):
        ele =  WebDriverWait(self.driver, 20).until(
            EC.element_to_be_clickable((By.XPATH, '//*[@data-ta="search-field-dropdown"]'))
        )
        time.sleep(2)
        ele.click()
        ele =  WebDriverWait(self.driver, 20).until(
            EC.element_to_be_clickable((By.XPATH, "//*[@title='Title']"))
        )
        time.sleep(2)
        ele.click()

        ele =  WebDriverWait(self.driver, 20).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "#search-option"))
        )
        time.sleep(2)
        ele.send_keys(keyword)

        try:
            ele = WebDriverWait(self.driver, 20).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, '#pendo-close-guide-e0e49982'))
            )
            time.sleep(2)
            ele.click()
            print("guide 弹窗关闭")
        except Exception as e:
            print('没有guide弹窗')


        ele =  WebDriverWait(self.driver, 20).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "button[data-ta='run-search']"))
        )
        time.sleep(2)
        ele.click()

    def get_list(self):
        time.sleep(2)
        url = self.driver.current_url
        url = url.split('/')
        url[-1] = str(self.page)
        url = '/'.join(url)
        print(url)
        self.driver.get(url)
        ele = WebDriverWait(self.driver, 20).until(
            EC.element_to_be_clickable((By.ID, 'onetrust-accept-btn-handler'))
        )
        time.sleep(2)
        ele.click()

        print("cookie 关闭弹窗")


        # 如果存在guide 弹窗，关闭
        try:
            ele = WebDriverWait(self.driver, 20).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, '#pendo-close-guide-30f847dd'))
            )
            time.sleep(2)
            ele.click()
            print("guide 弹窗关闭")
        except Exception as e:
            print('没有guide弹窗', e)

        original_top = 0
        while True:
            # 循环下拉滚动条
            self.driver.execute_script("window.scrollBy(0,500)")
            time.sleep(0.5)
            check_height = self.driver.execute_script(
                "return document.documentElement.scrollTop || window.pageYOffset || document.body.scrollTop;")
            # 如果滚动条距离上面的距离不再改变，也就是滚动后的距离和之前距离顶部的位置没有改变，说明到达最下方，跳出循环
            if check_height == original_top:
                break
            original_top = check_height

        time.sleep(2)

    def get_links(self):
        document = Selector(self.driver.page_source)
        links = document.css("a[data-ta='summary-record-title-link']::attr(href)").getall()
        self.links = links
        print("获取这么多条数: ", links)
        print("获取这么多条数: ", len(links))
        if len(links) < 40:
            raise Exception("链接获取不正常")



    def get_page(self):
        for link_tmp in self.links:
            link = "https://webofscience.clarivate.cn" + link_tmp
            self.driver.get(link)
            time.sleep(5)
            try:
                ele =  WebDriverWait(self.driver, 20).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, '#HiddenSecTa-showMoreDataButton'))
                )
                time.sleep(2)
                ele.click()
                document = Selector(self.driver.page_source)
                title = document.css("h2::text").get()
                by = " ".join(document.xpath('//*[@id="SumAuthTa-MainDiv-author-en"]/span//text()').getall())
                source = " ".join(document.xpath('//*[@id="snMainArticle"]/div[6]/span/div//text()').getall())
                publish = " ".join(document.xpath('//*[@id="FullRTa-pubdate"]//text()').getall())
                early_access = " ".join(document.xpath('//*[@id="FullRTa-earlyAccess"]//text()').getall())
                indexed = " ".join(document.xpath('//*[@id="FullRTa-indexedDate"]//text()').getall())
                document_type = " ".join(document.xpath('*[@id="snMainArticle"]/div[7]/div[5]/span//text()').getall())
                abstract = " ".join(document.xpath('//*[@id="snMainArticle"]/div[8]/div/div/span//text()').getall())
                language = " ".join(document.xpath('//span[@id="HiddenSecTa-language-0"]//text()').getall())
                issn = " ".join(document.xpath('//span[@id="HiddenSecTa-ISSN"]//text()').getall())
                citations = " ".join(document.xpath('//*[@id="snCitationData"]/section[1]/div/div[1]/div[1]//text()').getall())
                citedReferences = " ".join(document.xpath('//*[@id="FullRRPTa-wos-citation-network-refCountLink"]//text()').getall())
                keywords = " ".join(document.xpath('//*[@id="snMainArticle"]/app-full-record-keywords/div/span/div[1]//text()').getall())
                accessionNo = " ".join(document.xpath('//*[@id="HiddenSecTa-accessionNo"]//text()').getall())
                data = {
                    "title": title,
                    "by": by,
                    "source": source,
                    "publish": publish,
                    "early_access": early_access,
                    "indexed": indexed,
                    "document_type": document_type,
                    "abstract": abstract,
                    "keywords": keywords,
                    "language": language,
                    "citations": citations,
                    "cited_references": citedReferences,
                    "issn": issn,
                    "accession_no": accessionNo,

                }
                with open('./wos_by_title.csv', 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(data.values())
            except Exception as e:
                print('获取页面失败', e)
                with open('./error_page_by_title_wos.txt', 'a') as f:
                    f.write(f"{link_tmp}\n")
                continue


    def close(self):
        self.driver.quit()







def start():
    """
    开始爬取
    """
    bar = tqdm(total=44)
    count =13
    for i in range(count,44):
        print(f"正在爬取第{i}页")
        clarivate = Clarivate(i)
        # 清除 localStorage
        clarivate.driver.execute_script('window.localStorage.clear();')
        clarivate.search()

        try:
            clarivate.get_list()
        except Exception as e:
            print('获取列表失败 存入 error_page_by_title', e)
            clarivate.close()
            # 记录失败的页数
            with open('./error_page_by_title.txt', 'a') as f:
                f.write(f"{i}\n")
            continue

        try:
            print("正在获取链接")
            clarivate.get_links()
        except Exception as e:
            print('获取链接失败 存入 error_page_by_title')
            clarivate.close()
            # 记录失败的页数
            with open('./error_page_by_title.txt', 'a') as f:
                f.write(f"{i}\n")
            continue

        print("正在获取页面")
        clarivate.get_page()
        clarivate.close()
        bar.update(1)



def get_error_page():
    """
    重新获取错误的页面
    """
    with open('./error_page_by_title.txt', 'r') as f:
        lines = f.readlines()
        clarivate = Clarivate(1)
        for line in lines:
            clarivate.page = int(line)
            clarivate.get_list()
            clarivate.get_links()
            clarivate.get_page()
            clarivate.close()


def get_error_links():
    """
    重新获取错误的链接
    """
    with open('./error_page_by_title_wos.txt', 'r') as f:
        lines = f.readlines()
        clarivate = Clarivate(1,error_links=lines)
        clarivate.get_page()
        clarivate.close()


if __name__ == '__main__':
    get_error_links()
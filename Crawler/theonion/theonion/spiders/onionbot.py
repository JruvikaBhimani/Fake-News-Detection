# -*- coding: utf-8 -*-
import scrapy
from theonion.items import TheonionItem
#from scrapy.selector import HtmlXPathSelector
from lxml import html

class OnionbotSpider(scrapy.Spider):
    name = 'onionbot'
    allowed_domains = ['http://www.theonion.com/']
    start_urls = ['http://www.theonion.com/']

    def parse(self, response):
        #  print "response:", response.body
        #This will create a list of buyers:
        
        headlines = response.xpath('//h2[@class="headline"]/a/text()')
        #This will create a list of prices
        body = response.xpath('//div[@class="desc"]/text()')
        #       print "before for titles:", headline
        #print "desc:", desc
        for headlines, body in zip(headlines, body):
            headline = headlines.extract().strip()
            body = body.extract().strip()
            item = TheonionItem()
            item['headline'] = headline
            item['body'] = body
            item['check'] = 'fake'
            yield item

#            title = titles.xpath('/html/body/table/tbody/tr[1401]/td[2]/span/span[2]/a/text()').extract()
##            link = titles.select("a/@href").extract()

##            print "link:", link
#            print "titles:",titles

#        print "Response = ",response
#        headline = response.css('.summary.inner.headline::text').extract()
#        body = response.css('.summary.inner.desc::text').extract()
#        #        times = response.css('time::attr(title)').extract()
#        #comments = response.css('.comments::text').extract()
#        #headline = response.css('.headline::text').extract()[0].strip()
#        #body = response.css('.desc::text').extract()[0]
#        print "headline:",headline
#        print "body:", body
#        item = TheonionItem()
#        item['headline'] = headline
#        item['body'] = body
#        yield item
        pass

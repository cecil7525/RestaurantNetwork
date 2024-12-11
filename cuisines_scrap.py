#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 14:49:19 2024

@author: victorsobrino
"""

import requests
from bs4 import BeautifulSoup

def cuisine_scrap():
    # URL of the Wikipedia page containing the list of cuisines
    url = 'https://en.wikipedia.org/wiki/List_of_cuisines'
     
    # Send a GET request to fetch the page content
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
     
    # Initialize an empty set to store cuisine names
    cuisines = set()
     
    # Extract cuisine names from the page
    for section in soup.find_all('div', class_='div-col'):
        for link in section.find_all('a'):
            cuisine_name = link.get_text().strip()
            if cuisine_name:
                cuisines.add(cuisine_name.lower())
     
    return cuisines
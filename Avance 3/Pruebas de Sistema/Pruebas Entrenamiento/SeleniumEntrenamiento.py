# -*- coding: utf-8 -*-
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import Select
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import NoAlertPresentException
import unittest, time, re

class SeleniumEntrenamiento(unittest.TestCase):
    def setUp(self):
        self.driver = webdriver.Firefox()
        self.driver.implicitly_wait(30)
        self.base_url = "http://127.0.0.1:8000/"
        self.verificationErrors = []
        self.accept_next_alert = True
    
    def test_selenium_entrenamiento(self):
        driver = self.driver
        driver.get(self.base_url + "/EigenFaces/trainer/")
        driver.find_element_by_id("procesarCentroide").click()
        driver.find_element_by_id("procesarCentroide").click()
        # ERROR: Caught exception [ERROR: Unsupported command [waitForFrameToLoad |  | ]]
        driver.find_element_by_name("file").click()
        # Warning: waitForTextPresent may require manual changes
        for i in range(60):
            try:
                if re.search(r"^[\s\S]*1-1\.pgm[\s\S]*$", driver.find_element_by_css_selector("BODY").text): break
            except: pass
            time.sleep(1)
        else: self.fail("time out")
        driver.find_element_by_id("procesarCentroide").click()
        # ERROR: Caught exception [ERROR: Unsupported command [waitForFrameToLoad |  | ]]
        driver.find_element_by_id("procesarCentroide").click()
        # ERROR: Caught exception [ERROR: Unsupported command [waitForFrameToLoad | Seleccione un archivo. | ]]
    
    def is_element_present(self, how, what):
        try: self.driver.find_element(by=how, value=what)
        except NoSuchElementException as e: return False
        return True
    
    def is_alert_present(self):
        try: self.driver.switch_to_alert()
        except NoAlertPresentException as e: return False
        return True
    
    def close_alert_and_get_its_text(self):
        try:
            alert = self.driver.switch_to_alert()
            alert_text = alert.text
            if self.accept_next_alert:
                alert.accept()
            else:
                alert.dismiss()
            return alert_text
        finally: self.accept_next_alert = True
    
    def tearDown(self):
        self.driver.quit()
        self.assertEqual([], self.verificationErrors)

if __name__ == "__main__":
    unittest.main()

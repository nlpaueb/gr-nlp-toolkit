import unittest
import os
import shutil

from gr_nlp_toolkit.data.downloader_stub import DownloaderStub
from gr_nlp_toolkit.data.processor_cache import ProcessorCache


class TestProcessorCache(unittest.TestCase):
    def test_download_processors_sequentially(self):

        sep = os.sep
        cache_path = "./test"
        stub = DownloaderStub()
        processor_cache = ProcessorCache(stub , cache_path)
        processor_cache.get_processor_path('ner')
        self.assertTrue(os.path.exists(cache_path + sep + "ner_processor"))
        processor_cache.get_processor_path('pos')
        self.assertTrue(os.path.exists(cache_path + sep + "pos_processor"))
        dp_path = processor_cache.get_processor_path('dp')
        self.assertTrue(type(dp_path) == str)
        self.assertTrue(os.path.exists(cache_path + sep + "dp_processor"))
        self.assertTrue(dp_path == (cache_path + sep + "dp_processor"))
        # Remove any files created
        shutil.rmtree(cache_path)

    def test_download_processor_removing_file_and_folder(self):

        home = os.path.expanduser("~")
        sep = os.sep
        cache_path = "./test"
        stub = DownloaderStub()
        processor_cache = ProcessorCache(stub, cache_path)
        processor_cache.get_processor_path('ner')
        self.assertTrue(os.path.exists(cache_path + sep + "ner_processor"))
        os.remove(cache_path + sep + "ner_processor")
        # Assert that the file is removed
        self.assertTrue(not os.path.exists(cache_path + sep + "ner_processor"))
        processor_cache.get_processor_path('ner')
        # Assert that the file has appeared again
        self.assertTrue(os.path.exists(cache_path + sep + "ner_processor"))
        processor_cache.get_processor_path('pos')
        # Remove entire directory
        shutil.rmtree(cache_path)
        processor_cache.get_processor_path('pos')
        # Assert that the certain processor has appeared again
        self.assertTrue(os.path.exists(cache_path + sep + "pos_processor"))
        # Remove any files created
        shutil.rmtree(cache_path)

if __name__ == '__main__':
    unittest.main()

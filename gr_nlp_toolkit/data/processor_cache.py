import os
from pathlib import Path
from os.path import expanduser

from gr_nlp_toolkit.data.downloader import Downloader


class ProcessorCache:
    def __init__(self, downloader : Downloader, cache_path : str):
        """
        Initializes the cache of processors creating necessary directories
        :param downloader: an object with the Downloader interface
        """
        # Get home directory
        self.home = expanduser("~")
        self.sep = os.sep
        self.cache_path = cache_path
        self.downloader = downloader
        # Initialize the filenames for each processor
        self.processor_names_to_filenames = {
            'ner': 'ner_processor',
            'pos': 'pos_processor',
            'dp': 'dp_processor'
        }
        self.update_cache_path()

    def update_cache_path(self):
        Path(self.cache_path).mkdir(parents=True, exist_ok=True)

    def get_processor_path(self, processor_name: str) -> str:
        # Update cache path in case any changes occured
        self.update_cache_path()
        target_filename = self.processor_names_to_filenames[processor_name]
        if not os.path.exists(self.cache_path + self.sep + target_filename):
            self.downloader.download_processor(processor_name, self.cache_path + self.sep + target_filename)
        # Return the path
        return self.cache_path + self.sep + target_filename

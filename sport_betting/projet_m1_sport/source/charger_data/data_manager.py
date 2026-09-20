import threading
import queue
import time
import random
import os
from .match_service import MatchService

class DataManager:
    def __init__(self, update_callback=None):
        """
        Gère les données des matchs
        Param update_callback: Fonction à appeler pour les mises à jour UI.
        Signature: update_callback(event_type, data)
        """
        self.update_callback = update_callback
        
        # Services
        self.match_service = MatchService()
    

    def get_matches(self, weeks_back=2, weeks_forward=2):
        return self.match_service.get_matches_window(weeks_back, weeks_forward)
    

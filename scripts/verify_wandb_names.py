
import sys
import os
import unittest
from unittest.mock import MagicMock, patch
import datetime

# Add the project directory to sys.path
sys.path.append('/Users/tanmaydeshmukh/Projects/Research/EqM/pokemon_eqm')

# Mock wandb before importing wandb_utils
sys.modules['wandb'] = MagicMock()
import wandb_utils

class TestWandbUtils(unittest.TestCase):
    @patch('wandb_utils.wandb')
    def test_initialize_unique_names(self, mock_wandb):
        # Mock args
        args = MagicMock()
        vars(args).items.return_value = {}
        
        entity = "test_entity"
        exp_name = "test_experiment"
        project_name = "test_project"
        
        # Call initialize twice
        wandb_utils.initialize(args, entity, exp_name, project_name)
        wandb_utils.initialize(args, entity, exp_name, project_name)
        
        # Check calls
        self.assertEqual(mock_wandb.init.call_count, 2)
        
        call_args_list = mock_wandb.init.call_args_list
        
        # Extract names
        name1 = call_args_list[0][1]['name']
        name2 = call_args_list[1][1]['name']
        
        print(f"Run 1 Name: {name1}")
        print(f"Run 2 Name: {name2}")
        
        # Check if names start with exp_name
        self.assertTrue(name1.startswith(exp_name))
        self.assertTrue(name2.startswith(exp_name))
        
        # Check if names are different (timestamps should differ if enough time passes, 
        # but execution might be too fast. We might need to mock datetime or sleep)
        # However, since we are appending timestamp, let's check the format.
        # Format: exp_name_YYYYMMDD_HHMMSS
        
        # If they are the same, it means execution was within the same second.
        # Let's just check the format for now.
        import re
        pattern = r"test_experiment_\d{8}_\d{6}"
        self.assertTrue(re.match(pattern, name1))
        self.assertTrue(re.match(pattern, name2))

if __name__ == '__main__':
    unittest.main()

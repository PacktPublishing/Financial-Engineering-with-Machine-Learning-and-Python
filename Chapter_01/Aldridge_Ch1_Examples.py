# Python Aldridge Ch 1
# Financial Engineering with Python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from collections import Counter
import re

# ============================================================================
# Example 1: CEO Speech Word Frequency Analysis
# ============================================================================

def analyze_speech(speech_text, top_n=10):
    """
    Analyze a speech and return word frequency counts.
    
    Parameters:
    -----------
    speech_text : str
        The text of the speech to analyze
    top_n : int
        Number of top words to return
    
    Returns:
    --------
    dict : Top n words and their counts
    """
    # Convert to lowercase and split into words
    words = re.findall(r'\b[a-z]+\b', speech_text.lower())
    
    # Remove common stop words
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 
                  'to', 'for', 'of', 'with', 'by', 'from', 'as', 'is', 'was',
                  'are', 'were', 'been', 'be', 'have', 'has', 'had', 'do', 
                  'does', 'did', 'will', 'would', 'could', 'should', 'may',
                  'might', 'must', 'can', 'this', 'that', 'these', 'those',
                  'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what', 'which',
                  'who', 'when', 'where', 'why', 'how'}
    
    filtered_words = [word for word in words if word not in stop_words]
    
    # Count word frequencies
    word_counts = Counter(filtered_words)
    
    return dict(word_counts.most_common(top_n))



# ============================================================================
# Example 2: Basic File Reading (as shown at end of chapter)
# ============================================================================

def read_transcript_file(filename):
    """
    Read a transcript file line by line and process it.
    This demonstrates the example from the book.
    
    Parameters:
    -----------
    filename : str
        Path to the transcript file
    
    Returns:
    --------
    list : Processed lines from the file
    """
    try:
        text_file = open(filename, 'r')
        
        # Read the whole file to a string
        lines = text_file.readlines()
        data = []
        
        # Strip the newline character
        for line in lines:
            data.append(line.strip())
        
        # Close file
        text_file.close()
        
        return data
    
    except FileNotFoundError:
        print(f"File {filename} not found.")
        return []


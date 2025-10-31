"""Helper utility functions"""

import random
import string
import pandas as pd
from io import BytesIO


def generate_random_collection_name():
    """Generate a random collection name"""
    random_string = ''.join(random.choices(string.ascii_lowercase, k=6))
    random_number = random.randint(1000, 9999)
    return f"rag_collection_{random_string}_{random_number}"


def create_sample_evaluation_data():
    """Create sample evaluation data for download"""
    sample_data = {
        'Question': [
            'What is the company refund policy?',
            'How do I reset my password?',
            'What are the system requirements for the software?',
            'Can you explain the pricing tiers?',
            'How do I contact customer support?'
        ],
        'Ground truth': [
            'The company offers a 30-day money-back guarantee on all products. Customers can request a full refund within 30 days of purchase by contacting customer support with their order number.',
            'To reset your password, click on the "Forgot Password" link on the login page. Enter your registered email address and follow the instructions sent to your email to create a new password.',
            'The minimum system requirements are: Windows 10 or macOS 10.14, 8GB RAM, 2GHz processor, and 10GB free disk space. For optimal performance, we recommend 16GB RAM and an SSD.',
            'We offer three pricing tiers: Basic ($9/month) with core features, Professional ($29/month) with advanced features and priority support, and Enterprise (custom pricing) with unlimited access and dedicated support.',
            'You can contact customer support via email at support@company.com, through the live chat on our website (available 9 AM - 6 PM EST), or by calling 1-800-XXX-XXXX during business hours.'
        ],
        'Chat history': [
            '',
            'User: I forgot my login credentials. Assistant: I can help you with that. Are you having trouble with your username or password?',
            'User: I want to install your software. Assistant: Great! I can help you with the installation process.',
            '',
            'User: I have been having issues with my account. Assistant: I understand you\'re experiencing issues. Could you tell me more about the specific problem?'
        ]
    }
    return pd.DataFrame(sample_data)


def create_sample_questions_data():
    """Create sample questions data for RAG generation"""
    sample_data = {
        'Question': [
            'What is the company refund policy?',
            'How do I reset my password?',
            'What are the system requirements for the software?',
            'Can you explain the pricing tiers?',
            'How do I contact customer support?'
        ]
    }
    return pd.DataFrame(sample_data)

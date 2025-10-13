# monitoring/usage_analytics.py
#USAGE_ANALYTICS = '''
"""
Comprehensive Usage Analytics and User Behavior Analysis
"""

import sqlite3
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
from collections import defaultdict
from dataclasses import dataclass
import plotly.express as px
import plotly.graph_objects as go

@dataclass
class UserSession:
    session_id: str
    user_id: str
    start_time: datetime
    end_time: Optional[datetime]
    actions_count: int
    pages_visited: List[str]
    duration_seconds: Optional[float]

class UsageAnalyticsEngine:
    """
    Advanced usage analytics for medical AI system
    """
    
    def __init__(self, analytics_db: str = "usage_analytics.db"):
        self.analytics_db = analytics_db
        self._initialize_database()
        self.active_sessions = {}
        
        logging.info("✅ Usage analytics engine initialized")
    
    def _initialize_database(self):
        """Initialize analytics database"""
        with sqlite3.connect(self.analytics_db) as conn:
            cursor = conn.cursor()
            
            # User sessions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    start_time TEXT,
                    end_time TEXT,
                    actions_count INTEGER,
                    pages_visited TEXT,
                    duration_seconds REAL,
                    user_agent TEXT,
                    ip_address TEXT
                )
            ''')
            
            # Page views table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS page_views (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    user_id TEXT,
                    page_path TEXT,
                    timestamp TEXT,
                    time_on_page REAL
                )
            ''')
            
            # Feature usage table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS feature_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    feature_name TEXT,
                    usage_count INTEGER,
                    last_used TEXT,
                    avg_usage_duration REAL
                )
            ''')
            
            conn.commit()
    
    def track_page_view(self, session_id: str, user_id: str, page_path: str):
        """Track page view event"""
        try:
            with sqlite3.connect(self.analytics_db) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO page_views (session_id, user_id, page_path, timestamp)
                    VALUES (?, ?, ?, ?)
                ''', (session_id, user_id, page_path, datetime.now().isoformat()))
                conn.commit()
        except Exception as e:
            logging.error(f"❌ Failed to track page view: {str(e)}")
    
    def get_usage_dashboard_data(self, days: int = 30) -> Dict[str, Any]:
        """Get comprehensive usage data for dashboard"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            with sqlite3.connect(self.analytics_db) as conn:
                # Daily active users
                dau_df = pd.read_sql_query('''
                    SELECT DATE(timestamp) as date, COUNT(DISTINCT user_id) as active_users
                    FROM page_views 
                    WHERE timestamp >= ? AND timestamp <= ?
                    GROUP BY DATE(timestamp)
                    ORDER BY date
                ''', conn, params=(start_date.isoformat(), end_date.isoformat()))
                
                # Most visited pages
                pages_df = pd.read_sql_query('''
                    SELECT page_path, COUNT(*) as visits
                    FROM page_views 
                    WHERE timestamp >= ? AND timestamp <= ?
                    GROUP BY page_path
                    ORDER BY visits DESC
                    LIMIT 10
                ''', conn, params=(start_date.isoformat(), end_date.isoformat()))
                
                return {
                    'daily_active_users': dau_df.to_dict('records'),
                    'most_visited_pages': pages_df.to_dict('records'),
                    'total_users': len(dau_df['active_users'].sum()) if len(dau_df) > 0 else 0,
                    'period_start': start_date.isoformat(),
                    'period_end': end_date.isoformat()
                }
                
        except Exception as e:
            logging.error(f"❌ Failed to get usage dashboard data: {str(e)}")
            return {'error': str(e)}

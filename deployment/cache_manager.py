# deployment/cache_manager.py
#CACHE_MANAGER = '''
"""
Intelligent Caching System for API Responses and Model Predictions
"""

import sqlite3
import json
import hashlib
import pickle
import time
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import logging
import threading
from functools import wraps

class IntelligentCacheManager:
    """
    Multi-level caching system with TTL and intelligent invalidation
    """
    
    def __init__(self, cache_db: str = "cache_system.db", max_memory_cache: int = 1000):
        self.cache_db = cache_db
        self.max_memory_cache = max_memory_cache
        
        # In-memory cache for fastest access
        self.memory_cache = {}
        self.cache_access_times = {}
        self.cache_lock = threading.RLock()
        
        self._initialize_database()
        self._start_cleanup_thread()
        
        logging.info("✅ Intelligent cache manager initialized")
    
    def _initialize_database(self):
        """Initialize cache database"""
        with sqlite3.connect(self.cache_db) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS cache_entries (
                    cache_key TEXT PRIMARY KEY,
                    cache_value BLOB,
                    created_at TEXT,
                    expires_at TEXT,
                    access_count INTEGER DEFAULT 0,
                    last_accessed TEXT,
                    cache_type TEXT,
                    metadata TEXT
                )
            ''')
            
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_expires_at ON cache_entries(expires_at)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_cache_type ON cache_entries(cache_type)')
            
            conn.commit()
    
    def cache_response(self, cache_type: str = "api_response", ttl_seconds: int = 3600):
        """Decorator for caching function responses"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Generate cache key
                cache_key = self._generate_cache_key(func.__name__, args, kwargs)
                
                # Try to get from cache
                cached_result = self.get(cache_key)
                if cached_result is not None:
                    return cached_result
                
                # Execute function and cache result
                result = func(*args, **kwargs)
                self.set(cache_key, result, ttl_seconds, cache_type)
                
                return result
            return wrapper
        return decorator
    
    def set(self, key: str, value: Any, ttl_seconds: int = 3600, cache_type: str = "general"):
        """Set cache entry with TTL"""
        try:
            expires_at = datetime.now() + timedelta(seconds=ttl_seconds)
            
            with self.cache_lock:
                # Add to memory cache if space available
                if len(self.memory_cache) < self.max_memory_cache:
                    self.memory_cache[key] = {
                        'value': value,
                        'expires_at': expires_at,
                        'cache_type': cache_type
                    }
                    self.cache_access_times[key] = datetime.now()
                
                # Add to database cache
                with sqlite3.connect(self.cache_db) as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT OR REPLACE INTO cache_entries 
                        (cache_key, cache_value, created_at, expires_at, cache_type, last_accessed, access_count)
                        VALUES (?, ?, ?, ?, ?, ?, 0)
                    ''', (
                        key,
                        pickle.dumps(value),
                        datetime.now().isoformat(),
                        expires_at.isoformat(),
                        cache_type,
                        datetime.now().isoformat()
                    ))
                    conn.commit()
                    
        except Exception as e:
            logging.error(f"❌ Failed to set cache entry: {str(e)}")
    
    def get(self, key: str) -> Optional[Any]:
        """Get cache entry with automatic expiration"""
        try:
            current_time = datetime.now()
            
            with self.cache_lock:
                # Check memory cache first
                if key in self.memory_cache:
                    entry = self.memory_cache[key]
                    if current_time < entry['expires_at']:
                        self.cache_access_times[key] = current_time
                        return entry['value']
                    else:
                        # Expired - remove from memory cache
                        del self.memory_cache[key]
                        if key in self.cache_access_times:
                            del self.cache_access_times[key]
                
                # Check database cache
                with sqlite3.connect(self.cache_db) as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        SELECT cache_value, expires_at FROM cache_entries 
                        WHERE cache_key = ?
                    ''', (key,))
                    
                    result = cursor.fetchone()
                    if result:
                        cache_value, expires_at_str = result
                        expires_at = datetime.fromisoformat(expires_at_str)
                        
                        if current_time < expires_at:
                            # Update access statistics
                            cursor.execute('''
                                UPDATE cache_entries 
                                SET access_count = access_count + 1, last_accessed = ?
                                WHERE cache_key = ?
                            ''', (current_time.isoformat(), key))
                            conn.commit()
                            
                            # Load into memory cache if space available
                            value = pickle.loads(cache_value)
                            if len(self.memory_cache) < self.max_memory_cache:
                                cursor.execute('SELECT cache_type FROM cache_entries WHERE cache_key = ?', (key,))
                                cache_type = cursor.fetchone()[0]
                                
                                self.memory_cache[key] = {
                                    'value': value,
                                    'expires_at': expires_at,
                                    'cache_type': cache_type
                                }
                                self.cache_access_times[key] = current_time
                            
                            return value
                        else:
                            # Expired - remove from database
                            cursor.execute('DELETE FROM cache_entries WHERE cache_key = ?', (key,))
                            conn.commit()
                
                return None
                
        except Exception as e:
            logging.error(f"❌ Failed to get cache entry: {str(e)}")
            return None
    
    def invalidate(self, pattern: str = None, cache_type: str = None):
        """Invalidate cache entries by pattern or type"""
        try:
            with self.cache_lock:
                # Clear memory cache
                keys_to_remove = []
                for key, entry in self.memory_cache.items():
                    if cache_type and entry['cache_type'] == cache_type:
                        keys_to_remove.append(key)
                    elif pattern and pattern in key:
                        keys_to_remove.append(key)
                
                for key in keys_to_remove:
                    del self.memory_cache[key]
                    if key in self.cache_access_times:
                        del self.cache_access_times[key]
                
                # Clear database cache
                with sqlite3.connect(self.cache_db) as conn:
                    cursor = conn.cursor()
                    
                    if cache_type:
                        cursor.execute('DELETE FROM cache_entries WHERE cache_type = ?', (cache_type,))
                    elif pattern:
                        cursor.execute('DELETE FROM cache_entries WHERE cache_key LIKE ?', (f'%{pattern}%',))
                    else:
                        cursor.execute('DELETE FROM cache_entries')
                    
                    conn.commit()
                    
        except Exception as e:
            logging.error(f"❌ Failed to invalidate cache: {str(e)}")
    
    def _generate_cache_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Generate deterministic cache key"""
        key_data = f"{func_name}_{str(args)}_{str(sorted(kwargs.items()))}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _start_cleanup_thread(self):
        """Start background thread for cache cleanup"""
        cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        cleanup_thread.start()
    
    def _cleanup_worker(self):
        """Background worker for cache cleanup"""
        while True:
            try:
                self._cleanup_expired_entries()
                self._manage_memory_cache_size()
                time.sleep(300)  # Run every 5 minutes
            except Exception as e:
                logging.error(f"❌ Cache cleanup error: {str(e)}")
                time.sleep(60)
    
    def _cleanup_expired_entries(self):
        """Remove expired entries from database"""
        try:
            current_time = datetime.now().isoformat()
            
            with sqlite3.connect(self.cache_db) as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM cache_entries WHERE expires_at < ?', (current_time,))
                deleted_count = cursor.rowcount
                conn.commit()
                
                if deleted_count > 0:
                    logging.info(f"🗑️ Cleaned up {deleted_count} expired cache entries")
                    
        except Exception as e:
            logging.error(f"❌ Failed to cleanup expired entries: {str(e)}")
    
    def _manage_memory_cache_size(self):
        """Manage memory cache size using LRU eviction"""
        try:
            with self.cache_lock:
                if len(self.memory_cache) > self.max_memory_cache:
                    # Sort by last access time and remove oldest entries
                    sorted_keys = sorted(
                        self.cache_access_times.keys(),
                        key=lambda k: self.cache_access_times[k]
                    )
                    
                    keys_to_remove = sorted_keys[:len(self.memory_cache) - self.max_memory_cache]
                    
                    for key in keys_to_remove:
                        if key in self.memory_cache:
                            del self.memory_cache[key]
                        if key in self.cache_access_times:
                            del self.cache_access_times[key]
                            
        except Exception as e:
            logging.error(f"❌ Failed to manage memory cache size: {str(e)}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            with sqlite3.connect(self.cache_db) as conn:
                cursor = conn.cursor()
                
                # Total entries
                cursor.execute('SELECT COUNT(*) FROM cache_entries')
                total_entries = cursor.fetchone()[0]
                
                # Entries by type
                cursor.execute('''
                    SELECT cache_type, COUNT(*) 
                    FROM cache_entries 
                    GROUP BY cache_type
                ''')
                entries_by_type = dict(cursor.fetchall())
                
                # Hit rate (would need to track hits/misses)
                memory_cache_size = len(self.memory_cache)
                
                return {
                    'total_entries': total_entries,
                    'memory_cache_size': memory_cache_size,
                    'max_memory_cache': self.max_memory_cache,
                    'entries_by_type': entries_by_type,
                    'memory_usage_percent': (memory_cache_size / self.max_memory_cache) * 100
                }
                
        except Exception as e:
            logging.error(f"❌ Failed to get cache stats: {str(e)}")
            return {'error': str(e)}
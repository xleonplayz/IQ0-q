"""Thread safety utilities for the NV simulator.

This module provides utilities for ensuring thread safety in the simulator,
including decorators and context managers for standardized locking patterns.
"""

import functools
import threading
import logging

# Configure module logger
logger = logging.getLogger(__name__)

def thread_safe(method):
    """
    Decorator to make a method thread-safe by using the instance's lock.
    
    The decorated class must have a 'lock' attribute that is a threading.Lock
    or threading.RLock instance.
    
    Parameters
    ----------
    method : callable
        The method to decorate
        
    Returns
    -------
    callable
        Thread-safe wrapper around the method
    """
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, 'lock'):
            logger.warning(
                f"Thread-safe decorator used on {method.__name__} but no lock "
                f"attribute found in {self.__class__.__name__}"
            )
            return method(self, *args, **kwargs)
            
        with self.lock:
            return method(self, *args, **kwargs)
    
    return wrapper


class ThreadSafeSingleton:
    """
    Base class for implementing a thread-safe singleton pattern.
    
    This class provides a thread-safe implementation of the singleton pattern,
    ensuring that only one instance of a class exists and that access to it
    is properly synchronized.
    
    Usage:
    ------
    class MySingleton(ThreadSafeSingleton):
        def __init__(self, config=None):
            super().__init__()
            # Additional initialization
    """
    
    _instances = {}
    _lock = threading.RLock()
    
    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                cls._instances[cls] = super(ThreadSafeSingleton, cls).__new__(cls)
                cls._instances[cls]._initialized = False
            return cls._instances[cls]
    
    def __init__(self):
        with self.__class__._lock:
            if getattr(self, '_initialized', False):
                return
                
            self._initialized = True
            self.lock = threading.RLock()


class ResourceMonitor:
    """
    Monitor and manage resource usage for large simulations.
    
    This class provides utilities for monitoring memory usage, scheduling
    garbage collection, and handling low-memory situations.
    """
    
    def __init__(self, max_memory_gb=4.0, cleanup_interval=10, 
                 warning_threshold=0.8, critical_threshold=0.95):
        """
        Initialize the resource monitor.
        
        Parameters
        ----------
        max_memory_gb : float
            Maximum memory usage in gigabytes
        cleanup_interval : int
            Number of operations between scheduled cleanups
        warning_threshold : float
            Memory usage fraction that triggers a warning
        critical_threshold : float
            Memory usage fraction that triggers emergency cleanup
        """
        self.max_memory_gb = max_memory_gb
        self.cleanup_interval = cleanup_interval
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        
        self._op_count = 0
        self.lock = threading.RLock()
    
    def record_operation(self):
        """
        Record an operation and perform cleanup if needed.
        
        Returns
        -------
        bool
            True if cleanup was performed, False otherwise
        """
        with self.lock:
            self._op_count += 1
            if self._op_count >= self.cleanup_interval:
                self._op_count = 0
                return self.check_and_cleanup()
            return False
    
    def check_and_cleanup(self):
        """
        Check memory usage and perform cleanup if necessary.
        
        Returns
        -------
        bool
            True if cleanup was performed, False otherwise
        """
        try:
            import psutil
            
            # Get current memory usage
            process = psutil.Process()
            memory_gb = process.memory_info().rss / (1024**3)
            
            if memory_gb > self.critical_threshold * self.max_memory_gb:
                logger.warning(
                    f"Critical memory usage: {memory_gb:.2f} GB "
                    f"(limit: {self.max_memory_gb:.2f} GB). "
                    f"Performing emergency cleanup."
                )
                self.emergency_cleanup()
                return True
                
            elif memory_gb > self.warning_threshold * self.max_memory_gb:
                logger.info(
                    f"High memory usage: {memory_gb:.2f} GB "
                    f"(limit: {self.max_memory_gb:.2f} GB). "
                    f"Performing standard cleanup."
                )
                self.standard_cleanup()
                return True
                
            return False
            
        except ImportError:
            logger.warning("psutil not available, skipping memory check")
            return False
    
    def standard_cleanup(self):
        """
        Perform standard cleanup operations.
        """
        import gc
        gc.collect()
    
    def emergency_cleanup(self):
        """
        Perform emergency cleanup when memory is critically low.
        """
        import gc
        
        # Force full garbage collection
        for i in range(3):  # Collect all generations
            gc.collect(i)
            
        # Attempt to clear any caches
        # This is a hook for subclasses to implement
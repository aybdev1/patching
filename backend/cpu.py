from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import psutil
import time
from fastapi import APIRouter
 
router = APIRouter()


router = APIRouter()

def get_cpu_info():
    cpu_percent_per_core = psutil.cpu_percent(percpu=True)
    avg_cpu_usage = sum(cpu_percent_per_core) / len(cpu_percent_per_core)
    freq = psutil.cpu_freq()._asdict()
    stats = psutil.cpu_stats()._asdict()
    times = psutil.cpu_times()._asdict()
    return {
        "cpu_usage_per_core": cpu_percent_per_core,
        "avg_cpu_usage": avg_cpu_usage,
        "cpu_freq": freq,
        "cpu_stats": stats,
        "cpu_times": times,
        "cpu_count": psutil.cpu_count(logical=True)
    }

@router.get("/api/cpu")
def read_cpu_info():
    """Return real-time CPU information."""
    return get_cpu_info()
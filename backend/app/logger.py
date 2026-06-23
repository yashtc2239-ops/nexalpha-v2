"""
Structured JSON logging via loguru.
Why structured (not print()): in production, logs get shipped to systems like
ELK / CloudWatch / Datadog that parse JSON. print() debugging dies the moment
you deploy to a server you can't SSH into and tail a file on.
"""
import sys
from loguru import logger

logger.remove()
logger.add(
    sys.stdout,
    serialize=True,  # JSON output
    level="INFO",
    backtrace=False,
    diagnose=False,
)

__all__ = ["logger"]

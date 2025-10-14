import os, logging


logger = logging.getLogger("Graph GPS Transforms")
logging.basicConfig(format="%(levelname)s %(asctime)s %(message)s", level=os.environ.get("GPS_LOG_LEVEL",
                                                                                         logging.DEBUG))

import logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger('adad')
logger2 = logging.getLogger(__name__)

logger.info('Start reading database')
# read database here
records = {'john': 55, 'tom': 66}
logger.debug('Records: %s', records)
logger.info('Updating records ...')
# update records here
logger.info('Finish updating records')
logger2.error('value error')
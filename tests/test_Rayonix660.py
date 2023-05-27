import logging

import sys
import psana
import pytest 

import smalldata_tools
from conftest import datasource, detector

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

logger.info('Loading detector: Rayonix660')

@pytest.mark.parametrize('datasource', [{'exp': 'xpptut15', 'run': 660}], indirect=True)
@pytest.mark.parametrize('detector', [{'name': 'Rayonix660'}], indirect=True)
def test_detector_type(datasource, detector):
    logger.debug('Running detector type test')
    det = detector
    assert(isinstance(det, smalldata_tools.DetObject.RayonixObject))
    logger.debug('Pass the test')
    
